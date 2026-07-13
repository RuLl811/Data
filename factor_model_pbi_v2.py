# -*- coding: utf-8 -*-
"""
Modelo de factores para nowcast de PBI — versión corregida (v2)
================================================================
Correcciones vs v1 (auditoría 2026-07-13):
  [C1]  Selección de features DENTRO de cada ventana (elimina look-ahead bias).
  [C2]  Ventana EXPANSIVA (min 24 trimestres) en lugar de rolling 12q.
  [C3]  COVID: dummy 2020Q1-2021Q2 en la regresión + winsorización del panel,
        en lugar de eliminar observaciones (preserva continuidad temporal).
  [C4]  Número de factores por criterio Bai-Ng ICp2 (re-estimado por ventana),
        con robustez para k fijo en {1,2,3}.
  [C5]  Mapa de transformaciones por tipo de serie (tasas/spreads -> diff;
        valores <=0 enmascarados antes del log). Sin NaN silenciosos.
  [C6]  Benchmarks (media histórica expansiva, AR(1)) + OOS R^2 Campbell-Thompson
        + test Diebold-Mariano con corrección Harvey-Leybourne-Newbold.
  [C7]  Intervalo de PREDICCIÓN (obs_ci) en el nowcast puntual, no IC de la media.
  [C8]  Bug future_ix: nowcast del PRIMER trimestre futuro (future_ix[0]).
  [C9]  Sin warnings.filterwarnings global; sin código ejecutable a nivel módulo;
        sin reindex(fill_value=0) silencioso (assert de columnas); APIs pandas 2.2+/3.0
        ('QE', .ffill(), .iloc); I/O y plotting separados de la lógica.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import PCA

# =====================
# Configuración
# =====================
@dataclass
class Config:
    path_excel: str = "/mnt/user-data/uploads/base_PBI.xlsx"
    sheet_name: str = "base_desest"
    date_col: str = "Date"
    pib_col: str = "DPBI_desest"          # ya en diferencias (no transformar)
    # [C5] transformaciones por tipo de serie (default: log_diff)
    diff_cols: tuple = ("badlar_TNA", "EMBI")   # tasas / spreads -> diff simple
    min_corr: float = 0.45                # umbral de |corr| con y (dentro de ventana)
    fallback_top_n: int = 15              # si nada supera el umbral
    min_train_quarters: int = 24          # [C2] mínimo para ventana expansiva
    max_factors_ic: int = 6               # máximo k evaluado por Bai-Ng
    winsor_z: float = 4.0                 # [C3] clip del panel estandarizado
    covid_start: str = "2020-01-01"
    covid_end: str = "2021-06-30"

# =====================
# Bloque 1 — Carga
# =====================
def load_base(cfg: Config) -> pd.DataFrame:
    df = pd.read_excel(cfg.path_excel, sheet_name=cfg.sheet_name)
    df[cfg.date_col] = pd.to_datetime(df[cfg.date_col])
    df = df.set_index(cfg.date_col).sort_index()
    # [C3] NO se eliminan 2020-2021: se tratan con dummy + winsorización
    return df

# =====================
# Bloque 2 — Transformaciones [C5]
# =====================
def transform_series(df: pd.DataFrame, cfg: Config) -> tuple[pd.Series, pd.DataFrame]:
    y = df[cfg.pib_col].groupby(pd.Grouper(freq="QE")).last()
    y.name = "gdp_qoq"
    y = y.dropna()

    X = df.drop(columns=[cfg.pib_col]).copy()
    Xt = pd.DataFrame(index=X.index)
    for col in X.columns:
        s = X[col]
        if col in cfg.diff_cols:
            Xt[col] = s.diff()
        else:
            # log_diff con máscara explícita de no-positivos (NaN declarado, no silencioso)
            s_pos = s.where(s > 0)
            n_masked = int((s <= 0).sum())
            if n_masked > 0:
                print(f"[transform] {col}: {n_masked} obs <=0 enmascaradas antes del log")
            Xt[col] = np.log(s_pos).diff()

    Xq = Xt.groupby(pd.Grouper(freq="QE")).mean()          # media trimestral (skipna)
    # y se restringe a trimestres con dato de PBI; Xq conserva trimestres
    # posteriores (necesarios para el nowcast del trimestre en curso)
    y = y.loc[y.index.intersection(Xq.index)]
    return y, Xq

def covid_dummy(index: pd.DatetimeIndex, cfg: Config) -> pd.Series:
    d = pd.Series(0.0, index=index, name="covid")
    d.loc[(index >= cfg.covid_start) & (index <= cfg.covid_end)] = 1.0
    return d

# =====================
# Bloque 3 — Selección de features (solo con datos de train) [C1]
# =====================
def select_features(y_tr: pd.Series, X_tr: pd.DataFrame, cfg: Config) -> list[str]:
    Z = X_tr.ffill()
    corrs = Z.corrwith(y_tr).abs().sort_values(ascending=False).dropna()
    keep = corrs[corrs >= cfg.min_corr].index.tolist()
    if len(keep) == 0:
        keep = corrs.head(cfg.fallback_top_n).index.tolist()
    return keep

# =====================
# Bloque 4 — PCA + Bai-Ng ICp2 [C4]
# =====================
def bai_ng_icp2(Z_std: np.ndarray, kmax: int) -> int:
    """ICp2 de Bai & Ng (2002) sobre panel estandarizado T x N."""
    T, N = Z_std.shape
    kmax = min(kmax, N, T - 1)
    pca = PCA(n_components=kmax).fit(Z_std)
    F = pca.transform(Z_std)
    L = pca.components_
    ics = []
    for k in range(1, kmax + 1):
        resid = Z_std - F[:, :k] @ L[:k, :]
        V = np.mean(resid ** 2)
        penalty = k * ((N + T) / (N * T)) * np.log(min(N, T))
        ics.append(np.log(V) + penalty)
    return int(np.argmin(ics)) + 1

def fit_project_pca(Z_tr: pd.DataFrame, Z_te: pd.DataFrame, cfg: Config,
                    k_fixed: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Estandariza y winsoriza con stats de TRAIN, ajusta PCA en train, proyecta test."""
    mu, sd = Z_tr.mean(axis=0), Z_tr.std(axis=0, ddof=0).replace(0, 1.0)
    Ztr = ((Z_tr - mu) / sd).clip(-cfg.winsor_z, cfg.winsor_z)     # [C3] winsor
    Zte = ((Z_te - mu) / sd).clip(-cfg.winsor_z, cfg.winsor_z)

    k = k_fixed if k_fixed is not None else bai_ng_icp2(Ztr.values, cfg.max_factors_ic)
    pca = PCA(n_components=k).fit(Ztr.values)
    cols = [f"F{i+1}" for i in range(k)]
    F_tr = pd.DataFrame(pca.transform(Ztr.values), index=Z_tr.index, columns=cols)
    F_te = pd.DataFrame(pca.transform(Zte.values), index=Z_te.index, columns=cols)
    return F_tr, F_te, k

# =====================
# Bloque 5 — Ecuación puente (OLS + HAC con maxlags Newey-West)
# =====================
def hac_maxlags(T: int) -> int:
    return max(1, int(np.floor(4 * (T / 100.0) ** (2.0 / 9.0))))

def bridge_ols(y_tr: pd.Series, F_tr: pd.DataFrame, d_tr: pd.Series):
    X = F_tr.copy()
    if d_tr.abs().sum() > 0:                 # dummy solo si hay obs COVID en train
        X["covid"] = d_tr
    X = sm.add_constant(X, has_constant="add")
    D = pd.concat([y_tr, X], axis=1).dropna()
    model = sm.OLS(D.iloc[:, 0], D.iloc[:, 1:]).fit(
        cov_type="HAC", cov_kwds={"maxlags": hac_maxlags(len(D))}
    )
    return model

def predict_bridge(model, F_te: pd.DataFrame, d_te: float) -> float:
    X = F_te.copy()
    if "covid" in model.model.exog_names:
        X["covid"] = d_te
    X = sm.add_constant(X, has_constant="add")
    # [C9] sin fill_value=0 silencioso: exigimos coincidencia exacta de columnas
    missing = set(model.model.exog_names) - set(X.columns)
    assert not missing, f"Regresores faltantes en X_test: {missing}"
    X = X[model.model.exog_names]
    return float(model.predict(X).iloc[0])

# =====================
# Bloque 6 — OOS expansivo con benchmarks [C1][C2][C6]
# =====================
def expanding_nowcast(y: pd.Series, Xq: pd.DataFrame, cfg: Config,
                      k_fixed: int | None = None) -> pd.DataFrame:
    idx = y.index
    dummy = covid_dummy(idx, cfg)
    rows = []
    for t in range(cfg.min_train_quarters, len(idx)):
        tr_idx, te_idx = idx[:t], idx[t]

        # [C1] pipeline íntegramente recursivo: selección + PCA + OLS con train only
        keep = select_features(y.loc[tr_idx], Xq.loc[tr_idx], cfg)
        Z_tr = Xq.loc[tr_idx, keep].ffill()
        Z_te = Xq.loc[[te_idx], keep]
        # última info disponible si el trimestre test tiene NaN en alguna serie
        Z_te = Z_te.fillna(Z_tr.iloc[-1])

        F_tr, F_te, k = fit_project_pca(Z_tr, Z_te, cfg, k_fixed=k_fixed)
        model = bridge_ols(y.loc[tr_idx], F_tr, dummy.loc[tr_idx])
        y_hat = predict_bridge(model, F_te, float(dummy.loc[te_idx]))

        # Benchmarks estimados con la MISMA información (train hasta t-1)
        bench_mean = float(y.loc[tr_idx].mean())
        ar = bridge_ols(y.loc[tr_idx],
                        y.loc[tr_idx].shift(1).to_frame("y_L1"),
                        dummy.loc[tr_idx])
        Xar = pd.DataFrame({"y_L1": [float(y.loc[tr_idx].iloc[-1])]}, index=[te_idx])
        y_ar = predict_bridge(ar, Xar, float(dummy.loc[te_idx]))

        rows.append({"date": te_idx, "y_true": float(y.loc[te_idx]),
                     "y_hat": y_hat, "y_mean": bench_mean, "y_ar1": y_ar,
                     "k": k, "n_vars": len(keep), "covid": float(dummy.loc[te_idx])})
    out = pd.DataFrame(rows).set_index("date")
    for c in ["y_hat", "y_mean", "y_ar1"]:
        out[f"err_{c[2:]}"] = out["y_true"] - out[c]
    return out

# =====================
# Bloque 7 — Evaluación: OOS R^2 + Diebold-Mariano (HLN) [C6]
# =====================
def dm_test_hln(e1: np.ndarray, e2: np.ndarray, h: int = 1) -> tuple[float, float]:
    """DM con corrección de muestra chica de Harvey-Leybourne-Newbold.
    H0: igual precisión. Loss cuadrático. Devuelve (stat, p-valor, dist t_{T-1})."""
    from scipy import stats
    d = e1 ** 2 - e2 ** 2
    T = len(d)
    dbar = d.mean()
    # varianza HAC de dbar (Newey-West, lags h-1; para h=1 es var muestral)
    gamma0 = np.mean((d - dbar) ** 2)
    var_dbar = gamma0
    for lag in range(1, h):
        g = np.mean((d[lag:] - dbar) * (d[:-lag] - dbar))
        var_dbar += 2 * (1 - lag / h) * g
    dm = dbar / np.sqrt(var_dbar / T)
    hln = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    stat = hln * dm
    p = 2 * (1 - stats.t.cdf(abs(stat), df=T - 1))
    return float(stat), float(p)

def evaluate(fcst: pd.DataFrame, label: str = "") -> pd.DataFrame:
    res = {}
    for name, col in [("Modelo", "err_hat"), ("Media", "err_mean"), ("AR(1)", "err_ar1")]:
        e = fcst[col].dropna()
        res[name] = {"RMSE": np.sqrt((e ** 2).mean()),
                     "MAE": e.abs().mean(), "Bias": e.mean()}
    tab = pd.DataFrame(res).T
    # OOS R^2 (Campbell-Thompson) contra media expansiva
    ss_model = (fcst["err_hat"] ** 2).sum()
    ss_mean = (fcst["err_mean"] ** 2).sum()
    ss_ar = (fcst["err_ar1"] ** 2).sum()
    r2_mean = 1 - ss_model / ss_mean
    r2_ar = 1 - ss_model / ss_ar
    dm_m, p_m = dm_test_hln(fcst["err_hat"].values, fcst["err_mean"].values)
    dm_a, p_a = dm_test_hln(fcst["err_hat"].values, fcst["err_ar1"].values)
    print(f"\n===== Evaluación OOS {label} (n={len(fcst)}) =====")
    print(tab.round(4).to_string())
    print(f"OOS R2 vs media histórica : {r2_mean:+.3f}  | DM-HLN stat {dm_m:+.2f}, p={p_m:.3f}")
    print(f"OOS R2 vs AR(1)           : {r2_ar:+.3f}  | DM-HLN stat {dm_a:+.2f}, p={p_a:.3f}")
    return tab

# =====================
# Bloque 8 — Nowcast del próximo trimestre [C7][C8]
# =====================
def predict_next_quarter(y: pd.Series, Xq: pd.DataFrame, cfg: Config, alpha: float = 0.10):
    last_y = y.index.max()
    future_ix = Xq.index[Xq.index > last_y]
    assert len(future_ix) > 0, "No hay trimestres futuros en Xq."
    test_idx = future_ix[0]                                   # [C8] PRIMER trimestre futuro

    dummy = covid_dummy(y.index, cfg)
    keep = select_features(y, Xq.loc[y.index], cfg)
    Z_tr = Xq.loc[y.index, keep].ffill()
    Z_te = Xq.loc[[test_idx], keep].fillna(Z_tr.iloc[-1])

    F_tr, F_te, k = fit_project_pca(Z_tr, Z_te, cfg)
    model = bridge_ols(y, F_tr, dummy)

    X = F_te.copy()
    if "covid" in model.model.exog_names:
        X["covid"] = 0.0
    X = sm.add_constant(X, has_constant="add")[model.model.exog_names]
    sf = model.get_prediction(X).summary_frame(alpha=alpha)   # [C7] obs_ci = pred. interval
    n_months = Xq.loc[test_idx].notna().sum()                 # transparencia sobre ragged edge
    return {"date": test_idx, "k": k, "n_vars": len(keep),
            "y_hat_pct": 100 * float(sf["mean"].iloc[0]),
            "pi_low_pct": 100 * float(sf["obs_ci_lower"].iloc[0]),
            "pi_high_pct": 100 * float(sf["obs_ci_upper"].iloc[0]),
            "months_info": f"{Xq.loc[[test_idx], keep].notna().sum(axis=1).iloc[0]}/{len(keep)} series con dato en el trimestre"}

# =====================
# Ejecución [C9]
# =====================
def main():
    cfg = Config()
    df = load_base(cfg)
    y, Xq = transform_series(df, cfg)
    print(f"Muestra: {y.index.min().date()} → {y.index.max().date()} | {len(y)} trimestres")

    # Headline: k por Bai-Ng ICp2 por ventana
    fcst = expanding_nowcast(y, Xq, cfg, k_fixed=None)
    print(f"\nk elegido por ICp2 (distribución OOS): {fcst['k'].value_counts().to_dict()}")
    print(f"n variables seleccionadas por ventana: min {fcst['n_vars'].min()}, "
          f"max {fcst['n_vars'].max()}")

    evaluate(fcst, "— muestra completa")
    evaluate(fcst[fcst["covid"] == 0], "— ex-COVID (headline)")

    # Robustez: k fijo
    for k in (1, 2, 3):
        f_k = expanding_nowcast(y, Xq, cfg, k_fixed=k)
        f_k_ex = f_k[f_k["covid"] == 0]
        e = f_k_ex["err_hat"]
        r2 = 1 - (e ** 2).sum() / (f_k_ex["err_mean"] ** 2).sum()
        print(f"[robustez] k={k}: RMSE ex-COVID {np.sqrt((e**2).mean()):.4f} | "
              f"OOS R2 vs media {r2:+.3f}")

    fcst.to_excel("/home/claude/forecast_results_v2.xlsx")

    # Nowcast próximo trimestre
    res = predict_next_quarter(y, Xq, cfg, alpha=0.10)
    print(f"\nNowcast {res['date'].date()} (k={res['k']}, {res['n_vars']} vars, "
          f"{res['months_info']}):")
    print(f"  {res['y_hat_pct']:+.2f}% t/t | PI 90%: "
          f"[{res['pi_low_pct']:+.2f}%, {res['pi_high_pct']:+.2f}%]")
    return y, Xq, fcst

if __name__ == "__main__":
    main()
