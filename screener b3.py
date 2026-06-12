"""
============================================================================
SCREENER DIARIO B3
============================================================================
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2

try:
    from sklearn.covariance import MinCovDet
    _HAS_MCD = True
except ImportError:  # fallback robusto si no hay sklearn
    _HAS_MCD = False

SIGNAL_COLS = ["volumen_z", "volregime_z", "resid_z", "mom_cross_z"]


# ===========================================================================
# CONFIGURACION
# ===========================================================================
@dataclass
class Config:
    # --- entrada / estado ---
    excel_path: str = "datos_b3.xlsx"
    sheet_close: str = "close"
    sheet_volume: str = "volume"
    index_col: str = "IBOV"          # header de la columna de mercado en "close"
    state_file: str = "screener_state.json"   # cooldown + log de alertas
    out_dir: str = "."

    # --- ventanas (dias habiles) ---
    vol_window: int = 30             # baseline anomalia de volumen
    vol_short: int = 5               # vol realizada corta
    vol_long: int = 60               # vol realizada larga
    beta_window: int = 180           # beta movil vs IBOV
    mom_window: int = 60             # retorno trailing para momentum cross-sectional
    rank_lookback: int = 5           # ventana para velocidad de ranking
    min_periods: int = 30

    # --- umbrales (|z| o ratio) ---
    thr_vol_z: float = 3.5
    thr_volregime_z: float = 1
    thr_resid_z: float = 1
    thr_mom_cross_z: float = 2.0

    # --- alertas ---
    min_signals: int = 1             # alertar si >= N senales disparan
    cooldown_days: int = 0           # no re-alertar el mismo nombre dentro de K dias
    use_cooldown: bool = True
    min_adtv_brl: float = 1_000_000  # liquidez minima (mediana de close*volume)
    adtv_window: int = 21

    use_mahalanobis: bool = True     # gate principal por D^2 (si False, usa n_signals)
    cov_window: int = 504            # panel para estimar Sigma
    maha_quantile: float = 0.75       # corte empirico del D^2 historico
    mcd_support_fraction: float = 0.75  # subconjunto concentrado para MCD

    top_n_plots: int = 10
    plot_lookback: int = 126
    save_plot: bool = True

# ===========================================================================
# CARGA DE DATOS
# ===========================================================================
def load_data(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    close = pd.read_excel(cfg.excel_path, sheet_name=cfg.sheet_close, index_col=0, skiprows=1)
    volume = pd.read_excel(cfg.excel_path, sheet_name=cfg.sheet_volume, index_col=0, skiprows=1)
    close.index = pd.to_datetime(close.index)
    volume.index = pd.to_datetime(volume.index)
    close = close.sort_index()
    volume = volume.sort_index()

    if cfg.index_col not in close.columns:
        raise KeyError(
            f"No encuentro la columna de indice '{cfg.index_col}' en la hoja "
            f"'{cfg.sheet_close}'. Columnas: {list(close.columns)}"
        )

    ibov = close[cfg.index_col].astype(float)
    close = close.drop(columns=[cfg.index_col]).astype(float)

    tickers = close.columns.intersection(volume.columns)
    if len(tickers) == 0:
        raise ValueError("Los tickers de 'close' y 'volume' no coinciden.")
    close = close[tickers]
    volume = volume[tickers].astype(float)

    common_dates = close.index.intersection(volume.index).intersection(ibov.index)
    return close.loc[common_dates], volume.loc[common_dates], ibov.loc[common_dates]

# ===========================================================================
# SENALES (todas vectorizadas columna a columna)
# ===========================================================================
def signal_volume_anomaly(volume: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """1) z-score (mediana/MAD) del log-volumen vs baseline movil."""
    log_vol = np.log(volume.replace(0, np.nan))
    center = log_vol.rolling(cfg.vol_window, min_periods=cfg.min_periods).median()
    abs_dev = (log_vol - center).abs()
    mad = abs_dev.rolling(cfg.vol_window, min_periods=cfg.min_periods).median()
    mad = mad.replace(0, np.nan)
    z = 0.6745 * (log_vol - center) / mad
    return z.replace([np.inf, -np.inf], np.nan)


def signal_vol_regime(rets: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """2) z-score del ratio vol_corta / vol_larga vs su propia historia.

    Detecta apertura de regimen de volatilidad. (Si tenes high/low en el Excel,
    reemplaza la vol realizada por ATR; el resto del pipeline no cambia.)
    """
    short_v = rets.rolling(cfg.vol_short, min_periods=cfg.vol_short).std()
    long_v = rets.rolling(cfg.vol_long, min_periods=cfg.min_periods).std()
    ratio = (short_v / long_v).replace([np.inf, -np.inf], np.nan)
    mu = ratio.rolling(cfg.vol_long, min_periods=cfg.min_periods).mean()
    sd = ratio.rolling(cfg.vol_long, min_periods=cfg.min_periods).std().replace(0, np.nan)
    return (ratio - mu) / sd


def signal_idio_dislocation(
    rets: pd.DataFrame, ibov_ret: pd.Series, cfg: Config
) -> pd.DataFrame:
    """3) z-score del retorno idiosincratico (residuo vs beta-IBOV movil).

    beta_i = Cov(r_i, r_m) / Var(r_m) en ventana movil (momentos poblacionales
    consistentes). Aisla el movimiento propio del nombre del beta de mercado.
    """
    w = cfg.beta_window
    m_i = rets.rolling(w, min_periods=cfg.min_periods).mean()
    m_m = ibov_ret.rolling(w, min_periods=cfg.min_periods).mean()
    e_cross = rets.mul(ibov_ret, axis=0).rolling(w, min_periods=cfg.min_periods).mean()
    cov = e_cross.sub(m_i.mul(m_m, axis=0))
    var_m = (ibov_ret**2).rolling(w, min_periods=cfg.min_periods).mean() - m_m**2
    beta = cov.div(var_m.replace(0, np.nan), axis=0)

    resid = rets.sub(beta.mul(ibov_ret, axis=0))
    resid_sd = resid.rolling(w, min_periods=cfg.min_periods).std().replace(0, np.nan)
    return resid / resid_sd


def signal_xsec_momentum_panel(close: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """4) z transversal del retorno trailing, fila a fila (todas las fechas)."""
    trail = close.pct_change(cfg.mom_window)
    mu = trail.mean(axis=1)
    sd = trail.std(axis=1, ddof=0).replace(0, np.nan)
    return trail.sub(mu, axis=0).div(sd, axis=0)


def signal_xsec_momentum(close: pd.DataFrame, cfg: Config) -> tuple[pd.Series, pd.Series]:
    mom_cross_z = signal_xsec_momentum_panel(close, cfg).iloc[-1]
    trail = close.pct_change(cfg.mom_window)
    rank_now = trail.iloc[-1].rank(ascending=True)
    rank_prev = trail.iloc[-1 - cfg.rank_lookback].rank(ascending=True)
    return mom_cross_z, rank_now - rank_prev


# ===========================================================================
# MAHALANOBIS
# ===========================================================================
def _robust_location_scatter(X: np.ndarray, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    p = X.shape[1]
    if _HAS_MCD and X.shape[0] >= 5 * p:
        try:
            mcd = MinCovDet(
                support_fraction=cfg.mcd_support_fraction, random_state=0
            ).fit(X)
            S = mcd.covariance_ + 1e-8 * np.eye(p)
            return mcd.location_, S
        except Exception:
            pass
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0) * 1.4826
    mad[mad == 0] = 1.0
    Xw = np.clip(X, med - 3 * mad, med + 3 * mad)
    S = np.cov(Xw, rowvar=False) + 1e-6 * np.eye(p)
    return med, S

def mahalanobis_composite(
    panels: dict[str, pd.DataFrame], cfg: Config
) -> tuple[pd.DataFrame, float]:
    cols = SIGNAL_COLS

    parts = [panels[c].iloc[-cfg.cov_window:].stack().rename(c) for c in cols]
    fit_df = pd.concat(parts, axis=1).dropna()
    mu, S = _robust_location_scatter(fit_df.values, cfg)
    S_inv = np.linalg.pinv(S)

    dz = fit_df.values - mu
    d2_hist = np.einsum("ij,jk,ik->i", dz, S_inv, dz)
    thr = float(np.quantile(d2_hist, cfg.maha_quantile))

    today = pd.concat([panels[c].iloc[-1].rename(c) for c in cols], axis=1)
    today_filled = today.fillna(0.0)              # NaN -> neutro (sesga D^2 a la baja)
    diff = today_filled.values - mu
    proj = diff @ S_inv                           # (Sigma^-1 z)_k por nombre
    contrib = diff * proj                         # contribucion de cada senal a D^2
    d2 = contrib.sum(axis=1)

    out = pd.DataFrame(index=today.index)
    out["maha_d2"] = d2
    out["maha_top"] = [cols[i] for i in np.argmax(contrib, axis=1)]  # senal dominante
    return out, thr


# ===========================================================================
# ENSAMBLE DE SENALES DE HOY + LIQUIDEZ
# ===========================================================================
def compute_today_table(
    close: pd.DataFrame, volume: pd.DataFrame, ibov: pd.Series, cfg: Config
) -> tuple[pd.DataFrame, float]:
    rets = np.log(close).diff()
    ibov_ret = np.log(ibov).diff()

    # paneles historicos completos (se reusan para hoy y para estimar Sigma)
    panels = {
        "volumen_z": signal_volume_anomaly(volume, cfg),
        "volregime_z": signal_vol_regime(rets, cfg),
        "resid_z": signal_idio_dislocation(rets, ibov_ret, cfg),
        "mom_cross_z": signal_xsec_momentum_panel(close, cfg),
    }
    _, rank_delta = signal_xsec_momentum(close, cfg)
    adtv = volume.rolling(cfg.adtv_window, min_periods=5).median().iloc[-1]

    tab = pd.DataFrame({c: panels[c].iloc[-1] for c in SIGNAL_COLS})
    tab["rank_delta"] = rank_delta
    tab["adtv_brl"] = adtv

    fired = pd.DataFrame(
        {
            "VOL": tab["volumen_z"].abs() >= cfg.thr_vol_z,
            "VOLREG": tab["volregime_z"].abs() >= cfg.thr_volregime_z,
            "IDIO": tab["resid_z"].abs() >= cfg.thr_resid_z,
            "MOM": tab["mom_cross_z"].abs() >= cfg.thr_mom_cross_z,
        }
    ).fillna(False)
    tab["n_signals"] = fired.sum(axis=1)
    tab["signals"] = fired.apply(lambda r: ",".join(fired.columns[r.values]), axis=1)

    # compuesto Mahalanobis
    maha, thr = mahalanobis_composite(panels, cfg)
    tab = tab.join(maha)
    tab["liquid"] = tab["adtv_brl"] >= cfg.min_adtv_brl
    return tab, thr


# ===========================================================================
# COOLDOWN / ESTADO PERSISTENTE
# ===========================================================================
def load_state(cfg: Config) -> dict:
    p = Path(cfg.out_dir) / cfg.state_file
    if p.exists():
        return json.loads(p.read_text())
    return {"last_alert": {}}


def save_state(cfg: Config, state: dict) -> None:
    p = Path(cfg.out_dir) / cfg.state_file
    p.write_text(json.dumps(state, indent=2, default=str))


def apply_cooldown(alerts: pd.DataFrame, asof: datetime, cfg: Config, state: dict) -> pd.DataFrame:
    if not cfg.use_cooldown:
        return alerts
    last = state.get("last_alert", {})
    keep = []
    for tk in alerts.index:
        prev = last.get(tk)
        if prev is None:
            keep.append(True)
        else:
            days = (asof - pd.to_datetime(prev)).days
            keep.append(days >= cfg.cooldown_days)
    return alerts[pd.Series(keep, index=alerts.index)]


# ===========================================================================
# OUTPUT: TERMINAL + GRAFICOS
# ===========================================================================
def print_terminal(alerts: pd.DataFrame, asof: datetime, cfg: Config) -> None:
    print("=" * 78)
    print(f"SCREENER B3  |  as-of {asof.date()}  |  {len(alerts)} nombres a monitorear")
    print("=" * 78)
    if alerts.empty:
        print("Sin alertas.")
        return

    cols = ["maha_d2", "maha_top", "n_signals", "signals",
            "volumen_z", "volregime_z", "resid_z", "mom_cross_z", "rank_delta"]
    view = alerts[cols].copy()
    fmt = {c: "{:+.2f}".format for c in ["volumen_z", "volregime_z", "resid_z", "mom_cross_z"]}
    fmt["rank_delta"] = "{:+.0f}".format
    fmt["maha_d2"] = "{:.2f}".format
    print(view.to_string(formatters=fmt))
    print("-" * 78)
    print("TICKERS:", " ".join(alerts.index.tolist()))
    print("=" * 78)

def plot_top(alerts: pd.DataFrame, close: pd.DataFrame, cfg: Config, asof: datetime) -> None:
    if alerts.empty:
        return
    top = alerts.head(cfg.top_n_plots)
    n = len(top)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.0 * nrows), squeeze=False)
    axes = axes.ravel()

    for ax, (tk, row) in zip(axes, top.iterrows()):
        s = close[tk].dropna().iloc[-cfg.plot_lookback:]
        ax.plot(s.index, s.values, lw=1.1)
        ax.scatter(s.index[-1], s.iloc[-1], color="red", zorder=5, s=28)
        ax.set_title(f"{tk}  [{row['maha_top']}]  D2={row['maha_d2']:.1f}", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(f"Top {n} - Screener Latam  ({asof.date()})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if cfg.save_plot:
        out = Path(cfg.out_dir) / f"screener_top_{asof.date()}.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        print(f"[plot] guardado en {out}")
    plt.show()

# ===========================================================================
# MAIN
# ===========================================================================
def run(cfg: Config) -> pd.DataFrame:
    close, volume, ibov = load_data(cfg)
    asof = close.index[-1].to_pydatetime()

    tab, maha_thr = compute_today_table(close, volume, ibov, cfg)

    if cfg.use_mahalanobis:
        gate = (tab["maha_d2"] >= maha_thr) & tab["liquid"]
        alerts = tab[gate].sort_values("maha_d2", ascending=False).copy()
        print(f"[maha] corte D^2 empirico (q={cfg.maha_quantile:.2f}) = {maha_thr:.2f}"
              f"  |  chi2 ref = {chi2.ppf(cfg.maha_quantile, df=len(SIGNAL_COLS)):.2f}")
    else:
        gate = (tab["n_signals"] >= cfg.min_signals) & tab["liquid"]
        alerts = tab[gate].sort_values(["n_signals", "maha_d2"], ascending=False).copy()

    state = load_state(cfg)
    alerts = apply_cooldown(alerts, asof, cfg, state)

    print_terminal(alerts, asof, cfg)
    plot_top(alerts, close, cfg, asof)

    for tk in alerts.index:
        state.setdefault("last_alert", {})[tk] = asof.isoformat()
    save_state(cfg, state)
    return alerts

if __name__ == "__main__":
    config = Config(excel_path="datos_b3.xlsx")
    run(config)
