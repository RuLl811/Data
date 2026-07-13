from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple

from sklearn.decomposition import PCA
import statsmodels.api as sm

import seaborn as sns
import matplotlib.pyplot as plt

import matplotlib.dates as mdates

# Reproductibilidad
SEED = 42
np.random.seed(SEED)

@dataclass
class Config:
    path_excel: str = fr"/Elecciones/Modelo de PBI/base_PBI.xlsx"  # actualizar si cambia
    sheet_name: str = "base_desest"              # hoja del Excel
    date_col: str = "Date"                        # nombre de la columna fecha en el Excel
    pib_col: str = "DPBI_desest"                         # columna de PIB (ya diferencial, no transformar)
    x_transform: str = "log_diff"                 # transformación por defecto para indicadores X
    min_corr: float = 0.45
    max_factors: int = 10                           # máximo candidatos de factores para el scree
    window_quarters: int = 12     # ventana móvil (trimestres) para estimación rolling
    p_lags_y: int = 0                              # rezagos de y en la ecuación puente/factor. Agrega PBI regazado como var. indep.
    p_lags_f: int = 0                              # rezagos de factores en la regresión de y

CFG = Config()

# =====================
# BLOQUE 1 — Carga & auditoría de datos
# =====================
def load_base(cfg: Config) -> pd.DataFrame:
    df = pd.read_excel(cfg.path_excel, sheet_name=cfg.sheet_name)
    df[cfg.date_col] = pd.to_datetime(df[cfg.date_col])
    df = df.set_index(cfg.date_col).sort_index()
    df = df[~((df.index >= "2020-01-01") & (df.index <= "2021-06-30"))] # Quito el 2020 y 2021
    return df

def audit_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla de control: rango de fechas, conteo de meses, % NA por columna."""
    start, end = df.index.min(), df.index.max()
    expected = pd.date_range(start=start, end=end, freq='MS')
    missing = expected.difference(df.index)
    aud = pd.DataFrame({
        "start": [start],
        "end": [end],
        "n_rows": [len(df)],
        "n_expected_MS": [len(expected)],
        "missing_MS": [len(missing)]
    })
    na_pct = df.isna().mean().sort_values(ascending=False)
    aud_na = na_pct.to_frame('na_pct')
    return aud, aud_na

# =====================
# BLOQUE 2 — Transformaciones
# =====================

def transform_monthly_series(df: pd.DataFrame, cfg: Config) -> Tuple[pd.Series, pd.DataFrame]:
    df = df.copy()
    # Agrupo el PBI
    y_m = df[cfg.pib_col].copy()
    y = y_m.groupby(pd.Grouper(freq='Q')).last()
    y.name = 'gdp_qoq'

    X = df.drop(columns=[cfg.pib_col]).copy()
    if cfg.x_transform == "log_diff":
        X = np.log(X).diff()
    elif cfg.x_transform == "diff":
        X = X.diff()
    elif cfg.x_transform == "std":

        pass
    else:
        raise ValueError("x_transform no reconocido")

    Xq = X.groupby(pd.Grouper(freq='Q')).mean()

    common_idx = y.index.intersection(Xq.index)
    y = y.reindex(common_idx)
    Xq = Xq.reindex(common_idx)

    mask = ~pd.Series(y.index).between('2020-01-01', '2021-06-30').values
    y = y.loc[mask]
    Xq = Xq.loc[mask]

    return y, Xq

# =====================
# BLOQUE 3 — Selección de features por correlación con y (pre-PCA)
# =====================
def select_features_by_corr(y: pd.Series, Xq: pd.DataFrame, min_corr: float) -> pd.DataFrame:
    Z = Xq.loc[y.index]
    Z = Z.fillna(method='ffill')

    corrs = Z.corrwith(y).sort_values(ascending=False)

    keep = corrs[corrs >= min_corr].index.tolist() # Seleccion por correlación de toda la ventana
    # Selección segun rolling corr -> se busca los indicadores con mejor rolling 12M en los ultimos 2 años
    """
    keep = [
        "bk", # 0.543742
        "bcd", # 0.527201
        "metalmec", # 0.480046
        "ipi_desest", #0.619221
        "automov", # 0.517494
        "ISAC_desest", #0.544849
        "bui", # 0.49
        "despacho_cem", # 0.497340
        "bui"
    ]
    """
    if len(keep) == 0:
        keep = corrs.head(15).index.tolist()

    corr_matrix = Z[keep].corr()

    mask = np.tril(np.ones_like(corr_matrix, dtype=bool))
    '''
    sns.set_theme(style="white", font_scale=1.1)
    plt.figure(figsize=(14, 12))
    ax = sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap="coolwarm",
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.8,
        cbar_kws={"shrink": 0.8, "label": "Correlación"},
        annot_kws={"size": 9, "color": "black"})
    plt.title("Matriz de correlación — variables seleccionadas para DFM", fontsize=16, weight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    #plt.show()
    
    #  correlación rolling 12m vs GDP ===
    rollcorr = pd.DataFrame(index=Z.index, columns=keep, dtype=float)
    for col in keep:
        rollcorr[col] = Z[col].rolling(window=12, min_periods=9).corr(y)

    n_vars = len(keep)
    ncols = 3
    nrows = int(np.ceil(n_vars / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, nrows * 2), sharex=True)
    axes = axes.flatten()

    for i, col in enumerate(keep):
        axes[i].plot(rollcorr.index, rollcorr[col], label=col, color='steelblue')
        axes[i].axhline(0, color='gray', lw=1, linestyle='--')
        axes[i].set_title(col, fontsize=9)
        axes[i].grid(alpha=0.3)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Correlación rolling 12 meses vs y", fontsize=16, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    #plt.show()
    '''
    print(corrs)
    return Z[keep]

# =====================
# BLOQUE 4 — PCA (scores F, var. explicada y loadings)
# =====================

def fit_pca(Z: pd.DataFrame, max_factors: int) -> Tuple[PCA, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    Zc = (Z - Z.mean(axis=0)) / Z.std(axis=0, ddof=0) # Estandarizo la serie para que aporten en forma simétrica

    pca = PCA(n_components=min(max_factors, Z.shape[1]), random_state=SEED)
    pca.fit(Zc)

    # Scores (factores comunes)
    F = pd.DataFrame(pca.transform(Zc), index=Z.index, columns=[f"F{i+1}" for i in range(pca.n_components_)])

    # Varianza explicada
    evr = pd.DataFrame({
        'component': np.arange(1, F.shape[1] + 1),
        'explained_var_ratio': pca.explained_variance_ratio_,
        'cum_explained': np.cumsum(pca.explained_variance_ratio_)
    })

    # Loadings (pesos de cada variable en cada PC)
    pca_columns = [f"PC{i+1}" for i in range(pca.n_components_)]
    loadings = pd.DataFrame(pca.components_.T, columns=pca_columns, index=Z.columns)
    '''
    plt.plot(evr['component'], evr['cum_explained'], marker='o')
    plt.axhline(0.9, color='gray', ls='--', label='90% var explicada')
    plt.xlabel('Número de factores')
    plt.ylabel('Varianza acumulada explicada')
    plt.title('Scree plot — PCA')
    plt.legend()
    plt.grid(alpha=0.3)
    #plt.show()
    
    def plot_scree(pca, show_kaiser=True, title="Scree Plot — Autovalores Ordenados"):

        lambdas = np.asarray(pca.explained_variance_)  # autovalores (varianzas de cada PC)
        pcs = np.arange(1, len(lambdas) + 1)

        plt.figure(figsize=(7.5, 4.5))
        plt.plot(pcs, lambdas, marker='o', linewidth=1.5)
        if show_kaiser:
            plt.axhline(1.0, linestyle='--', color='red', linewidth=1, label='λ = 1')
            plt.legend()
        plt.title(title, fontsize=14, weight='bold')
        plt.xlabel("Componente principal")
        plt.ylabel("Autovalor (λ)")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    #plot_scree(pca)


    #print(loadings.iloc[:, :4])
    '''
    return pca, F, evr, loadings


# =====================
# Ecuación del **modelo de factores**
# =====================

def regress_y_on_factors(y: pd.Series, F: pd.DataFrame, p_lags_y: int = 0, p_lags_f: int = 0, n_factors: int = 3):

    Fk = F.iloc[:, :min(n_factors, F.shape[1])] # Selección de primeros k factores

    # Construcción de X con lags opcionales (por defecto 0)
    X_list = [Fk]
    for i in range(1, p_lags_f + 1):
        X_list.append(Fk.shift(i).add_suffix(f"_L{i}"))
    X = pd.concat(X_list, axis=1)

    y_dep = y.copy()
    for i in range(1, p_lags_y + 1):
        X[f"y_L{i}"] = y_dep.shift(i)

    # Armado de vectores para OLS
    D = pd.concat([y_dep, X], axis=1).dropna()
    y_al = D.iloc[:, 0]
    X_al = sm.add_constant(D.iloc[:, 1:])

    model = sm.OLS(y_al, X_al).fit(cov_type='HAC', cov_kwds={'maxlags': 4}) # Modelo OLS
    return model


# =====================
# Nowcast rolling (1 trimestre)
# =====================

def rolling_nowcast(y: pd.Series, Xq: pd.DataFrame, cfg: Config, n_factors: int = 3) -> pd.DataFrame:
    Z = select_features_by_corr(y, Xq, cfg.min_corr)
    idx = y.index
    preds = []

    for t_end in range(cfg.window_quarters, len(idx)):
        train_idx = idx[t_end - cfg.window_quarters:t_end]
        test_idx = idx[t_end]

        # PCA en ventana
        Z_tr = Z.loc[train_idx]
        pca, F_tr, evr, loadings = fit_pca(Z_tr, cfg.max_factors)

        # Proyectar test usando
        Z_tr_mean = Z_tr.mean(axis=0)
        Z_tr_std = Z_tr.std(axis=0, ddof=0).replace(0, 1.0)
        Z_te = (Z.loc[[test_idx]] - Z_tr_mean) / Z_tr_std
        F_te = pd.DataFrame(pca.transform(Z_te.values), index=[test_idx], columns=[f"F{i+1}" for i in range(pca.n_components_)])

        # Ajustar modelo de factores en la ventana
        model = regress_y_on_factors(y.loc[train_idx], F_tr, cfg.p_lags_y, cfg.p_lags_f, n_factors=n_factors)

        # Armar X_test con lags (opcional)
        X_test = F_te.iloc[:, :min(n_factors, F_te.shape[1])].copy()
        for i in range(1, cfg.p_lags_f + 1):
            X_test = pd.concat([X_test, F_tr.iloc[[-i]].add_suffix(f"_L{i}")], axis=1)
        for i in range(1, cfg.p_lags_y + 1):
            X_test[f"y_L{i}"] = y.loc[train_idx][-i]

        X_test = sm.add_constant(X_test)

        train_cols = model.model.exog_names  # ['const', 'F1', 'F2', ..., 'y_L1', ...]
        X_test = X_test.reindex(columns=train_cols,
                                fill_value=0.0)

        y_hat = float(model.predict(X_test))

        preds.append({
            'date': test_idx,
            'y_true': float(y.loc[test_idx]),
            'y_hat': y_hat,
            'n_factors': n_factors,
            'R2_train': float(model.rsquared_adj)
        })

    fcst = pd.DataFrame(preds).set_index('date')
    fcst['err'] = fcst['y_true'] - fcst['y_hat']
    fcst.to_excel('forecast_results.xlsx')
    return fcst


# =====================
# Métricas
# =====================
def rmse(s: pd.Series) -> float:
    return float(np.sqrt(np.mean(np.square(s))))

def eval_nowcast(fcst: pd.DataFrame) -> pd.Series:
    return pd.Series({
        'RMSE': rmse(fcst['err']),
        'MAE': float(np.mean(np.abs(fcst['err']))),
        'Bias': float(np.mean(fcst['err']))
    })

# =====================
# Ejecución
# =====================
if __name__ == "__main__":
    df = load_base(CFG)
    aud, aud_na = audit_monthly(df)

    y, Xq = transform_monthly_series(df, CFG)
    Z = select_features_by_corr(y, Xq, CFG.min_corr)
    pca, F, evr, loadings = fit_pca(Z, CFG.max_factors)

    fcst = rolling_nowcast(y, Xq, CFG, n_factors=4)
    print(eval_nowcast(fcst))

    pass

# Grafico de errores rolling
def plot_rolling_errors(fcst: pd.DataFrame, window: int = 12, title: str | None = None):
    """
    Grafica RMSE, MSE y Bias en ventana rolling sobre la serie de errores fcst['err'].
    """
    e = fcst['err'].dropna()

    # Rolling metrics
    mse  = e.rolling(window).apply(lambda s: np.mean(s**2), raw=True)
    rmse = e.rolling(window).apply(lambda s: np.sqrt(np.mean(s**2)), raw=True)
    bias = e.rolling(window).mean()

    roll = pd.concat([mse.rename('MSE'), rmse.rename('RMSE'), bias.rename('Bias')], axis=1)

    # Métricas globales para referencia (línea horizontal)
    overall_mse = float((e**2).mean()) if len(e) else np.nan
    overall_rmse = float(np.sqrt(overall_mse)) if np.isfinite(overall_mse) else np.nan
    overall_bias = float(e.mean()) if len(e) else np.nan

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    (ax1, ax2, ax3) = axes

    ax1.plot(roll.index, roll['RMSE'], label=f'RMSE {window}q')

    ax1.set_title(title or f"Errores rolling (ventana = {window} trimestres)")
    ax1.set_ylabel("RMSE")
    ax1.grid(alpha=0.3)
    ax1.legend(loc='upper right')

    ax2.plot(roll.index, roll['MSE'], label=f'MSE {window}q')

    ax2.set_ylabel("MSE")
    ax2.grid(alpha=0.3)
    ax2.legend(loc='upper right')

    ax3.plot(roll.index, roll['Bias'], label=f'Bias {window}q')

    ax3.set_ylabel("Bias")
    ax3.set_xlabel("Fecha")
    ax3.grid(alpha=0.3)
    ax3.legend(loc='upper right')

    plt.tight_layout()
    plt.show()


fcst = rolling_nowcast(y, Xq, CFG, n_factors=3)
plot_rolling_errors(fcst, window=12)

def plot_rolling_r2(fcst: pd.DataFrame, window: int = 12):
    """
    Grafica el R² rolling del modelo de nowcast.
    """

    if 'R2_train' in fcst.columns:
        r2_series = fcst['R2_train'].dropna()
    else:
        y_true, y_hat = fcst['y_true'], fcst['y_hat']
        def rolling_r2(y_t, y_h, w):
            r2 = []
            for i in range(len(y_t)):
                if i < w:
                    r2.append(np.nan)
                else:
                    y_sub, yhat_sub = y_t.iloc[i-w:i], y_h.iloc[i-w:i]
                    ss_res = np.sum((y_sub - yhat_sub)**2)
                    ss_tot = np.sum((y_sub - y_sub.mean())**2)
                    r2.append(1 - ss_res/ss_tot if ss_tot > 0 else np.nan)
            return pd.Series(r2, index=y_t.index)
        r2_series = rolling_r2(y_true, y_hat, window)

    # Asegurar que el índice sea datetime
    if isinstance(r2_series.index, pd.PeriodIndex):
        r2_series.index = r2_series.index.to_timestamp(how='end')

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(r2_series.index, r2_series, label=f'R² rolling ({window} trimestres)', color='steelblue')
    ax.axhline(r2_series.mean(), color='gray', linestyle='--', linewidth=1.2, label=f'Promedio = {r2_series.mean():.2f}')
    ax.set_title("Evolución del R² rolling", fontsize=14, weight='bold')
    ax.set_ylabel("R²")
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right')

    # Limitar eje X al rango de datos reales
    xmin, xmax = r2_series.index.min(), r2_series.index.max()
    ax.set_xlim(xmin, xmax)

    # Ticks anuales limpios (sin 2020–2021)
    year_ticks = pd.date_range(xmin, xmax, freq='YS')
    year_ticks = [d for d in year_ticks if d.year not in {2020, 2021}]
    ax.set_xticks(year_ticks)
    ax.set_xticklabels([str(d.year) for d in year_ticks])

    plt.tight_layout()
    #plt.show()


#plot_rolling_r2(fcst, window=12)


def plot_nowcast_vs_actual(fcst: pd.DataFrame, y: pd.Series):

    y_aligned = y.loc[fcst.index].dropna()
    y_hat = fcst['y_hat'].loc[y_aligned.index]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(y_aligned.index, y_aligned, label='PBI observado', color='black', lw=1.8)
    ax.plot(y_hat.index, y_hat, label='Nowcast (modelo)', color='tab:blue', lw=2)
    ax.fill_between(y_aligned.index, y_hat, y_aligned, color='gray', alpha=0.15)
    ax.set_title("Nowcast del PBI vs Real", fontsize=14, weight='bold')
    ax.set_ylabel("Variación trimestral (%)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

#plot_nowcast_vs_actual(fcst, y)


def predict_next_quarter(y: pd.Series, Xq: pd.DataFrame, cfg: Config,
                         n_factors: int = 3, alpha: float = 0.05):
    """
    Devuelve el nowcast del próximo trimestre (fecha, punto y IC) usando:
    - selección de features por correlación (misma que en rolling)
    - PCA en ventana de entrenamiento (cfg.window_quarters)
    - regresión OLS con HAC
    - proyección del último trimestre disponible en Xq (parcial o completo)

    Retorna:
        dict con keys: date, y_hat, y_hat_pct, ci_low, ci_high, model, pca, keep, F_tr
    """

    Z_all = select_features_by_corr(y, Xq, cfg.min_corr)

    last_y = y.index.max()
    future_ix = Z_all.index[Z_all.index > last_y]
    if len(future_ix) == 0:

        test_idx = Z_all.index.max()
    else:

        test_idx = future_ix[-1]

    train_idx = y.index[-cfg.window_quarters:]
    Z_tr = Z_all.loc[train_idx]

    pca, F_tr, _, _ = fit_pca(Z_tr, cfg.max_factors)           # PCA fit sobre Z_tr estandarizado

    Z_tr_mean = Z_tr.mean(axis=0)
    Z_tr_std  = Z_tr.std(axis=0, ddof=0).replace(0, 1.0)
    Z_te_std  = (Z_all.loc[[test_idx]] - Z_tr_mean) / Z_tr_std

    F_te = pd.DataFrame(
        pca.transform(Z_te_std.values),
        index=[test_idx],
        columns=[f"F{i+1}" for i in range(pca.n_components_)]
    )

    model = regress_y_on_factors(
        y.loc[train_idx], F_tr,
        p_lags_y=cfg.p_lags_y, p_lags_f=cfg.p_lags_f,
        n_factors=n_factors
    )

    X_test = F_te.iloc[:, :min(n_factors, F_te.shape[1])].copy()

    for i in range(1, cfg.p_lags_f + 1):
        X_test = pd.concat([X_test, F_tr.iloc[[-i]].add_suffix(f"_L{i}")], axis=1)
    for i in range(1, cfg.p_lags_y + 1):
        X_test[f"y_L{i}"] = y.loc[train_idx][-i]

    X_test = sm.add_constant(X_test, has_constant='add')


    X_test = X_test.reindex(columns=model.model.exog_names, fill_value=0.0)


    pred = model.get_prediction(X_test)
    sf = pred.summary_frame(alpha=alpha)  # contiene mean, mean_ci_lower, mean_ci_upper

    y_hat = float(sf['mean'].iloc[0])
    ci_low = float(sf['mean_ci_lower'].iloc[0])
    ci_high = float(sf['mean_ci_upper'].iloc[0])

    return {
        'date': test_idx,
        'y_hat': y_hat,
        'y_hat_pct': 100.0 * y_hat,        # si y está en log-diff ≈ % trimestral
        'ci_low': ci_low,
        'ci_high': ci_high,
        'model': model,
        'pca': pca,
        'keep': Z_tr.columns.tolist(),
        'F_tr': F_tr
    }


res = predict_next_quarter(y, Xq, CFG, n_factors=3, alpha=0.10)  # IC 90% (opcional)
print(f"Nowcast {res['date'].date()}: {res['y_hat_pct']:.2f}% trimestral "
      f"(IC[{100*res['ci_low']:.2f}%, {100*res['ci_high']:.2f}%])")