# -*- coding: utf-8 -*-
"""
=============================================================================
 PIPELINE DE NOWCASTING DEL PBI ARGENTINO — THREE-PASS REGRESSION FILTER
=============================================================================
Implementación de nivel de producción del 3PRF de Kelly & Pruitt (2015,
Journal of Econometrics 186, 294-316) aplicado al nowcast del crecimiento
trimestral desestacionalizado del PBI de Argentina (DPBI_desest) a partir
de un panel de ~50 indicadores mensuales de alta frecuencia.

Arquitectura (módulos OOP):
    1. DataLoader            : ingesta del Excel y construcción del panel.
    2. StationarityTransformer: log/diferenciación guiada por tests ADF.
    3. PanelImputer          : imputación sin look-ahead (medias de train).
    4. PanelScaler           : estandarización z-score sin look-ahead.
    5. ThreePassRegressionFilter: estimador 3PRF con proxies automáticos
       (Tabla 2 del paper), fit/predict y transparencia de loadings.
    6. Backtester            : backtest OOS pseudo-real-time con ventana
       expansiva, métricas (RMSE, MAE, Hit Rate, R² OOS) y test
       Diebold-Mariano contra un benchmark AR(1).
    7. Visualizer            : gráficos de performance OOS, residuos y
       loadings top-10 ("caja abierta").

Decisiones econométricas clave (documentadas in-line):
    * Los indicadores mensuales se agregan a frecuencia trimestral por
      promedio simple (lógica bridge) ANTES de transformar, de modo que
      las tasas de variación resultantes sean t/t-1 trimestrales,
      homogéneas con el target.
    * El target DPBI_desest ya es una tasa de crecimiento q/q
      desestacionalizada: no se re-transforma.
    * El 3PRF se estima en su versión de tres pasadas de OLS (Tabla 1 del
      paper) y no en forma cerrada, porque (i) es la formulación que
      tolera paneles desbalanceados y (ii) hace explícita la lógica
      supervisada del filtro. El coeficiente implícito alpha sobre cada
      predictor (Teorema 2 / Ec. 4) se recupera para interpretar el
      modelo.
    * Horizonte h=0 (nowcast contemporáneo): y_t se regresa sobre F_t.
      El parámetro `horizon` permite h>=1 (forecast puro y_{t+h} ~ F_t).

Autor: pipeline generado para uso interno de research cuantitativo.
=============================================================================
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # backend no interactivo para exportar archivos
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid", context="talk", palette="deep")

# ---------------------------------------------------------------------------
# 1. INGESTA DE DATOS
# ---------------------------------------------------------------------------


class DataLoader:
    """Carga la hoja de indicadores desestacionalizados y separa target de
    predictores.

    La hoja `base_desest` contiene:
        * `Date`        : fin de mes.
        * `DPBI_desest` : crecimiento trimestral q/q del PBI desestacionalizado,
                          repetido en los tres meses del trimestre (NaN en los
                          meses cuyo trimestre aún no fue publicado).
        * 50 indicadores mensuales desestacionalizados en niveles.
    """

    TARGET_COL: str = "DPBI_desest"

    def __init__(self, xlsx_path: str | Path, sheet_name: str = "base_desest") -> None:
        self.xlsx_path = Path(xlsx_path)
        self.sheet_name = sheet_name

    def load(self) -> Tuple[pd.Series, pd.DataFrame]:
        """Devuelve (y_trimestral, X_mensual_en_niveles).

        Returns
        -------
        y : pd.Series
            Crecimiento q/q del PBI, indexado a fin de trimestre (Period Q).
        x_monthly : pd.DataFrame
            Panel mensual de predictores en niveles.
        """
        df = pd.read_excel(self.xlsx_path, sheet_name=self.sheet_name)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()

        # --- Target: colapsar la serie mensual repetida a frecuencia Q ---
        # Tomamos el último valor de cada trimestre (los tres meses son
        # idénticos por construcción); dropna elimina trimestres sin dato.
        y = (
            df[self.TARGET_COL]
            .groupby(df.index.to_period("Q"))
            .last()
            .dropna()
            .rename("dpbi")
        )

        x_monthly = df.drop(columns=[self.TARGET_COL])
        return y, x_monthly


# ---------------------------------------------------------------------------
# 2. TRANSFORMACIÓN A ESTACIONARIEDAD
# ---------------------------------------------------------------------------


@dataclass
class TransformSpec:
    """Registro de la transformación aplicada a una variable (auditoría)."""

    use_log: bool
    n_diffs: int
    adf_pvalue_final: float

    @property
    def label(self) -> str:
        base = "log" if self.use_log else "nivel"
        return f"{base} + {self.n_diffs} dif." if self.n_diffs else base


class StationarityTransformer:
    """Transforma cada predictor trimestral a una serie estacionaria.

    Regla de decisión (estándar en la literatura de diffusion indexes,
    Stock & Watson 2002):
        1. Si la serie es estrictamente positiva -> aplicar logaritmo
           (estabiliza varianza; la primera diferencia del log es una tasa
           de crecimiento, directamente comparable con el target).
        2. Aplicar diferencias sucesivas (máx. `max_diffs`) hasta que el
           test ADF rechace raíz unitaria al nivel `alpha`.

    Nota metodológica: la decisión de transformación usa la muestra
    completa. Esto es una elección ESTRUCTURAL (qué es la variable), no
    predictiva, y es práctica habitual en nowcasting; el look-ahead
    relevante (medias, desvíos, imputación, coeficientes) se elimina
    aguas abajo, dentro del backtest.
    """

    def __init__(self, alpha: float = 0.05, max_diffs: int = 2) -> None:
        self.alpha = alpha
        self.max_diffs = max_diffs
        self.specs_: Dict[str, TransformSpec] = {}

    @staticmethod
    def _adf_pvalue(series: pd.Series) -> float:
        """P-value del ADF con selección automática de rezagos (AIC)."""
        clean = series.dropna()
        if clean.nunique() < 3 or len(clean) < 12:
            return 0.0  # serie degenerada: se trata como estacionaria
        try:
            return float(adfuller(clean, autolag="AIC")[1])
        except Exception:
            return 0.0

    def fit_transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Aplica log/diferencias por columna y guarda la especificación."""
        out: Dict[str, pd.Series] = {}
        for col in x.columns:
            s = x[col].astype(float)
            use_log = bool((s.dropna() > 0).all())
            if use_log:
                s = np.log(s)

            n_diffs = 0
            pval = self._adf_pvalue(s)
            while pval > self.alpha and n_diffs < self.max_diffs:
                s = s.diff()
                n_diffs += 1
                pval = self._adf_pvalue(s)

            self.specs_[col] = TransformSpec(use_log, n_diffs, pval)
            out[col] = s
        return pd.DataFrame(out, index=x.index)

    def summary(self) -> pd.DataFrame:
        """Tabla de auditoría de las transformaciones aplicadas."""
        return pd.DataFrame(
            {
                "transformacion": {k: v.label for k, v in self.specs_.items()},
                "adf_pvalue": {k: round(v.adf_pvalue_final, 4) for k, v in self.specs_.items()},
            }
        )


# ---------------------------------------------------------------------------
# 3. IMPUTACIÓN Y ESCALADO SIN LOOK-AHEAD
# ---------------------------------------------------------------------------


class PanelImputer:
    """Imputación por media de columna estimada SOLO con datos de train.

    Tras la transformación a tasas de crecimiento estandarizables, la media
    incondicional es el imputador insesgado más conservador: equivale a
    asignar contribución nula del predictor faltante al factor (su z-score
    imputado será 0 tras el escalado). Se descartan columnas sin ningún
    dato en la ventana de entrenamiento.
    """

    def __init__(self) -> None:
        self.means_: Optional[pd.Series] = None
        self.valid_cols_: Optional[List[str]] = None

    def fit(self, x_train: pd.DataFrame) -> "PanelImputer":
        self.means_ = x_train.mean(axis=0, skipna=True)
        self.valid_cols_ = self.means_.dropna().index.tolist()
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        assert self.means_ is not None, "Llamar a fit() primero."
        return x[self.valid_cols_].fillna(self.means_[self.valid_cols_])


class PanelScaler:
    """Estandarización z-score con momentos de la ventana de entrenamiento.

    El 3PRF (como PCR y PLS) no es invariante a escala: el paper trabaja
    con predictores de varianza unitaria (Sección 2.1). Usar momentos de
    train evita contaminar el backtest con información futura.
    """

    def __init__(self) -> None:
        self.mu_: Optional[pd.Series] = None
        self.sigma_: Optional[pd.Series] = None

    def fit(self, x_train: pd.DataFrame) -> "PanelScaler":
        self.mu_ = x_train.mean(axis=0)
        sigma = x_train.std(axis=0, ddof=1)
        # Piso numérico para columnas casi constantes en la ventana.
        self.sigma_ = sigma.replace(0.0, np.nan).fillna(1.0)
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        assert self.mu_ is not None, "Llamar a fit() primero."
        return (x - self.mu_) / self.sigma_


# ---------------------------------------------------------------------------
# 4. ESTIMADOR THREE-PASS REGRESSION FILTER
# ---------------------------------------------------------------------------


class ThreePassRegressionFilter:
    """Three-Pass Regression Filter (Kelly & Pruitt, 2015) con proxies
    automáticos.

    Las tres pasadas (Tabla 1 del paper), todas por OLS con constante:

        Pasada 1 (time series, una por predictor i = 1..N):
            x_{i,t} = phi_{0,i} + z_t' phi_i + eps_{i,t}
            -> retiene phi_i_hat  (sensibilidad del predictor a los proxies,
               que representan los factores RELEVANTES para el target).

        Pasada 2 (cross-section, una por período t = 1..T):
            x_{i,t} = phi_{0,t} + phi_i_hat' F_t + err_{i,t}
            -> retiene F_t_hat  (factor(es) predictivo(s) en t).

        Pasada 3 (time series predictiva):
            y_{t+h} = beta_0 + F_t_hat' beta + eta_{t+h}
            -> el fitted value es el pronóstico 3PRF.

    Los proxies se construyen con el algoritmo automático de la Tabla 2:
    z^(1) = y; z^(k) = residuo del 3PRF con k-1 proxies. Esto garantiza
    (Teorema 7) que los proxies cargan solo sobre factores relevantes, que
    es la ventaja estructural del 3PRF frente a PCR cuando los factores
    dominantes del panel son irrelevantes para el PBI.

    Parameters
    ----------
    n_factors : int
        L = número de proxies automáticos / factores relevantes a extraer.
    horizon : int
        h. Con h=0 es un NOWCAST contemporáneo (y_t sobre F_t); con h>=1
        es un forecast puro (y_{t+h} sobre F_t), como en el paper.

    Attributes (post-fit)
    ---------------------
    phi_ : pd.DataFrame (N x L)      loadings de la pasada 1.
    factors_ : pd.DataFrame (T x L)  factores estimados en la pasada 2.
    beta0_, beta_ : float, np.ndarray  coeficientes de la pasada 3.
    alpha_ : pd.Series (N,)          coeficiente predictivo implícito sobre
                                     cada predictor (Ec. 4 / Teorema 2):
                                     y_hat = y_bar + (x_t - x_bar)' alpha.
                                     Es la métrica de transparencia central.
    """

    def __init__(self, n_factors: int = 1, horizon: int = 0) -> None:
        if n_factors < 1:
            raise ValueError("n_factors debe ser >= 1.")
        if horizon < 0:
            raise ValueError("horizon debe ser >= 0.")
        self.n_factors = n_factors
        self.horizon = horizon

        self.phi_: Optional[pd.DataFrame] = None
        self.factors_: Optional[pd.DataFrame] = None
        self.beta0_: float = np.nan
        self.beta_: Optional[np.ndarray] = None
        self.alpha_: Optional[pd.Series] = None
        self.columns_: Optional[List[str]] = None
        self._x_train_mean: Optional[pd.Series] = None
        self._y_train_mean: float = np.nan

    # -------------------------- helpers OLS --------------------------- #

    @staticmethod
    def _ols(y: np.ndarray, x: np.ndarray) -> np.ndarray:
        """OLS con constante vía lstsq (estable ante colinealidad leve).

        Devuelve el vector [b0, b1, ..., bK].
        """
        design = np.column_stack([np.ones(len(x)), x])
        coefs, *_ = np.linalg.lstsq(design, y, rcond=None)
        return coefs

    # ------------------------- las tres pasadas ------------------------ #

    def _three_passes(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Ejecuta las pasadas 1-3 para un set de proxies Z dado.

        x : (T, N)  panel estandarizado.
        y : (T,)    target alineado con x (mismo índice temporal).
        z : (T, L)  proxies.

        Returns
        -------
        phi (N, L), factors (T, L), beta0+beta (L+1,), fitted (T,)
        """
        t_obs, n_pred = x.shape
        n_prox = z.shape[1]

        # ---- Pasada 1: N regresiones de series de tiempo -------------- #
        # x_i sobre Z. Vectorizada: un solo lstsq con matriz de diseño común.
        design_z = np.column_stack([np.ones(t_obs), z])
        coefs1, *_ = np.linalg.lstsq(design_z, x, rcond=None)  # (L+1, N)
        phi = coefs1[1:, :].T  # (N, L) — se descarta la constante

        # ---- Pasada 2: T regresiones de corte transversal ------------- #
        # x_t sobre phi_hat. También vectorizada: diseño común (N, L+1).
        design_phi = np.column_stack([np.ones(n_pred), phi])
        coefs2, *_ = np.linalg.lstsq(design_phi, x.T, rcond=None)  # (L+1, T)
        factors = coefs2[1:, :].T  # (T, L)

        # ---- Pasada 3: regresión predictiva --------------------------- #
        # Alineación temporal: y_{t+h} sobre F_t  =>  F[:-h] vs y[h:].
        h = self.horizon
        f_lhs = factors[: t_obs - h] if h > 0 else factors
        y_rhs = y[h:] if h > 0 else y
        beta_full = self._ols(y_rhs, f_lhs)  # (L+1,)
        fitted = design_z[:, :1] * 0  # placeholder shape (T,1) -> flatten
        fitted = beta_full[0] + factors @ beta_full[1:]
        return phi, factors, beta_full, fitted

    # ----------------------------- API -------------------------------- #

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "ThreePassRegressionFilter":
        """Estima el 3PRF con proxies automáticos (Tabla 2 del paper).

        Parameters
        ----------
        x : DataFrame (T x N) estandarizado, indexado por trimestre.
        y : Series (T_y,) target; su índice debe ser subconjunto del de x
            (los últimos trimestres de x pueden no tener y publicado: son
            justamente los que se nowcastean con predict()).
        """
        # Ventana de estimación: trimestres con target E indicadores.
        common_idx = x.index.intersection(y.index)
        x_fit = x.loc[common_idx]
        y_fit = y.loc[common_idx].astype(float)

        x_mat = x_fit.to_numpy(dtype=float)
        y_vec = y_fit.to_numpy(dtype=float)
        t_obs = len(y_vec)

        # ---- Algoritmo de proxies automáticos (Tabla 2) --------------- #
        # r0 = y; z_k = residuo del 3PRF con proxies 1..k-1.
        proxies: List[np.ndarray] = []
        residual = y_vec - y_vec.mean()
        beta_full = np.array([y_vec.mean()])
        phi = factors = fitted = None
        for _ in range(self.n_factors):
            proxies.append(residual.copy())
            z = np.column_stack(proxies)
            phi, factors, beta_full, fitted = self._three_passes(x_mat, y_vec, z)
            # Residuo alineado con y (para h=0 coincide con fitted completo).
            h = self.horizon
            resid_aligned = y_vec[h:] - fitted[: t_obs - h] if h > 0 else y_vec - fitted
            # Para el proxy siguiente se necesita longitud T: se rellena el
            # borde con ceros (contribución neutra), solo relevante si h>0.
            residual = np.zeros(t_obs)
            residual[: len(resid_aligned)] = resid_aligned

        # ---- Persistencia de resultados -------------------------------- #
        self.columns_ = list(x.columns)
        factor_cols = [f"F{k+1}" for k in range(self.n_factors)]
        self.phi_ = pd.DataFrame(phi, index=self.columns_, columns=factor_cols)
        self.factors_ = pd.DataFrame(factors, index=common_idx, columns=factor_cols)
        self.beta0_ = float(beta_full[0])
        self.beta_ = beta_full[1:].copy()
        self._x_train_mean = x_fit.mean(axis=0)
        self._y_train_mean = float(y_vec.mean())

        # ---- Coeficiente implícito alpha (transparencia, Ec. 4) -------- #
        # y_hat = y_bar + J_T X W (W' Sxx W)^{-1} W' s_Xy,  W = J_N X' J_T Z.
        z_final = np.column_stack(proxies)
        self.alpha_ = self._implied_alpha(x_mat, y_vec, z_final)
        return self

    def _implied_alpha(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> pd.Series:
        """Coeficiente predictivo N-dimensional implícito (forma cerrada).

        Es el objeto del Teorema 2: N * alpha_i -> (phi_i - phi_bar)' beta.
        Permite responder "qué variables mueven el nowcast" sin abrir las
        tres pasadas. Se calcula sobre datos demeaned (J_T, J_N del paper).
        """
        t_obs, n_pred = x.shape
        jx = x - x.mean(axis=0, keepdims=True)          # J_T X
        jz = z - z.mean(axis=0, keepdims=True)          # J_T Z
        w = jx.T @ jz                                    # X' J_T Z  (N, L)
        w = w - w.mean(axis=0, keepdims=True)            # J_N (·)   (N, L)
        sxx = jx.T @ jx                                  # (N, N)
        sxy = jx.T @ (y - y.mean())                      # (N,)
        core = w.T @ sxx @ w                             # (L, L)
        alpha = w @ np.linalg.solve(core, w.T @ sxy)     # (N,)
        return pd.Series(alpha, index=self.columns_, name="alpha")

    def predict(self, x_new: pd.DataFrame) -> pd.Series:
        """Nowcast/forecast para nuevos períodos.

        La pasada 2 se re-ejecuta para cada t nuevo: es la operación de
        filtrado (mapear el corte transversal x_t al espacio factorial vía
        los loadings phi_hat estimados en train), sin re-estimar phi ni beta.
        """
        assert self.phi_ is not None, "Llamar a fit() primero."
        x_mat = x_new[self.columns_].to_numpy(dtype=float)
        n_pred = x_mat.shape[1]

        design_phi = np.column_stack([np.ones(n_pred), self.phi_.to_numpy()])
        coefs, *_ = np.linalg.lstsq(design_phi, x_mat.T, rcond=None)
        f_new = coefs[1:, :].T  # (T_new, L)
        yhat = self.beta0_ + f_new @ self.beta_
        return pd.Series(yhat, index=x_new.index, name="yhat_3prf")

    # ------------------------- transparencia --------------------------- #

    def top_predictors(self, k: int = 10) -> pd.DataFrame:
        """Top-k predictores por |alpha| (peso predictivo implícito).

        `alpha` responde a la pregunta del multiple-regression de Cochrane
        (2011) citada en el paper: contribución marginal de cada x_i al
        pronóstico, bajo la restricción de que los factores irrelevantes
        no influyen (Teorema 8).
        """
        assert self.alpha_ is not None, "Llamar a fit() primero."
        ranked = self.alpha_.reindex(self.alpha_.abs().sort_values(ascending=False).index)
        return ranked.head(k).to_frame()

    def contribution_last_obs(self, x_last: pd.Series, k: int = 10) -> pd.DataFrame:
        """Descompone el último nowcast: contribución_i = alpha_i * (x_i - x_bar_train).

        Suma de contribuciones + media histórica del target = nowcast
        (identidad exacta de la representación y_hat = y_bar + (x-x_bar)'alpha).
        """
        assert self.alpha_ is not None and self._x_train_mean is not None
        dev = x_last[self.columns_] - self._x_train_mean
        contrib = (self.alpha_ * dev).rename("contribucion_pp")
        ranked = contrib.reindex(contrib.abs().sort_values(ascending=False).index)
        return ranked.head(k).to_frame()


# ---------------------------------------------------------------------------
# 5. BACKTESTING OUT-OF-SAMPLE
# ---------------------------------------------------------------------------


@dataclass
class BacktestResult:
    """Contenedor de resultados del backtest OOS."""

    predictions: pd.DataFrame  # columnas: y_real, y_hat, y_bench
    rmse: float = field(init=False)
    mae: float = field(init=False)
    hit_rate: float = field(init=False)
    r2_oos: float = field(init=False)  # 1 - MSE(modelo)/MSE(media histórica)
    dm_stat: float = field(init=False)
    dm_pvalue: float = field(init=False)

    def __post_init__(self) -> None:
        df = self.predictions.dropna()
        err = df["y_real"] - df["y_hat"]
        err_bench = df["y_real"] - df["y_bench"]
        err_mean = df["y_real"] - df["y_histmean"]

        self.rmse = float(np.sqrt(np.mean(err**2)))
        self.mae = float(np.mean(np.abs(err)))
        self.hit_rate = float(np.mean(np.sign(df["y_hat"]) == np.sign(df["y_real"])))
        self.r2_oos = float(1.0 - np.sum(err**2) / np.sum(err_mean**2))
        self.dm_stat, self.dm_pvalue = self._diebold_mariano(err.values, err_bench.values)

    @staticmethod
    def _diebold_mariano(e1: np.ndarray, e2: np.ndarray) -> Tuple[float, float]:
        """Test Diebold-Mariano (1995) con varianza HAC (Newey-West).

        H0: igual precisión (loss = error cuadrático). DM < 0 => el modelo 1
        (3PRF) tiene menor pérdida que el benchmark.
        """
        from scipy import stats

        d = e1**2 - e2**2
        n = len(d)
        d_bar = d.mean()
        lag = max(1, int(np.floor(n ** (1 / 3))))
        gamma0 = np.var(d, ddof=0)
        var_d = gamma0
        for j in range(1, lag + 1):
            gamma_j = np.mean((d[j:] - d_bar) * (d[:-j] - d_bar))
            var_d += 2.0 * (1.0 - j / (lag + 1)) * gamma_j
        var_d = max(var_d, 1e-12)
        dm = d_bar / np.sqrt(var_d / n)
        pval = 2.0 * (1.0 - stats.norm.cdf(abs(dm)))
        return float(dm), float(pval)


class Backtester:
    """Backtest OOS pseudo-real-time con ventana expansiva.

    En cada trimestre t del período de evaluación:
        1. Se entrena TODO el pipeline (imputer -> scaler -> 3PRF) usando
           exclusivamente información hasta t-1 (target) y los indicadores
           hasta t (los datos mensuales del trimestre t están disponibles
           antes de la publicación del PBI: es el timing real del nowcast).
        2. Se nowcastea y_t y se compara contra el dato publicado.

    Benchmark: AR(1) del crecimiento del PBI estimado con la misma ventana
    (el benchmark honesto para un nowcast de actividad); se reporta además
    la media histórica (denominador del R² OOS, como en el paper).
    """

    def __init__(
        self,
        n_factors: int = 1,
        min_train_quarters: int = 20,
        horizon: int = 0,
    ) -> None:
        self.n_factors = n_factors
        self.min_train_quarters = min_train_quarters
        self.horizon = horizon

    @staticmethod
    def _ar1_forecast(y_train: pd.Series) -> float:
        """Pronóstico un paso adelante de un AR(1) con constante (OLS)."""
        y = y_train.to_numpy(dtype=float)
        if len(y) < 8:
            return float(y.mean())
        design = np.column_stack([np.ones(len(y) - 1), y[:-1]])
        coefs, *_ = np.linalg.lstsq(design, y[1:], rcond=None)
        return float(coefs[0] + coefs[1] * y[-1])

    def run(self, x_q: pd.DataFrame, y: pd.Series) -> BacktestResult:
        """Ejecuta el backtest y devuelve métricas + trayectoria de predicciones."""
        # Alineación defensiva: solo trimestres con target E indicadores
        # (la diferenciación consume el primer trimestre del panel).
        y = y.loc[y.index.intersection(x_q.index)]
        eval_quarters = y.index[self.min_train_quarters:]
        records: List[Dict[str, float]] = []

        for q in eval_quarters:
            y_train = y.loc[: q].iloc[:-1]  # target hasta t-1 (t aún no publicado)
            x_train = x_q.loc[y_train.index]  # panel alineado con el target conocido
            x_now = x_q.loc[[q]]  # indicadores del trimestre a nowcastear

            # Pipeline sin look-ahead: momentos e imputación solo con train.
            imputer = PanelImputer().fit(x_train)
            x_train_i = imputer.transform(x_train)
            x_now_i = imputer.transform(x_now)

            scaler = PanelScaler().fit(x_train_i)
            x_train_s = scaler.transform(x_train_i)
            x_now_s = scaler.transform(x_now_i)

            model = ThreePassRegressionFilter(
                n_factors=self.n_factors, horizon=self.horizon
            ).fit(x_train_s, y_train)
            y_hat = float(model.predict(x_now_s).iloc[0])

            records.append(
                {
                    "quarter": q,
                    "y_real": float(y.loc[q]),
                    "y_hat": y_hat,
                    "y_bench": self._ar1_forecast(y_train),
                    "y_histmean": float(y_train.mean()),
                }
            )

        preds = pd.DataFrame(records).set_index("quarter")
        return BacktestResult(predictions=preds)


# ---------------------------------------------------------------------------
# 6. VISUALIZACIÓN
# ---------------------------------------------------------------------------


class Visualizer:
    """Exporta los tres gráficos requeridos a `output_dir`."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_oos_performance(self, preds: pd.DataFrame, nowcast: Optional[Tuple[str, float]] = None) -> Path:
        """PBI real vs. nowcast 3PRF vs. benchmark AR(1), en % q/q."""
        fig, ax = plt.subplots(figsize=(13, 6.5))
        idx = preds.index.to_timestamp() if hasattr(preds.index, "to_timestamp") else preds.index
        ax.plot(idx, preds["y_real"] * 100, marker="o", lw=2.2, label="PBI real (q/q, s.e.)", color="#1f3b73")
        ax.plot(idx, preds["y_hat"] * 100, marker="s", lw=2.0, ls="--", label="Nowcast 3PRF", color="#c0392b")
        ax.plot(idx, preds["y_bench"] * 100, lw=1.3, ls=":", label="Benchmark AR(1)", color="#7f8c8d")
        if nowcast is not None:
            ax.scatter([pd.Period(nowcast[0]).to_timestamp()], [nowcast[1] * 100],
                       s=160, marker="*", color="#e67e22", zorder=5,
                       label=f"Nowcast {nowcast[0]}: {nowcast[1]*100:.2f}%")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title("Backtest OOS (ventana expansiva) — Crecimiento trimestral del PBI")
        ax.set_ylabel("% trimestral")
        ax.legend(frameon=True, fontsize=12)
        fig.tight_layout()
        path = self.output_dir / "oos_real_vs_predicho.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_residuals(self, preds: pd.DataFrame) -> Path:
        """Distribución de residuos OOS (histograma + KDE) y residuos en el tiempo."""
        resid = (preds["y_real"] - preds["y_hat"]) * 100
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

        sns.histplot(resid, kde=True, ax=axes[0], color="#1f3b73", bins=12, stat="density")
        axes[0].axvline(0, color="black", lw=0.9)
        axes[0].axvline(resid.mean(), color="#c0392b", ls="--", lw=1.5,
                        label=f"media = {resid.mean():.2f} p.p.")
        axes[0].set_title("Distribución de residuos OOS")
        axes[0].set_xlabel("Error (p.p. de crecimiento q/q)")
        axes[0].legend(fontsize=11)

        idx = preds.index.to_timestamp() if hasattr(preds.index, "to_timestamp") else preds.index
        axes[1].bar(idx, resid, width=70, color=np.where(resid >= 0, "#1f3b73", "#c0392b"))
        axes[1].axhline(0, color="black", lw=0.9)
        axes[1].set_title("Residuos en el tiempo (sesgo/heterocedasticidad)")
        axes[1].set_ylabel("p.p.")
        fig.tight_layout()
        path = self.output_dir / "residuos_oos.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_loadings(self, model: ThreePassRegressionFilter, contrib: pd.DataFrame, k: int = 10) -> Path:
        """Caja abierta: |alpha| top-k y contribución al último nowcast."""
        top_alpha = model.top_predictors(k)
        fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

        colors_a = np.where(top_alpha["alpha"] >= 0, "#1f3b73", "#c0392b")
        axes[0].barh(top_alpha.index[::-1], top_alpha["alpha"][::-1], color=colors_a[::-1])
        axes[0].axvline(0, color="black", lw=0.8)
        axes[0].set_title(f"Top-{k} predictores por peso predictivo implícito (α)")
        axes[0].set_xlabel("α (coef. sobre predictor estandarizado)")

        colors_c = np.where(contrib.iloc[:, 0] >= 0, "#1f3b73", "#c0392b")
        axes[1].barh(contrib.index[::-1], contrib.iloc[::-1, 0] * 100, color=colors_c[::-1])
        axes[1].axvline(0, color="black", lw=0.8)
        axes[1].set_title("Contribución al nowcast del último trimestre")
        axes[1].set_xlabel("Contribución (p.p. de crecimiento q/q)")
        fig.tight_layout()
        path = self.output_dir / "loadings_top10.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path


# ---------------------------------------------------------------------------
# 7. ORQUESTACIÓN
# ---------------------------------------------------------------------------


def aggregate_monthly_to_quarterly(x_monthly: pd.DataFrame) -> pd.DataFrame:
    """Agrega el panel mensual a trimestral por promedio simple (bridge).

    Promediar niveles dentro del trimestre antes de transformar hace que
    dlog trimestral aproxime la tasa de variación del promedio trimestral,
    que es la magnitud comparable con el PBI trimestral. `min_count=1`
    permite trimestres parciales (nowcast intra-trimestre con 1-2 meses).
    """
    return x_monthly.groupby(x_monthly.index.to_period("Q")).mean()


def run_pipeline(
    xlsx_path: str | Path,
    output_dir: str | Path = "outputs",
    n_factors: int = 1,
    min_train_quarters: int = 20,
    factor_grid: Tuple[int, ...] = (1, 2, 3),
) -> None:
    """Pipeline end-to-end: datos -> transformación -> backtest -> nowcast -> gráficos."""
    sep = "=" * 78

    # ---------------- 1. Datos ---------------- #
    print(sep, "\n 1. CARGA Y PREPROCESAMIENTO", "\n" + sep)
    y, x_monthly = DataLoader(xlsx_path).load()
    x_q_levels = aggregate_monthly_to_quarterly(x_monthly)

    transformer = StationarityTransformer(alpha=0.05, max_diffs=2)
    x_q = transformer.fit_transform(x_q_levels)
    # Se descarta la primera fila (perdida por diferenciación generalizada).
    x_q = x_q.iloc[1:]

    print(f"Target: {len(y)} trimestres ({y.index[0]} a {y.index[-1]})")
    print(f"Panel X: {x_q.shape[0]} trimestres x {x_q.shape[1]} predictores "
          f"({x_q.index[0]} a {x_q.index[-1]})")
    n_log = sum(s.use_log for s in transformer.specs_.values())
    n_diff = sum(s.n_diffs > 0 for s in transformer.specs_.values())
    print(f"Transformaciones: {n_log} series en log, {n_diff} diferenciadas (ADF 5%).")

    # ---------------- 2. Backtest OOS ---------------- #
    print("\n" + sep, "\n 2. BACKTEST OUT-OF-SAMPLE (ventana expansiva)", "\n" + sep)
    results: Dict[int, BacktestResult] = {}
    for L in factor_grid:
        bt = Backtester(n_factors=L, min_train_quarters=min_train_quarters)
        results[L] = bt.run(x_q, y)

    header = f"{'L (factores)':>12} | {'RMSE':>8} | {'MAE':>8} | {'HitRate':>8} | {'R2_OOS':>8} | {'DM vs AR1':>10} | {'p-val':>6}"
    print(header)
    print("-" * len(header))
    for L, res in results.items():
        print(f"{L:>12} | {res.rmse*100:>7.3f}% | {res.mae*100:>7.3f}% | "
              f"{res.hit_rate*100:>7.1f}% | {res.r2_oos*100:>7.1f}% | "
              f"{res.dm_stat:>10.2f} | {res.dm_pvalue:>6.3f}")

    best = results[n_factors]
    n_oos = len(best.predictions)
    print(f"\nEspecificación reportada: L={n_factors} | {n_oos} trimestres OOS "
          f"({best.predictions.index[0]} a {best.predictions.index[-1]})")
    print("(DM < 0 y p-val bajo => el 3PRF supera al AR(1) con pérdida cuadrática.)")

    # ---------------- 3. Modelo final + nowcast ---------------- #
    print("\n" + sep, "\n 3. MODELO FINAL Y NOWCAST DEL TRIMESTRE CORRIENTE", "\n" + sep)
    y = y.loc[y.index.intersection(x_q.index)]
    imputer = PanelImputer().fit(x_q.loc[y.index])
    scaler = PanelScaler().fit(imputer.transform(x_q.loc[y.index]))
    x_all_s = scaler.transform(imputer.transform(x_q))

    model = ThreePassRegressionFilter(n_factors=n_factors, horizon=0).fit(x_all_s, y)

    future_idx = x_q.index.difference(y.index)
    nowcast_tuple: Optional[Tuple[str, float]] = None
    if len(future_idx) > 0:
        nowcasts = model.predict(x_all_s.loc[future_idx])
        for q, v in nowcasts.items():
            n_meses = x_monthly.loc[
                x_monthly.index.to_period("Q") == q
            ].shape[0]
            print(f"  Nowcast {q}: {v*100:+.2f}% q/q  "
                  f"(con {n_meses} de 3 meses de datos del trimestre)")
        nowcast_tuple = (str(future_idx[-1]), float(nowcasts.iloc[-1]))

    print("\nTop-10 predictores por peso predictivo implícito |α| (último fit):")
    print(model.top_predictors(10).to_string(float_format=lambda v: f"{v:+.4f}"))

    last_q = x_all_s.index[-1]
    contrib = model.contribution_last_obs(x_all_s.loc[last_q], k=10)
    print(f"\nDescomposición del nowcast de {last_q} (contribuciones en p.p.):")
    print((contrib * 100).to_string(float_format=lambda v: f"{v:+.3f}"))

    # ---------------- 4. Gráficos ---------------- #
    print("\n" + sep, "\n 4. EXPORTANDO GRÁFICOS", "\n" + sep)
    viz = Visualizer(output_dir)
    p1 = viz.plot_oos_performance(best.predictions, nowcast=nowcast_tuple)
    p2 = viz.plot_residuals(best.predictions)
    p3 = viz.plot_loadings(model, contrib, k=10)
    for p in (p1, p2, p3):
        print(f"  -> {p}")

    # Export de la trayectoria OOS para auditoría.
    csv_path = Path(output_dir) / "backtest_oos_predicciones.csv"
    best.predictions.to_csv(csv_path)
    print(f"  -> {csv_path}")


if __name__ == "__main__":
    run_pipeline(
        xlsx_path="/mnt/user-data/uploads/base_PBI.xlsx",
        output_dir="/mnt/user-data/outputs",
        n_factors=1,            # L=1 suele bastar (BIC del paper elige 1 en macro)
        min_train_quarters=20,  # ~5 años de entrenamiento inicial
        factor_grid=(1, 2, 3),  # grilla de robustez reportada en consola
    )
