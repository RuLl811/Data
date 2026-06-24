r"""
avellaneda_sscore.py
====================
Senial estadistica de reversion sobre el residuo idiosincratico (Avellaneda-Lee 2010).

Idea: el residuo acumulado X_t = sum(e) se modela como Ornstein-Uhlenbeck
      dX = kappa (m - X) dt + sigma dW
estimado via AR(1):  X_{n+1} = a + b X_n + zeta
      kappa = -log(b)            (velocidad de reversion, por periodo)
      m     = a / (1 - b)        (equilibrio)
      sigma_eq = sqrt(var(zeta)/(1-b^2))
      s-score = (X_t - m) / sigma_eq
      half-life = log(2) / kappa

Senial:  s alto  -> caro  -> short ;  s bajo -> barato -> long.
Reglas:  abrir si |s| > open_th ; cerrar si |s| < close_th (umbrales AL).
Filtro:  tradear solo nombres cuya half-life de ventana < hl_max (reversion util).

Ventajas para este caso (utilities Merval, N=6, semanal):
  - Es time-series por nombre: NO necesita seccion transversal -> N=6 no molesta.
  - Umbrales + holding -> turnover ~9x/anio (vs ~83x del reversal crudo).
  - Parametros testeables: ADF de estacionariedad, kappa, half-life.

Depende de rolling_residuals() de idiosyncratic_momentum.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


# --------------------------------------------------------------------------- #
# Estimacion OU sobre una ventana                                              #
# --------------------------------------------------------------------------- #
def ou_sscore(window: np.ndarray):
    """Ajusta OU al residuo acumulado de la ventana. Devuelve dict o None."""
    x = window - window.mean()
    X = np.cumsum(x)
    Xl, Xn = X[:-1], X[1:]
    A = np.vstack([np.ones_like(Xl), Xl]).T
    a, b = np.linalg.lstsq(A, Xn, rcond=None)[0]
    if not (0 < b < 1):                 # sin reversion estable -> descartar
        return None
    kappa = -np.log(b)
    m = a / (1 - b)
    eps = Xn - (a + b * Xl)
    sigma_eq = np.sqrt(np.var(eps) / (1 - b ** 2))
    if sigma_eq == 0:
        return None
    return {"s": (X[-1] - m) / sigma_eq,
            "kappa": kappa,
            "halflife": np.log(2) / kappa}


# --------------------------------------------------------------------------- #
# s-score rodante por nombre                                                   #
# --------------------------------------------------------------------------- #
def rolling_sscore(resid: pd.DataFrame, window: int = 52):
    """Devuelve (sscore, halflife) panels alineados al indice de resid."""
    s_panel = pd.DataFrame(index=resid.index, columns=resid.columns, dtype=float)
    hl_panel = pd.DataFrame(index=resid.index, columns=resid.columns, dtype=float)
    for n in resid.columns:
        r = resid[n].values
        for i in range(window, len(r)):
            win = r[i - window:i]
            if np.isnan(win).any():
                continue
            fit = ou_sscore(win)
            if fit is None:
                continue
            s_panel.iloc[i, s_panel.columns.get_loc(n)] = fit["s"]
            hl_panel.iloc[i, hl_panel.columns.get_loc(n)] = fit["halflife"]
    return s_panel, hl_panel


def adf_diagnostics(resid: pd.DataFrame, maxlag: int = 4) -> pd.DataFrame:
    """p-value ADF por nombre. p chico -> residuo estacionario (revierte)."""
    out = {}
    for n in resid.columns:
        r = resid[n].dropna()
        out[n] = adfuller(r, maxlag=maxlag, autolag=None)[1]
    return pd.Series(out, name="adf_pvalue")


# --------------------------------------------------------------------------- #
# Backtest por umbrales                                                        #
# --------------------------------------------------------------------------- #
def backtest(sscore: pd.DataFrame, halflife: pd.DataFrame, stk: pd.DataFrame,
             open_th: float = 1.25, close_th: float = 0.5,
             hl_max: float = 8.0, cost_bps: float = 0.0, ppy: int = 52):
    """
    Maquina de estados por nombre. Posicion contraria al s-score (reversion).
    hl_max: filtro de tradeabilidad -> solo abrir si la half-life corriente < hl_max.
    Pesos = equiponderado entre posiciones activas. w(t-1) captura retorno(t).
    """
    names = sscore.columns
    pos = pd.DataFrame(0.0, index=sscore.index, columns=names)
    state = {n: 0 for n in names}
    for t in sscore.index:
        for n in names:
            s = sscore.loc[t, n]
            hl = halflife.loc[t, n]
            if np.isnan(s):
                pos.loc[t, n] = state[n]
                continue
            if state[n] == 0:
                tradeable = (not np.isnan(hl)) and (hl < hl_max)
                if tradeable and s > open_th:
                    state[n] = -1
                elif tradeable and s < -open_th:
                    state[n] = +1
            else:
                if abs(s) < close_th:
                    state[n] = 0
            pos.loc[t, n] = state[n]

    npos = pos.abs().sum(axis=1).replace(0, np.nan)
    w = pos.div(npos, axis=0).fillna(0)
    turn = (w - w.shift()).abs().sum(axis=1)
    pnl = (w.shift() * stk).sum(axis=1) - turn * cost_bps / 1e4
    pnl = pnl.dropna()
    return {
        "sharpe_ann": pnl.mean() / pnl.std() * np.sqrt(ppy) if pnl.std() > 0 else np.nan,
        "ann_ret": pnl.mean() * ppy,
        "turnover_ann": turn.mean() * ppy,
        "time_invested": (pos.abs().sum(axis=1) > 0).mean(),
        "pnl": pnl,
    }


# --------------------------------------------------------------------------- #
# Demo                                                                         #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from idiosyncratic_momentum import load_data, rolling_residuals, periods_per_month

    PATH = "/mnt/user-data/uploads/Momentum_Idiosincratico_V__1_.xlsx"
    stk, mkt, sec = load_data(PATH)
    ppm = periods_per_month(stk.index)
    resid = rolling_residuals(stk, mkt, sec, beta_window=int(round(36 * ppm)))

    adf = adf_diagnostics(resid)
    sscore, hl = rolling_sscore(resid, window=52)

    print("Estacionariedad y half-life del residuo:")
    for n in resid.columns:
        print(f"  {n:>6}  ADF p={adf[n]:.3f}  HL_med={hl[n].median():.1f}w")

    for c in (0, 50, 100):
        r = backtest(sscore, hl, stk, cost_bps=c)
        print(f"\ncost={c:>3}bps  Sharpe={r['sharpe_ann']:.2f}  "
              f"turnover={r['turnover_ann']:.1f}x  "
              f"%invertido={r['time_invested']*100:.0f}%  "
              f"ann_ret={r['ann_ret']*100:.1f}%")
