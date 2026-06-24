r"""
idiosyncratic_momentum.py
=========================
Momentum idiosincratico para utilities del Merval.

Pipeline:
  1) Regresion rodante  r_i = a + b_mkt*r_mkt + b_sec*r_(sec\i) + e_i   (RollingOLS)
  2) Senial = suma estandarizada (IR) de residuos sobre ventana de formacion, con skip
  3) Evaluacion: rank IC (Fama-MacBeth-like), long-short, forward returns

Notas de diseno:
  - La data del archivo es SEMANAL (paso 7d). El modulo detecta la frecuencia
    y traduce meses -> periodos. Si tenes la serie diaria, pasala: betas mas estables.
  - El bloque "Retorno Sector" del excel YA es sector-ex-accion (no reconstruir).
  - Corr(MERVAL, sector) ~ 0.90: el residuo queda bien identificado, los betas NO.
    Flag `orthogonalize_sector=True` para betas interpretables (no cambia el residuo).
  - N=6: quintiles/rank-IC son ruidosos por construccion. Ver evaluate() y el README.

Autor: para el framework de Ruben. Estilo: validacion empirica > apelacion teorica.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from scipy.stats import spearmanr

NAMES = ["CEPU", "EDN", "TRAN", "TGNO4", "TGSU2", "METR"]


# --------------------------------------------------------------------------- #
# 1. Carga                                                                     #
# --------------------------------------------------------------------------- #
def load_data(path: str):
    """Lee el layout especifico: 2 filas de header, 3 bloques.
       Devuelve (stk, mkt, sec) alineados por fecha."""
    raw = pd.read_excel(path, sheet_name="Sheet1", header=None)
    dates = pd.to_datetime(raw.iloc[2:, 0].values)
    sec = raw.iloc[2:, 1:7].astype(float);  sec.columns = NAMES; sec.index = dates
    mkt = raw.iloc[2:, 8].astype(float);     mkt.index = dates;  mkt.name = "mkt"
    stk = raw.iloc[2:, 10:16].astype(float); stk.columns = NAMES; stk.index = dates
    return stk, mkt, sec


def periods_per_month(index: pd.DatetimeIndex) -> float:
    """Detecta frecuencia desde el indice. Semanal -> ~4.33, diario -> ~21."""
    med_gap = pd.Series(index).diff().dt.days.median()
    return 30.4 / med_gap


# --------------------------------------------------------------------------- #
# 2. Residuos rodantes                                                         #
# --------------------------------------------------------------------------- #
def rolling_residuals(stk, mkt, sec, beta_window: int,
                      orthogonalize_sector: bool = False) -> pd.DataFrame:
    """
    Para cada nombre, RollingOLS con ventana que TERMINA en t.
    Residuo e_{i,t} = r_{i,t} - X_t @ beta_t  -> solo usa info <= t (sin look-ahead).

    CRITICO: usar min_count en la suma para que los NaN del warm-up de los betas
    NO se traten como 0 (ese bug genera residuos espurios = retorno crudo antes de
    tener ventana completa, e infla el IC).
    """
    resid = pd.DataFrame(index=stk.index, columns=stk.columns, dtype=float)
    for n in stk.columns:
        sec_factor = sec[n].rename("sec")
        if orthogonalize_sector:
            # sector ortogonal al mercado: deja betas interpretables, no cambia e
            d = pd.concat([mkt, sec_factor], axis=1).dropna()
            delta = np.polyfit(d["mkt"], d["sec"], 1)[0]
            sec_factor = (sec[n] - delta * mkt).rename("sec")
        X = sm.add_constant(pd.concat([mkt, sec_factor], axis=1))
        y = stk[n]
        ro = RollingOLS(y, X, window=beta_window, min_nobs=beta_window).fit()
        fitted = (ro.params * X).sum(axis=1, min_count=X.shape[1])  # <- min_count
        resid[n] = y - fitted
    return resid


# --------------------------------------------------------------------------- #
# 3. Senial                                                                    #
# --------------------------------------------------------------------------- #
def momentum_signal(resid: pd.DataFrame, J: int, skip: int = 1,
                    standardize: bool = True) -> pd.DataFrame:
    """
    Suma de residuos sobre ventana de formacion de J periodos, salteando los
    ultimos `skip` (reversion de corto plazo / bid-ask bounce).
    standardize=True -> divide por desvio de residuos * sqrt(J)  (IR, estilo BHM).
    Sin estandarizar, los nombres mas volatiles dominan la senial.
    """
    out = {}
    for n in resid.columns:
        s = resid[n].shift(skip)
        csum = s.rolling(J).sum()
        if standardize:
            cstd = s.rolling(J).std() * np.sqrt(J)
            out[n] = csum / cstd
        else:
            out[n] = csum
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# 4. Evaluacion                                                                #
# --------------------------------------------------------------------------- #
def rank_ic(signal: pd.DataFrame, fwd_ret: pd.DataFrame, min_names: int = 4):
    """Spearman cross-seccional senial(t) vs retorno forward(t). N=6 -> ruidoso."""
    ics = {}
    for t in signal.index:
        a, b = signal.loc[t], fwd_ret.loc[t]
        m = a.notna() & b.notna()
        if m.sum() >= min_names:
            ics[t] = spearmanr(a[m], b[m]).correlation
    ic = pd.Series(ics).dropna()
    n = len(ic)
    return {
        "ic_mean": ic.mean(), "ic_std": ic.std(), "n": n,
        "t_stat": ic.mean() / (ic.std() / np.sqrt(n)) if n > 1 else np.nan,
        "ir": ic.mean() / ic.std() if ic.std() > 0 else np.nan,
        "series": ic,
    }


def long_short(signal: pd.DataFrame, fwd_ret: pd.DataFrame, k: int = 2,
               cost_bps: float = 0.0, ppy: int = 52):
    """
    Long top-k / short bottom-k, equiponderado, rebalanceo cada periodo.
    cost_bps: costo round-trip por pata (spread+comision) aplicado al turnover.
    Con N=6, k=2 es el maximo razonable. Devuelve serie de PnL y stats.
    """
    pnl, w_prev = {}, pd.Series(0.0, index=signal.columns)
    for t in signal.index:
        r = signal.loc[t]; f = fwd_ret.loc[t]
        valid = r.notna()
        if valid.sum() < 2 * k:
            continue
        longs = r[valid].nlargest(k).index
        shorts = r[valid].nsmallest(k).index
        w = pd.Series(0.0, index=signal.columns)
        w[longs] = 1.0 / k; w[shorts] = -1.0 / k
        gross = (w * f).sum()
        turnover = (w - w_prev).abs().sum()
        pnl[t] = gross - turnover * cost_bps / 1e4
        w_prev = w
    pnl = pd.Series(pnl).dropna()
    return {
        "ret_mean": pnl.mean(), "vol": pnl.std(),
        "sharpe_ann": pnl.mean() / pnl.std() * np.sqrt(ppy) if pnl.std() > 0 else np.nan,
        "hit_rate": (pnl > 0).mean(), "n": len(pnl), "pnl": pnl,
    }


def fama_macbeth(signal: pd.DataFrame, fwd_ret: pd.DataFrame):
    """
    Pendiente cross-seccional periodo a periodo (fwd ~ a + g*signal), luego
    promedio temporal con t de FM. Mas potente que el IC para N chico porque
    usa la magnitud, no solo el orden. Igual: 6 puntos por corte = poca senial.
    """
    gammas = {}
    for t in signal.index:
        a, b = signal.loc[t], fwd_ret.loc[t]
        m = a.notna() & b.notna()
        if m.sum() >= 3:
            X = sm.add_constant(a[m].values)
            gammas[t] = sm.OLS(b[m].values, X).fit().params[1]
    g = pd.Series(gammas).dropna()
    return {"gamma_mean": g.mean(),
            "t_fm": g.mean() / (g.std() / np.sqrt(len(g))) if len(g) > 1 else np.nan,
            "n": len(g)}


# --------------------------------------------------------------------------- #
# Demo                                                                         #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    PATH = "/mnt/user-data/uploads/Momentum_Idiosincratico_V__1_.xlsx"
    stk, mkt, sec = load_data(PATH)
    ppm = periods_per_month(stk.index)
    print(f"Frecuencia: ~{ppm:.2f} periodos/mes  ({'semanal' if ppm<10 else 'diario'})")

    BETA_WINDOW = int(round(36 * ppm))   # ~3 anios, fiel a Blitz-Huij-Martens
    resid = rolling_residuals(stk, mkt, sec, beta_window=BETA_WINDOW)

    for months in (1, 3, 6):
        J = max(2, int(round(months * ppm)))
        sig = momentum_signal(resid, J=J, skip=1, standardize=True)
        fwd = stk.shift(-1)               # retorno del periodo siguiente
        ic = rank_ic(sig, fwd)
        ls = long_short(sig, fwd, k=2, cost_bps=0)
        fm = fama_macbeth(sig, fwd)
        print(f"\n[{months}m | J={J}w]  IC={ic['ic_mean']:+.4f} (t={ic['t_stat']:.2f}, "
              f"N={ic['n']})  | L2S2 Sharpe={ls['sharpe_ann']:.2f}  | "
              f"FM gamma t={fm['t_fm']:.2f}")
