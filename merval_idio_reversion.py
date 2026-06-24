r"""
merval_idio_reversion.py
========================
Pipeline completo de momentum/reversion idiosincratica para utilities del Merval,
en UN solo archivo. Combina:
  (1) extraccion de residuos por regresion rodante (mercado + sector-ex-accion)
  (2) senial de momentum (suma estandarizada de residuos)
  (3) senial estadistica de reversion: s-score Ornstein-Uhlenbeck (Avellaneda-Lee 2010)
  (4) solver de bandas optimas por nombre (Bertram 2010): first-passage-time analitico
      + Monte Carlo para el Sharpe.

Uso:
    python merval_idio_reversion.py
Editar PATH abajo si el excel esta en otra ruta.

Dependencias: numpy, pandas, statsmodels, scipy.
    pip install numpy pandas statsmodels scipy
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from statsmodels.tsa.stattools import adfuller
from scipy.stats import spearmanr, norm
from scipy.integrate import quad

PATH = "Momentum_Idiosincratico_V__1_.xlsx"   # <-- editar si hace falta
NAMES = ["CEPU", "EDN", "TRAN", "TGNO4", "TGSU2", "METR"]


# =========================================================================== #
# 1. CARGA                                                                     #
# =========================================================================== #
def load_data(path: str = PATH):
    """Layout: 2 filas de header, 3 bloques. Devuelve (stk, mkt, sec) por fecha.
       'Retorno Sector' ya es sector-EX-accion (no reconstruir)."""
    raw = pd.read_excel(path, sheet_name="Sheet1", header=None)
    dates = pd.to_datetime(raw.iloc[2:, 0].values)
    sec = raw.iloc[2:, 1:7].astype(float);  sec.columns = NAMES; sec.index = dates
    mkt = raw.iloc[2:, 8].astype(float);     mkt.index = dates;  mkt.name = "mkt"
    stk = raw.iloc[2:, 10:16].astype(float); stk.columns = NAMES; stk.index = dates
    return stk, mkt, sec


def periods_per_month(index) -> float:
    """Detecta frecuencia desde el indice. Semanal -> ~4.33, diario -> ~21."""
    med_gap = pd.Series(index).diff().dt.days.median()
    return 30.4 / med_gap


# =========================================================================== #
# 2. RESIDUOS RODANTES                                                         #
# =========================================================================== #
def rolling_residuals(stk, mkt, sec, beta_window, orthogonalize_sector=False):
    """e_{i,t} = r_i - X_t @ beta_t, betas estimadas en ventana que termina en t.
       min_count evita que los NaN del warm-up cuenten como 0 (bug que infla el IC)."""
    resid = pd.DataFrame(index=stk.index, columns=stk.columns, dtype=float)
    for n in stk.columns:
        sec_factor = sec[n].rename("sec")
        if orthogonalize_sector:
            d = pd.concat([mkt, sec_factor], axis=1).dropna()
            delta = np.polyfit(d["mkt"], d["sec"], 1)[0]
            sec_factor = (sec[n] - delta * mkt).rename("sec")
        X = sm.add_constant(pd.concat([mkt, sec_factor], axis=1))
        y = stk[n]
        ro = RollingOLS(y, X, window=beta_window, min_nobs=beta_window).fit()
        fitted = (ro.params * X).sum(axis=1, min_count=X.shape[1])
        resid[n] = y - fitted
    return resid


# =========================================================================== #
# 3. SENIAL DE MOMENTUM (suma estandarizada de residuos)                       #
# =========================================================================== #
def momentum_signal(resid, J, skip=1, standardize=True):
    """Suma de residuos sobre J periodos salteando los ultimos `skip`.
       standardize -> divide por desvio*sqrt(J) (IR, estilo Blitz-Huij-Martens)."""
    out = {}
    for n in resid.columns:
        s = resid[n].shift(skip)
        csum = s.rolling(J).sum()
        out[n] = csum / (s.rolling(J).std() * np.sqrt(J)) if standardize else csum
    return pd.DataFrame(out)


# =========================================================================== #
# 4. SENIAL ESTADISTICA: s-score Ornstein-Uhlenbeck (Avellaneda-Lee)           #
# =========================================================================== #
def ou_sscore(window):
    """Ajusta OU al residuo acumulado de la ventana. Devuelve dict o None.
       AR(1): X_{n+1}=a+b X_n+z -> kappa=-ln(b), m=a/(1-b), s=(X_t-m)/sigma_eq."""
    x = window - window.mean()
    X = np.cumsum(x); Xl, Xn = X[:-1], X[1:]
    b, a = np.polyfit(Xl, Xn, 1)
    if not (0 < b < 1):
        return None
    kappa = -np.log(b); m = a / (1 - b)
    eps = Xn - (a + b * Xl)
    sigma_eq = np.sqrt(np.var(eps) / (1 - b ** 2))
    if sigma_eq == 0:
        return None
    return {"s": (X[-1] - m) / sigma_eq, "kappa": kappa, "halflife": np.log(2) / kappa}


def rolling_sscore(resid, window=52):
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


def adf_diagnostics(resid, maxlag=4):
    """p-value ADF por nombre. p chico -> residuo estacionario (revierte)."""
    return pd.Series({n: adfuller(resid[n].dropna(), maxlag=maxlag, autolag=None)[1]
                      for n in resid.columns}, name="adf_pvalue")


def backtest_sscore(sscore, halflife, stream, open_th=1.25, close_th=0.5,
                    hl_max=8.0, cost_bps=0.0, ppy=52):
    """Maquina de estados por nombre, posicion contraria al s-score (reversion).
       `stream`: usar `stk` para P&L total, `resid` para P&L hedgeado (recomendado).
       hl_max: filtro de tradeabilidad -> solo abrir si half-life corriente < hl_max."""
    names = sscore.columns
    pos = pd.DataFrame(0.0, index=sscore.index, columns=names)
    state = {n: 0 for n in names}
    for t in sscore.index:
        for n in names:
            s = sscore.loc[t, n]; hl = halflife.loc[t, n]
            if np.isnan(s):
                pos.loc[t, n] = state[n]; continue
            if state[n] == 0:
                ok = (not np.isnan(hl)) and (hl < hl_max)
                if ok and s > open_th: state[n] = -1
                elif ok and s < -open_th: state[n] = +1
            elif abs(s) < close_th:
                state[n] = 0
            pos.loc[t, n] = state[n]
    npos = pos.abs().sum(axis=1).replace(0, np.nan)
    w = pos.div(npos, axis=0).fillna(0)
    turn = (w - w.shift()).abs().sum(axis=1)
    pnl = ((w.shift() * stream).sum(axis=1) - turn * cost_bps / 1e4).dropna()
    return {"sharpe_ann": pnl.mean() / pnl.std() * np.sqrt(ppy) if pnl.std() > 0 else np.nan,
            "ann_ret": pnl.mean() * ppy, "turnover_ann": turn.mean() * ppy,
            "time_invested": (pos.abs().sum(axis=1) > 0).mean(), "pnl": pnl}


# =========================================================================== #
# 5. SOLVER DE BANDAS OPTIMAS (Bertram 2010)                                   #
# =========================================================================== #
def g(alpha, beta):
    """E[T(alpha->beta)] = g/kappa.  g = sqrt(2pi) ∫ e^{x^2/2} Phi(x) dx (sigma_eq=1)."""
    val, _ = quad(lambda x: np.exp(x ** 2 / 2.0) * norm.cdf(x), alpha, beta, limit=200)
    return np.sqrt(2 * np.pi) * val


def ret_per_time_flip(a, q, kappa, sigma_eq, c_s):
    """Retorno esperado/periodo (analitico, continuo) para entrar -a / salir q, simetrico."""
    return ((a + q) - c_s) * sigma_eq / (g(-a, a) / kappa)


def fit_params(resid_col, window=52):
    """kappa (por periodo) y sigma_eq medianos de los ajustes OU rodantes."""
    r = resid_col.values; ks, ses = [], []
    for i in range(window, len(r)):
        w = r[i - window:i]
        if np.isnan(w).any(): continue
        X = np.cumsum(w - w.mean()); Xl, Xn = X[:-1], X[1:]
        b, a = np.polyfit(Xl, Xn, 1)
        if not (0 < b < 1): continue
        se = np.sqrt(np.var(Xn - (a + b * Xl)) / (1 - b ** 2))
        if se > 0:
            ks.append(-np.log(b)); ses.append(se)
    return np.median(ks), np.median(ses)


def simulate(kappa, sigma_eq, cost_oneway_bps, grid_a, grid_q, n=200_000, seed=1):
    """Monte Carlo del OU estandar. Devuelve Sharpe anual neto sobre la grilla (a,q)."""
    rng = np.random.default_rng(seed)
    phi = np.exp(-kappa); sd = np.sqrt(1 - np.exp(-2 * kappa))
    s = np.empty(n); s[0] = rng.standard_normal(); z = rng.standard_normal(n)
    for t in range(1, n):
        s[t] = phi * s[t - 1] + sd * z[t]
    cost = cost_oneway_bps / 1e4
    combos = [(a, q) for a in grid_a for q in grid_q if -a < q <= a + 1e-9]
    A = np.array([a for a, _ in combos]); Q = np.array([q for _, q in combos])
    state = np.zeros(len(combos)); pos_prev = np.zeros(len(combos))
    summ = np.zeros(len(combos)); summ2 = np.zeros(len(combos)); cnt = 0
    for t in range(1, n):
        pnl = pos_prev * (s[t] - s[t - 1]) * sigma_eq
        st = s[t]; new = state.copy(); flat = state == 0
        new[flat & (st <= -A)] = 1.0; new[flat & (st >= A)] = -1.0
        new[(state == 1) & (st >= Q)] = 0.0; new[(state == -1) & (st <= -Q)] = 0.0
        pnl -= np.abs(new - state) * cost
        summ += pnl; summ2 += pnl ** 2; cnt += 1
        pos_prev = new.copy(); state = new
    mean = summ / cnt; var = summ2 / cnt - mean ** 2
    sharpe = np.where(var > 0, mean / np.sqrt(var) * np.sqrt(52), np.nan)
    best = int(np.nanargmax(sharpe))
    return {"combos": combos, "sharpe": sharpe, "ann": mean * 52,
            "best": (A[best], Q[best]), "best_sharpe": sharpe[best]}


# =========================================================================== #
# DEMO                                                                         #
# =========================================================================== #
if __name__ == "__main__":
    stk, mkt, sec = load_data(PATH)
    ppm = periods_per_month(stk.index)
    print(f"Frecuencia ~{ppm:.2f} periodos/mes | {len(stk)} obs | "
          f"{stk.index.min().date()} -> {stk.index.max().date()}")

    resid = rolling_residuals(stk, mkt, sec, beta_window=int(round(36 * ppm)))

    # --- momentum ---
    print("\n[Momentum residual]")
    for months in (1, 3, 6):
        J = max(2, int(round(months * ppm)))
        sig = momentum_signal(resid, J=J, skip=1, standardize=True)
        fwd = stk.shift(-1)
        ics = [spearmanr(sig.loc[t].dropna(), fwd.loc[t].reindex(sig.loc[t].dropna().index)).correlation
               for t in sig.index if sig.loc[t].notna().sum() >= 4 and fwd.loc[t].notna().sum() >= 4]
        ic = pd.Series(ics).dropna()
        print(f"  {months}m: IC={ic.mean():+.4f} t={ic.mean()/(ic.std()/np.sqrt(len(ic))):.2f}")

    # --- s-score ---
    print("\n[s-score OU]  ADF y half-life:")
    adf = adf_diagnostics(resid); sscore, hl = rolling_sscore(resid, window=52)
    for n in NAMES:
        print(f"  {n:>6}  ADF p={adf[n]:.3f}  HL_med={hl[n].median():.1f}")
    for c in (0, 50, 100):
        r = backtest_sscore(sscore, hl, resid, cost_bps=c)   # P&L hedgeado
        print(f"  resid P&L cost={c:>3}bp  Sharpe={r['sharpe_ann']:.2f}  turn={r['turnover_ann']:.1f}x")

    # --- bandas optimas (modelo) por nombre ---
    print("\n[Bandas optimas modelo OU]  (validar SIEMPRE contra datos reales)")
    for n in NAMES:
        k, se = fit_params(resid[n]); c_s = (50 / 1e4) / se
        print(f"  {n:>6}  kappa={k:.3f}  HL={np.log(2)/k:.1f}  sigma_eq={se:.3f}  c~={c_s:.3f}")
