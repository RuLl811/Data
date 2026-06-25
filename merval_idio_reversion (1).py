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

# =========================================================================== #
# 6. GRAFICOS                                                                  #
# =========================================================================== #
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

def _positions(s_series, hl_series, open_th, close_th, hl_max):
    state = 0; out = pd.Series(0, index=s_series.index, dtype=int)
    for t in s_series.index:
        s = s_series[t]; h = hl_series[t]
        if not np.isnan(s):
            if state == 0:
                ok = (not np.isnan(h)) and h < hl_max
                if ok and s > open_th: state = -1
                elif ok and s < -open_th: state = +1
            elif abs(s) < close_th: state = 0
        out[t] = state
    return out

def plot_sscore(sscore, hl, open_th=1.25, close_th=0.5, hl_max=8.0,
                outfile="sscore_chart.png", footnote=None):
    """s-score por nombre en grilla pareja (todos del mismo tamano), con bandas de
       decision, sombreado de posiciones y marcas de entrada/salida."""
    C_LONG, C_SHORT, C_LINE = "#4C72B0", "#DD8452", "#222222"; YL = 4.2
    names = list(sscore.columns)
    pos = {n: _positions(sscore[n], hl[n], open_th, close_th, hl_max) for n in names}
    ntr = {n: int((pos[n].diff().abs() > 0).sum()) for n in names}

    def shade_mark(ax, s, p):
        ax.fill_between(s.index, -YL, YL, where=(p.values == 1), color=C_LONG, alpha=0.12, step="mid", lw=0)
        ax.fill_between(s.index, -YL, YL, where=(p.values == -1), color=C_SHORT, alpha=0.12, step="mid", lw=0)
        ch = p.diff().fillna(0)
        for e in p.index[(ch != 0) & (p != 0)]:
            ax.scatter(e, s[e], marker=("^" if p[e] == 1 else "v"), s=42,
                       color=(C_LONG if p[e] == 1 else C_SHORT), zorder=5, edgecolor="white", lw=0.5)
        ext = p.index[(ch != 0) & (p == 0)]
        ax.scatter(ext, s.reindex(ext), marker="x", s=32, color=C_LINE, zorder=5, lw=1.1)

    plt.rcParams.update({'font.size': 11, 'axes.titleweight': 'bold'})
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True, sharey=True)
    for ax, n in zip(axes.ravel(), names):
        sn = sscore[n].dropna(); pn = pos[n].reindex(sn.index)
        ax.plot(sn.index, sn.clip(-YL, YL), color=C_LINE, lw=0.9, zorder=4)
        shade_mark(ax, sn.clip(-YL, YL), pn)
        for y, st in [(open_th, "--"), (-open_th, "--"), (close_th, ":"), (-close_th, ":"), (0, "-")]:
            ax.axhline(y, ls=st, color=("#888" if y == 0 else C_LINE), lw=0.8, alpha=0.6)
        ax.set_ylim(-YL, YL); ax.set_title(f"{n}   HL~{hl[n].median():.1f}s   {ntr[n]} ops", fontsize=11)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_ylabel("s-score")
    leg = [Line2D([], [], color=C_LINE, lw=1.1, label="s-score"),
           Line2D([], [], ls="--", color=C_LINE, label=f"apertura +-{open_th}"),
           Line2D([], [], ls=":", color=C_LINE, label=f"cierre +-{close_th}"),
           Line2D([], [], marker="^", ls="", color=C_LONG, label="entra long", mec="white"),
           Line2D([], [], marker="v", ls="", color=C_SHORT, label="entra short", mec="white"),
           Line2D([], [], marker="x", ls="", color=C_LINE, label="cierra")]
    fig.legend(handles=leg, loc="upper center", ncol=6, frameon=True, fontsize=9.5, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle("Senial s-score (OU) -- utilities Merval | azul=long, naranja=short", fontsize=14, y=0.995)
    if footnote:
        fig.text(0.5, 0.004, footnote, ha="center", fontsize=10, color="#7a1f1f", weight="bold")
    plt.tight_layout(rect=[0, 0.02, 1, 0.93])
    plt.savefig(outfile, dpi=150, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"  grafico guardado: {outfile}")

def plot_bands_surface(sscore, hl, resid, cost_bps=50, hl_max=8.0,
                       outfile="bands_sharpe_surface.png"):
    """Superficie de Sharpe real (P&L residuo, neto) sobre (banda entrada a, banda salida q), pooled."""
    S = sscore.values; H = hl.values; R = resid.reindex(sscore.index).values
    T, N = S.shape
    def pooled_sharpe(a, q):
        pos = np.zeros((T, N)); state = np.zeros(N)
        for t in range(T):
            s = S[t]; h = H[t]; valid = ~np.isnan(s)
            flat = (state == 0) & valid; ok = flat & (~np.isnan(h)) & (h < hl_max)
            state[ok & (s > a)] = -1; state[ok & (s < -a)] = 1
            state[(state == 1) & valid & (s >= q)] = 0; state[(state == -1) & valid & (s <= -q)] = 0
            pos[t] = state
        w = np.zeros((T, N)); npos = np.abs(pos).sum(1); nz = npos > 0; w[nz] = pos[nz] / npos[nz, None]
        wp = np.vstack([np.zeros(N), w[:-1]])
        pnl = np.nansum(wp * R, 1) - np.abs(w - wp).sum(1) * cost_bps / 1e4
        pnl = pnl[~np.isnan(pnl)]
        return pnl.mean() / pnl.std() * np.sqrt(52) if pnl.std() > 0 else np.nan
    A = np.round(np.arange(0.5, 2.51, 0.25), 2); Q = np.round(np.arange(-0.75, 2.51, 0.25), 2)
    Z = np.full((len(Q), len(A)), np.nan)
    for i, q in enumerate(Q):
        for j, a in enumerate(A):
            if q <= a: Z[i, j] = pooled_sharpe(a, q)
    bi = np.unravel_index(np.nanargmax(Z), Z.shape)
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(Z, origin='lower', aspect='auto', cmap='RdBu_r', vmin=-0.4, vmax=0.4,
                   extent=[A[0]-0.125, A[-1]+0.125, Q[0]-0.125, Q[-1]+0.125])
    fig.colorbar(im, ax=ax).set_label('Sharpe anual (residuo, neto)')
    ax.plot([A[0], A[-1]], [A[0], A[-1]], ls='--', color='#333', lw=1, label='q=a (flip)')
    ax.axhline(0, color='#666', lw=0.8, ls=':')
    ax.scatter(A[bi[1]], Q[bi[0]], marker='*', s=320, color='#111', zorder=5,
               label=f'max pooled a={A[bi[1]]},q={Q[bi[0]]} (ojo: posible sobreajuste)')
    ax.scatter(1.25, -0.5, marker='X', s=140, color='#000', zorder=5, label='regla AL (1.25/-0.5)')
    ax.set_xlabel('banda de entrada  a'); ax.set_ylabel('banda de salida  q')
    ax.set_title('Sharpe real sobre (entrada, salida) -- pooled\nsuperficie ruidosa: la esquina AL es mala, el resto es marginal', fontsize=11, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9, framealpha=0.95)
    plt.tight_layout(); plt.savefig(outfile, dpi=150, bbox_inches='tight', facecolor='white'); plt.close(fig)
    print(f"  grafico guardado: {outfile}")

if __name__ == "__main__":
    stk, mkt, sec = load_data(PATH)
    ppm = periods_per_month(stk.index)
    print(f"Frecuencia ~{ppm:.2f}/mes | {len(stk)} obs | {stk.index.min().date()} -> {stk.index.max().date()}")
    resid = rolling_residuals(stk, mkt, sec, beta_window=int(round(36 * ppm)))

    print("\n[Momentum residual]")
    for months in (1, 3, 6):
        J = max(2, int(round(months * ppm)))
        sig = momentum_signal(resid, J=J, skip=1); fwd = stk.shift(-1)
        ics = [spearmanr(sig.loc[t].dropna(), fwd.loc[t].reindex(sig.loc[t].dropna().index)).correlation
               for t in sig.index if sig.loc[t].notna().sum() >= 4 and fwd.loc[t].notna().sum() >= 4]
        ic = pd.Series(ics).dropna()
        print(f"  {months}m: IC={ic.mean():+.4f} t={ic.mean()/(ic.std()/np.sqrt(len(ic))):.2f}")

    print("\n[s-score OU]")
    adf = adf_diagnostics(resid); sscore, hl = rolling_sscore(resid, window=52)
    for n in NAMES: print(f"  {n:>6} ADF p={adf[n]:.3f} HL_med={hl[n].median():.1f}")
    sh = {}
    for c in (0, 50, 100):
        r = backtest_sscore(sscore, hl, resid, cost_bps=c); sh[c] = r['sharpe_ann']
        print(f"  resid P&L cost={c:>3}bp Sharpe={r['sharpe_ann']:.2f} turn={r['turnover_ann']:.1f}x")

    print("\n[Graficos]")
    foot = (f"DECISION -- P&L hedgeado neto, Sharpe: {sh[0]:.2f} (0bp) / {sh[50]:.2f} (50bp) / {sh[100]:.2f} (100bp)  |  "
            "estacionario pero economicamente delgado  |  IN-SAMPLE, falta validacion OOS (purged k-fold)")
    plot_sscore(sscore, hl, outfile="sscore_chart.png", footnote=foot)
    plot_bands_surface(sscore, hl, resid, cost_bps=50, outfile="bands_sharpe_surface.png")
