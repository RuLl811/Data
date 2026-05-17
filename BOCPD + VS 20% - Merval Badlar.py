"""
================================================================================
MERVAL CCL/BADLAR — MODELO FCI v2 — SEÑAL CON HISTÉRESIS + COOLDOWN
================================================================================

Resultados backtesting (2010–2026, rebalanceo diario):


  ─────────────────────────────────────────────────────────────────
Arquitectura completa:
  [1] Señal       BOCPD (λ=21d) — detecta quiebres de régimen online
  [2] Histéresis  Entrada: pb > 0.70  |  Salida: pb < 0.35
  [3] Cooldown    14 días sin re-entrada tras cada SELL
  [4] Sizing      w(t) = señal × min(σ_target / σ_ewma(t), 1.0)
  [5] Filtro      umbral mínimo 5pp
  [6] Rebalanceo  semanal (viernes)
  [7] Cash        5% anual sobre porción no invertida
================================================================================
"""
import webbrowser
import numpy as np
import pandas as pd
# from pandas_ta import df
from scipy import stats, special, optimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
import warnings
import os
import sys
import argparse
import quantstats as qs

warnings.filterwarnings('ignore')
np.random.seed(42)

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — PARÁMETROS
# ══════════════════════════════════════════════════════════════════════════════

class Config:

    # ── Datos ────────────────────────────────────────────────────────────────
    DATA_PATH   = r"C:\Users\rumtl\PycharmProjects\Estrategias\Data\merval.xlsx"
    SHEET_NAME  = 'exc_ret_merval'
    OUTPUT_DIR  = r'C:\Users\rumtl\PycharmProjects\Estrategias\RegimeSwitching + MacroSignals\BOCPD\output'

    # ── BOCPD ────────────────────────────────────────────────────────────────
    LAM_BOCPD    = 21.0    # expected run length (días). Calibrado en validación.
    BOCPD_M0     = 0.0     # prior media de los retornos dentro de un run
    BOCPD_V0     = 1.0     # confianza en la prior de media (mayor = más conservador)
    BOCPD_A0     = 1.0     # forma del prior InvGamma de varianza
    BOCPD_PRUNE  = 1e-6    # umbral de poda: eliminar hipótesis de run < este valor

    # ── Volatility Scaling ───────────────────────────────────────────────────
    VOL_TARGET   = 0.20    # volatilidad objetivo anualizada (20%)
    EWMA_SPAN    = 21      # span del EWMA para estimación de vol condicional
    VOL_FLOOR    = 0.035    # vol mínima (evita divisiones extremas en días de baja vol)
    LEVERAGE_CAP = 1.0     # sin apalancamiento — máximo 100% equity

    # ── Señal con histéresis (MEJORA M1 + M2) ───────────────────────────────
    THRESH_BUY   = 0.65    # prob_bull mínima para ENTRAR al mercado

    THRESH_SELL  = 0.35    # prob_bull máxima para SALIR del mercado

    BULL_THRESH  = THRESH_BUY    # alias de compatibilidad con LiveSignal

    # ── Cooldown post-sell (MEJORA M3) ───────────────────────────────────────
    COOLDOWN_DAYS = 14    # días sin permitir re-entrada después de un SELL

    # ── Ejecución ────────────────────────────────────────────────────────────
    MIN_CHANGE   = 0.05    # mínimo cambio de peso para ejecutar rebalanceo (5pp)
    REBAL_FREQ   = 'W-FRI' # frecuencia de rebalanceo semanal (viernes)

    # ── Costos y cash ────────────────────────────────────────────────────────
    TC_BPS       = 15      # costos de transacción
    CASH_ANN     = 0.00    # retorno anual del cash

    # ── Burn-in y calibración ────────────────────────────────────────────────
    MIN_TRAIN    = 504     # período mínimo de burn-in (≈2 años de datos diarios)

    # ── Bootstrap y figuras ──────────────────────────────────────────────────
    N_BOOT       = 1000
    BLOCK_SIZE   = 21

    # ── Constantes ───────────────────────────────────────────────────────────
    ANN          = 252
    TC           = TC_BPS / 10_000

    # ── Styling ──────────────────────────────────────────────────────────────
    DARK_THEME   = True

    EVENTS = {
        '2011-10': 'CEPO I',   '2014-01': 'Deval.',
        '2015-12': 'Macri',    '2019-08': 'PASO',
        '2020-03': 'COVID',    '2023-12': 'Milei',
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — CARGA DE DATOS
# ══════════════════════════════════════════════════════════════════════════════

def load_data(path: str = Config.DATA_PATH,
              sheet: str = Config.SHEET_NAME) -> pd.DataFrame:
    """
    Carga la serie de precios desde el archivo Excel.

    Espera columnas: [Dates/fecha, precio] en cualquier orden.
    Retorna DataFrame con índice DatetimeIndex y columnas:
      Price   : precio de cierre
      LogRet  : retorno logarítmico diario (shift(1), primer valor = NaN)
    """
    df = pd.read_excel(path, sheet_name=sheet, engine='openpyxl')

    # Detectar columnas automáticamente
    df.columns = df.columns.str.strip()
    date_col  = df.columns[0]
    price_col = df.columns[1]

    df[date_col] = pd.to_datetime(df[date_col])
    df = (df.rename(columns={date_col: 'Date', price_col: 'Price'})
            .set_index('Date')
            .sort_index())
    df['LogRet'] = np.log(df['Price'] / df['Price'].shift(1))
    df = df.dropna(subset=['LogRet'])

    return df

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — BOCPD (Bayesian Online Change Point Detection)
# ══════════════════════════════════════════════════════════════════════════════

def bocpd_filter(
    obs:    np.ndarray,
    lam:    float = Config.LAM_BOCPD,
    m0:     float = Config.BOCPD_M0,
    v0:     float = Config.BOCPD_V0,
    a0:     float = Config.BOCPD_A0,
    b0:     float = None,
    prune:  float = Config.BOCPD_PRUNE,
) -> np.ndarray:
    """
    BOCPD causal y vectorizado (Adams & MacKay, 2007).

    Modelo generativo:
      · Los datos se dividen en runs separados por change points.
      · Dentro de cada run: y_t ~ t(2a, m, sqrt(b(v+1)/(av)))
        (predictiva Student-t bajo prior conjugado Normal-InverseGamma).
      · Hazard de quiebre: H = 1/λ por paso.

    Inferencia:
      · Mantiene P(r_t = k | y_{1:t}): distribución sobre el run length actual.
      · 100% causal: P(r_t | y_{1:t}) no usa datos futuros.
      · La predictiva Student-t captura fat tails automáticamente
        cuando el run tiene pocos datos (incertidumbre alta sobre σ²).

    Retorna:
      prob_bull[t] = P(μ_run > 0 | y_{1:t})
        → probabilidad de que el retorno esperado del run actual sea positivo.
    """
    if b0 is None:
        # b0 calibrado en los primeros 2λ datos para anclar la escala a la serie
        b0 = a0 * obs[:max(int(lam * 2), 20)].var()

    H  = 1.0 / lam
    n  = len(obs)

    # Estado inicial: un único run activo
    rp  = np.array([1.0])
    ms  = np.array([m0])
    vs  = np.array([v0])
    as_ = np.array([a0])
    bs  = np.array([b0])

    pb  = np.full(n, 0.5)

    for t in range(n):
        y = obs[t]

        # ── Predictiva Student-t vectorizada ─────────────────────────────
        df_t  = 2.0 * as_                                          # (K,)
        sc_t  = np.sqrt(bs * (vs + 1.0) / (as_ * vs + 1e-10))    # (K,)
        z     = (y - ms) / (sc_t + 1e-10)
        log_p = (special.gammaln((df_t + 1) / 2)
                 - special.gammaln(df_t / 2)
                 - 0.5 * np.log(df_t * np.pi)
                 - np.log(sc_t + 1e-10)
                 - (df_t + 1) / 2 * np.log1p(z * z / df_t))
        pred  = np.exp(np.clip(log_p - log_p.max(), -700, 0))    # estabilizado

        # ── Growth y changepoint ──────────────────────────────────────────
        joint  = rp * pred
        new_rp = np.concatenate([[H * joint.sum()], (1 - H) * joint])
        nc     = new_rp.sum()
        new_rp /= max(nc, 1e-300)

        # ── Actualización NIG vectorizada (conjugada exacta) ──────────────
        v_n = vs + 1.0
        m_n = (vs * ms + y) / v_n
        a_n = as_ + 0.5
        b_n = bs + 0.5 * vs * (y - ms) ** 2 / v_n

        new_ms = np.concatenate([[m0], m_n])
        new_vs = np.concatenate([[v0], v_n])
        new_as = np.concatenate([[a0], a_n])
        new_bs = np.concatenate([[b0], b_n])

        # ── Poda de hipótesis poco probables ─────────────────────────────
        keep   = new_rp >= prune
        keep[np.argmax(new_rp)] = True   # mantener siempre la más probable
        rp  = new_rp[keep]
        ms  = new_ms[keep]
        vs  = new_vs[keep]
        as_ = new_as[keep]
        bs  = new_bs[keep]

        # ── Señal bull: P(μ_run > 0 | y_{1:t}) ──────────────────────────
        # Marginal de μ dado el run: t(2a-1, m, sqrt(b/((a-0.5)v)))
        df_mu   = np.maximum(2.0 * as_ - 1.0, 0.1)
        sc_mu   = np.sqrt(bs / (np.maximum(as_ - 0.5, 0.1) * vs + 1e-10))
        t_stat  = (0.0 - ms) / (sc_mu + 1e-10)
        pb[t]   = float(np.dot(rp, 1.0 - stats.t.cdf(t_stat, df=df_mu)))

    return pb

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4 — VOLATILITY SCALING
# ══════════════════════════════════════════════════════════════════════════════

def ewma_vol(returns: np.ndarray,
             span:    int   = Config.EWMA_SPAN,
             ann:     int   = Config.ANN) -> np.ndarray:
    """
    Volatilidad EWMA a la baja (semi-desviación), causal (no mira hacia adelante).
    Solo considera retornos negativos: σ_down_t = √(EWMA(min(r,0)², span) × ANN).
    """
    down = np.minimum(returns, 0.0)
    ewma_var = pd.Series(down ** 2).ewm(span=span, min_periods=2).mean().values
    ev = np.sqrt(ewma_var * ann)
    # Imputar el primer valor (NaN por min_periods=2)
    fallback = np.sqrt(np.mean(down ** 2)) * np.sqrt(ann)
    ev[0] = fallback
    ev    = np.where(np.isnan(ev), fallback, ev)
    return ev

def position_weights(
    prob_bull:  np.ndarray,
    cond_vol:   np.ndarray,
    vol_tgt:    float = Config.VOL_TARGET,
    lev_cap:    float = Config.LEVERAGE_CAP,
    thresh_buy: float = Config.THRESH_BUY,
    thresh_sell:float = Config.THRESH_SELL,
    cooldown:   int   = Config.COOLDOWN_DAYS,
    min_chg:    float = Config.MIN_CHANGE,
) -> np.ndarray:
    """
    Calcula el peso objetivo causal con histéresis y cooldown post-sell.

    Lógica de señal (M1 + M2 — histéresis):
      · ENTRAR  si prob_bull(t-1) > thresh_buy  Y estamos en cash
      · SALIR   si prob_bull(t-1) < thresh_sell Y estamos en equity
      · MANTENER en cualquier otro caso (zona gris: thresh_sell ≤ pb ≤ thresh_buy)

    Cooldown (M3):
      · Después de cada SELL, se bloquea la re-entrada durante `cooldown` días.
      · El cooldown se cuenta en días calendario del índice (pasos del loop),
        no días naturales, para consistencia con la frecuencia de datos.

    Causalidad:
      · prob_bull[t-1] y cond_vol[t-1] se aplican al retorno de t.
      · El estado previo (prev_w, last_sell_step) es información de t-1.
      · Sin look-ahead bajo ninguna circunstancia.
    """
    n              = len(prob_bull)
    w              = np.zeros(n)
    prev_w         = 0.0
    last_sell_step = -cooldown - 1   # inicializar fuera del período de cooldown

    for t in range(1, n):
        pb_t    = prob_bull[t - 1]
        ev_t    = max(cond_vol[t - 1], Config.VOL_FLOOR)
        vs_mult = min(vol_tgt / ev_t, lev_cap)

        in_cooldown = (t - last_sell_step) < cooldown

        if prev_w == 0.0:
            # Estamos en cash — ¿entramos?
            if pb_t > thresh_buy and not in_cooldown:
                w_raw = vs_mult          # ENTRAR con peso VS
            else:
                w_raw = 0.0              # Mantener en cash
        else:
            # Estamos en equity — ¿salimos?
            if pb_t < thresh_sell:
                w_raw = 0.0              # SALIR
                last_sell_step = t       # registrar para cooldown
            else:
                # Zona gris o señal bull: mantener peso VS actualizado
                w_raw = vs_mult          # actualizar tamaño por cambio de vol

        # Umbral mínimo de cambio: no ejecutar si el cambio es < 5pp
        if abs(w_raw - prev_w) < min_chg:
            w_raw = prev_w

        w[t]   = w_raw
        prev_w = w_raw

    return w


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 5 — MOTOR DE SIMULACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def simulate(
    returns:   np.ndarray,
    prob_bull: np.ndarray,
    cond_vol:  np.ndarray,
    dates:     pd.DatetimeIndex,
    vol_tgt:   float = Config.VOL_TARGET,
    tc:        float = Config.TC,
    cash_ann:  float = Config.CASH_ANN,
    min_chg:   float = Config.MIN_CHANGE,
    rebal_freq: str  = Config.REBAL_FREQ,
) -> dict:
    """
    Simulación completa del portfolio FCI.

    Pipeline:
      1. Calcular pesos objetivo w_tgt (causaless)
      2. Aplicar frecuencia de rebalanceo semanal
      3. Calcular retornos netos: equity + cash - TC

    Retorna dict con todas las series temporales del portfolio.
    """
    n = len(returns)

    # Pesos objetivo con histéresis y cooldown (causaless)
    w_tgt = position_weights(
        prob_bull, cond_vol,
        vol_tgt    = vol_tgt,
        lev_cap    = Config.LEVERAGE_CAP,
        thresh_buy = Config.THRESH_BUY,
        thresh_sell= Config.THRESH_SELL,
        cooldown   = Config.COOLDOWN_DAYS,
        min_chg    = min_chg,
    )

    # Aplicar frecuencia de rebalanceo: solo ejecutar en días de rebalanceo
    rebal_set = set(pd.date_range(dates[0], dates[-1], freq=rebal_freq))
    w_exec    = np.zeros(n)
    prev_w    = 0.0
    for t in range(n):
        if dates[t] in rebal_set or t == 0:
            prev_w = w_tgt[t]
        w_exec[t] = prev_w

    # Retornos diarios netos
    delta_w  = np.abs(np.diff(np.concatenate([[0.0], w_exec])))
    tc_cost  = delta_w * tc
    cash_ret = (1.0 - w_exec) * (cash_ann / Config.ANN)
    strat    = w_exec * returns - tc_cost + cash_ret

    # NAV y drawdown
    nav  = np.cumprod(1.0 + strat)
    peak = np.maximum.accumulate(nav)
    dd   = (nav - peak) / (peak + 1e-10)

    # Registro de operaciones
    trades = []
    for t in range(1, n):
        dw = w_exec[t] - w_exec[t - 1]
        if abs(dw) > 0.005:
            trades.append({
                'date':        dates[t],
                'type':        'BUY' if dw > 0 else 'SELL',
                'w_prev':      round(w_exec[t - 1], 4),
                'w_new':       round(w_exec[t], 4),
                'delta_w':     round(dw, 4),
                'tc_bps':      round(abs(dw) * tc * 10_000, 2),
                'prob_bull':   round(prob_bull[t], 4),
                'ewma_vol':    round(cond_vol[t], 4),
            })

    return {
        'strat':    strat,
        'nav':      nav,
        'dd':       dd,
        'w_exec':   w_exec,
        'w_tgt':    w_tgt,
        'tc_cost':  tc_cost,
        'cash_ret': cash_ret,
        'trades':   pd.DataFrame(trades),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 6 — MÉTRICAS
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(res: dict, returns: np.ndarray, ann: int = Config.ANN) -> dict:
    """Métricas de performance completas del portfolio."""
    s  = res['strat']
    w  = res['w_exec']
    dd = res['dd']

    mu  = s.mean() * ann
    sig = s.std()  * ann ** 0.5
    sr  = mu / (sig + 1e-10)
    mdd = dd.min()
    cal = mu / (abs(mdd) + 1e-10)
    neg = s[s < 0].std() * ann ** 0.5 if (s < 0).any() else 1e-10
    sor = mu / (neg + 1e-10)
    ulc = float(np.sqrt(np.mean(dd ** 2)))

    # Hit rate: correctas en long + correctas en cash
    sig_lag = np.concatenate([[0.0], w[:-1]])
    r1      = returns[1:]
    long_ix = sig_lag[1:] > 0
    cash_ix = sig_lag[1:] == 0
    hr_l    = (r1[long_ix] > 0).mean() if long_ix.any() else 0.5
    hr_c    = (r1[cash_ix] < 0).mean() if cash_ix.any() else 0.5
    hit     = (hr_l + hr_c) / 2

    # IC (Information Coefficient)
    ic = float(np.corrcoef(sig_lag[:-1].astype(float),
                           returns[1:].astype(float))[0, 1])

    n_trades  = len(res['trades'])
    sw_yr     = n_trades / (len(s) / ann)
    tc_total  = res['tc_cost'].sum()
    cash_tot  = res['cash_ret'].sum()
    avg_w     = w.mean()
    nav_x     = res['nav'][-1]
    cagr      = nav_x ** (ann / len(s)) - 1

    return dict(
        sr=sr, mu=mu, sig=sig, mdd=mdd, cal=cal, sor=sor, ulcer=ulc,
        cagr=cagr, hit=hit, ic=ic,
        sw_yr=sw_yr, n_trades=n_trades,
        tc_total=tc_total, cash_total=cash_tot,
        avg_w=avg_w, nav_x=nav_x,
    )


def bootstrap_sharpe(strat: np.ndarray,
                     block: int = Config.BLOCK_SIZE,
                     n:     int = Config.N_BOOT) -> dict:
    """
    Bootstrap de bloque circular para inferencia sobre el Sharpe.

    El bloque circular preserva la autocorrelación de los retornos,
    que es crítica para mercados con clustering de volatilidad.
    """
    obs = strat.mean() * Config.ANN / (strat.std() * Config.ANN ** 0.5 + 1e-10)
    T   = len(strat)
    nb  = int(np.ceil(T / block))
    rng = np.random.default_rng(seed=42)
    sr_boot = np.zeros(n)
    for b in range(n):
        starts   = rng.integers(0, T, nb)
        boot     = np.concatenate([np.roll(strat, -s)[:block] for s in starts])[:T]
        sr_boot[b] = boot.mean() * Config.ANN / (boot.std() * Config.ANN ** 0.5 + 1e-10)
    return dict(
        obs    = obs,
        p5     = float(np.percentile(sr_boot, 5)),
        p25    = float(np.percentile(sr_boot, 25)),
        p75    = float(np.percentile(sr_boot, 75)),
        p95    = float(np.percentile(sr_boot, 95)),
        p_val  = float((sr_boot <= 0).mean()),
        dist   = sr_boot,
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 7 — VISUALIZACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def _setup_style(dark: bool = Config.DARK_THEME) -> dict:
    """Aplica estilo oscuro o claro y retorna el dict de colores."""
    if dark:
        plt.rcParams.update({
            'figure.facecolor': '#0d1117', 'axes.facecolor':  '#161b22',
            'axes.edgecolor':   '#30363d', 'axes.labelcolor': '#e6edf3',
            'text.color':       '#e6edf3', 'xtick.color':     '#8b949e',
            'ytick.color':      '#8b949e', 'grid.color':      '#21262d',
            'grid.alpha': 0.55, 'lines.linewidth': 1.0,
            'font.family': 'monospace', 'font.size': 8.5,
        })
        return dict(
            bg='#0d1117', ax_bg='#161b22',
            model='#58a6ff', model2='#3fb950', bench='#8b949e',
            bull='#3fb950', bear='#f85149', gold='#e3b341',
            tc='#f85149', cash='#e3b341', pos_fill='#1a2e1a',
        )
    else:
        plt.rcParams.update({'figure.facecolor': 'white', 'font.family': 'monospace'})
        return dict(
            bg='white', ax_bg='#f8f9fa',
            model='#1f77b4', model2='#2ca02c', bench='#7f7f7f',
            bull='#2ca02c', bear='#d62728', gold='#ff7f0e',
            tc='#d62728', cash='#ff7f0e', pos_fill='#e8f5e9',
        )


def _add_events(ax, dates, y_top=None):
    """Anota eventos macro en el eje con líneas verticales."""
    for d_str, label in Config.EVENTS.items():
        d = pd.Timestamp(d_str)
        if dates[0] <= d <= dates[-1]:
            ax.axvline(d, color='#8b949e', lw=0.7, ls=':', alpha=0.65)
            if y_top is not None:
                ax.text(d, y_top, label, rotation=90, fontsize=6.5,
                        color='#8b949e', va='top', ha='right')


def plot_backtest(df, res, m, bs, out_dir: str):
    """
    Dashboard de backtest — 6 paneles:
      A. NAV log (FCI vs B&H)
      B. Peso equity + señal BOCPD
      C. Drawdown comparado
      D. Distribución bootstrap Sharpe
      E. Tabla de métricas
      F. Retornos mensuales heatmap
    """
    c = _setup_style()
    fig = plt.figure(figsize=(22, 16))
    fig.patch.set_facecolor(c['bg'])
    gs  = mgridspec.GridSpec(3, 3, hspace=0.48, wspace=0.35, figure=fig)
    dates = df.index
    r     = df['LogRet'].values
    bh    = np.cumprod(1 + r)

    # ── A: NAV log ─────────────────────────────────────────────────────────
    ax = fig.add_subplot( gs[0, :2] )
    ax.plot( dates, bh, color=c['bench'], lw=0.9, ls='--',
             label='Buy & Hold', alpha=0.55 )
    ax.plot( dates, res['nav'], color=c['model'], lw=1.8,
             label=f'FCI Model  SR={m["sr"]:+.3f}' )
    ax.set_yscale( 'log' )
    ax.set_ylabel( 'NAV (escala log, base=1)', color=c['bench'] )
    ax.set_title( f'NAV Acumulado — Modelo vs Buy & Hold',
                  color=c['bench'] )
    ax.legend( fontsize=8, loc='upper left' )
    ax.grid( True, alpha=0.4 )

    # ── C: Drawdown ─────────────────────────────────────────────────────────
    ax = fig.add_subplot( gs[1, :2], sharex=fig.axes[0] )
    bh_dd = (bh - np.maximum.accumulate( bh )) / np.maximum.accumulate( bh )
    ax.fill_between( dates, bh_dd.astype( float ), 0,
                     color=c['bench'], alpha=0.20,
                     label=f'B&H  MDD={bh_dd.min():.0%}' )
    ax.fill_between( dates, res['dd'].astype( float ), 0,
                     color=c['model'], alpha=0.30,
                     label=f'FCI  MDD={m["mdd"]:.0%}' )
    ax.yaxis.set_major_formatter( plt.FuncFormatter( lambda x, _: f'{x:.0%}' ) )
    ax.set_ylabel( 'Drawdown', color=c['bench'] )
    ax.set_title( 'Drawdown Histórico', color=c['bench'] )
    ax.legend( fontsize=8 )
    ax.grid( True, alpha=0.4 )
    _add_events( ax, dates )

    # ── D: Bootstrap ────────────────────────────────────────────────────────
    ax = fig.add_subplot( gs[0, 2] )
    ax.hist( bs['dist'], bins=60, color=c['model'], alpha=0.65, density=True )
    ax.axvline( bs['obs'], color=c['gold'], lw=2.0,
                label=f'SR obs = {bs["obs"]:.3f}' )
    ax.axvspan( bs['p5'], bs['p95'], color=c['model'], alpha=0.12,
                label=f'CI 90%=[{bs["p5"]:.2f},{bs["p95"]:.2f}]' )
    ax.set_xlabel( 'Sharpe Ratio', color=c['bench'] )
    ax.set_title( f'Bootstrap Sharpe  (N={Config.N_BOOT}, bloque={Config.BLOCK_SIZE}d)\n'
                  f'p-val={bs["p_val"]:.3f}', color=c['bench'], fontsize=8 )
    ax.legend( fontsize=7 )
    ax.grid( True, alpha=0.4 )

    # ── E: Tabla de métricas ────────────────────────────────────────────────
    ax = fig.add_subplot( gs[1, 2] )
    ax.axis( 'off' )
    bh_sr = r.mean() * Config.ANN / (r.std() * Config.ANN ** 0.5)
    bh_mdd = bh_dd.min()
    data = [
        ['Métrica', 'Modelo', 'Buy & Hold'],
        ['Sharpe', f'{m["sr"]:+.3f}', f'{bh_sr:+.3f}'],
        ['Ret. Ann.', f'{m["mu"]:.1%}', f'{r.mean() * Config.ANN:.1%}'],
        ['Vol. Ann.', f'{m["sig"]:.1%}', f'{r.std() * Config.ANN ** .5:.1%}'],
        ['Max DD', f'{m["mdd"]:.1%}', f'{bh_mdd:.1%}'],
        ['Calmar', f'{m["cal"]:.3f}', f'{r.mean() * Config.ANN / abs( bh_mdd ):.3f}'],
        ['Sortino', f'{m["sor"]:.3f}', '—'],
        ['NAV final', f'×{m["nav_x"]:.1f}', f'×{bh[-1]:.1f}'],
        ['CI 90% SR', f'[{bs["p5"]:+.2f},{bs["p95"]:+.2f}]', '—'],
    ]
    tbl = ax.table( cellText=data[1:], colLabels=data[0],
                    cellLoc='center', loc='center', bbox=[0, 0, 1, 1] )
    tbl.auto_set_font_size( False )
    tbl.set_fontsize( 8 )
    for (ri, ci), cell in tbl.get_celld().items():
        bg = '#161b22' if ri % 2 == 0 else '#1a2332'
        cell.set_facecolor( bg )
        cell.set_edgecolor( '#30363d' )
        cell.set_text_props( color='#e6edf3' )
        if ri == 0:
            cell.set_facecolor( '#21262d' )
            cell.set_text_props( color='#58a6ff', fontweight='bold' )
        if ci == 1 and ri > 0:
            cell.set_facecolor( c['pos_fill'] )
    ax.set_title( 'Métricas de Performance', color=c['bench'], fontsize=9, pad=8 )

    # ── F: Retorno año a año ───────────────────────────────────────────────
    ax = fig.add_subplot( gs[2, :] )
    ret_s = pd.Series( res['strat'], index=dates ).fillna( 0.0 )
    ret_b = pd.Series( r, index=dates ).fillna( 0.0 )
    yr_s = ret_s.groupby( ret_s.index.year ).apply( lambda x: (1.0 + x).prod() - 1.0 )
    yr_b = ret_b.groupby( ret_b.index.year ).apply( lambda x: (1.0 + x).prod() - 1.0 )
    years = yr_s.index.values
    x = np.arange( len( years ) )
    w_bar = 0.38

    b1 = ax.bar( x - w_bar / 2, yr_s.values, w_bar, color=c['bull'],
                 edgecolor='#30363d', linewidth=0.5, label='FCI Model' )
    b2 = ax.bar( x + w_bar / 2, yr_b.values, w_bar, color=c['bench'],
                 edgecolor='#30363d', linewidth=0.5, label='Buy & Hold' )

    for rect, val in list( zip( b1, yr_s.values ) ) + list( zip( b2, yr_b.values ) ):
        ax.text( rect.get_x() + rect.get_width() / 2, val, f'{val:.0%}',
                 ha='center', va='bottom' if val >= 0 else 'top',
                 fontsize=6.5, color='#e6edf3' )

    ax.axhline( 0, color='#8b949e', lw=0.8 )
    ax.set_xticks( x )
    ax.set_xticklabels( [str( int( y ) ) for y in years] )
    ax.yaxis.set_major_formatter( plt.FuncFormatter( lambda v, _: f'{v:.0%}' ) )
    ax.set_ylabel( 'Retorno anual', color=c['bench'] )
    ax.set_title( 'Retorno Año a Año — Modelo (verde) vs Buy & Hold (gris)',
                  color=c['bench'] )
    ax.legend( fontsize=8, loc='upper left' )
    ax.grid( True, axis='y', alpha=0.4 )

    fig.suptitle(
        'MERVAL/BADLAR — MODELO BOCPD × VS20%\n'
        f'Período: {dates[0].date()} → {dates[-1].date()}  |',
        color=c['bench'], fontsize=14, y=1.01 )

    path = os.path.join( out_dir, 'backtest.png' )
    plt.savefig( path, dpi=140, bbox_inches='tight', facecolor=c['bg'] )
    plt.close( fig )
    print( f'  [✓] {path}' )


def plot_attribution(df, res, out_dir: str):
    """
    Atribución de retorno y análisis de riesgo:
      A. Retorno acumulado descompuesto (equity / TC / cash)
      B. Volatilidad EWMA vs target
      C. Rolling Sharpe anual
      D. VaR y CVaR histórico vs B&H
    """
    c     = _setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(20, 10))
    fig.patch.set_facecolor(c['bg'])
    dates = df.index
    r     = df['LogRet'].values

    # A: Atribución acumulada
    ax = axes[0, 0]
    cum_eq   = np.cumsum(res['w_exec'] * r)
    cum_tc   = -np.cumsum(res['tc_cost'])
    cum_cash = np.cumsum(res['cash_ret'])
    cum_net  = cum_eq + cum_tc + cum_cash
    ax.fill_between(dates, 0, cum_eq,
                    color=c['bull'],  alpha=0.38, label=f'Equity  {cum_eq[-1]:.1%}')
    ax.fill_between(dates, cum_eq, cum_eq + cum_tc,
                    color=c['tc'],    alpha=0.55, label=f'TC pagado  {-cum_tc[-1]:.1%}')
    ax.fill_between(dates, cum_eq + cum_tc, cum_eq + cum_tc + cum_cash,
                    color=c['cash'],  alpha=0.45, label=f'Cash  {cum_cash[-1]:.1%}')
    ax.plot(dates, cum_net, color=c['model'], lw=1.5, label=f'Neto  {cum_net[-1]:.1%}')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    ax.set_title('Atribución de Retorno Acumulado', color=c['bench'])
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.4)

    # B: Vol EWMA vs target
    ax  = axes[0, 1]
    ax2 = ax.twinx()
    ev_ann = ewma_vol(r)
    ax.fill_between(dates, ev_ann, color=c['bear'], alpha=0.30, label='Vol EWMA (ann.)')
    ax.axhline(Config.VOL_TARGET, color=c['gold'], lw=1.3, ls='--',
               label=f'σ_target={Config.VOL_TARGET:.0%}')
    vs = np.minimum(Config.VOL_TARGET / np.maximum(ev_ann, Config.VOL_FLOOR), 1.0)
    ax2.plot(dates, vs, color=c['model'], lw=0.8, alpha=0.65)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    ax2.set_ylim(0, 1.2)
    ax2.set_ylabel('Multiplicador VS', color='#8b949e')
    ax2.tick_params(axis='y', colors='#8b949e')
    ax.set_ylabel('Vol EWMA', color=c['bench'])
    ax.set_title('Volatility Scaling — σ_EWMA vs σ_target', color=c['bench'])
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.4)

    # C: Rolling Sharpe
    ax  = axes[1, 0]
    W   = Config.ANN
    s_s = pd.Series(res['strat'], index=dates)
    bh_s = pd.Series(r, index=dates)
    roll_sr = s_s.rolling(W).mean() * W / (s_s.rolling(W).std() * W ** 0.5 + 1e-10)
    roll_bh = bh_s.rolling(W).mean() * W / (bh_s.rolling(W).std() * W ** 0.5 + 1e-10)
    ax.axhline(0, color='#8b949e', lw=0.7, ls='--', alpha=0.5)
    ax.fill_between(dates, roll_sr, 0,
                    where=(roll_sr.fillna(0) > 0), color=c['bull'], alpha=0.22)
    ax.fill_between(dates, roll_sr, 0,
                    where=(roll_sr.fillna(0) < 0), color=c['bear'], alpha=0.22)
    ax.plot(dates, roll_sr, color=c['model'], lw=1.2, label='FCI Model')
    ax.plot(dates, roll_bh, color=c['bench'], lw=0.8, ls='--', alpha=0.5,
            label='Buy & Hold')
    ax.set_ylabel('Sharpe Rolling 252d', color=c['bench'])
    ax.set_title('Rolling Sharpe Anual', color=c['bench'])
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.4)
    _add_events(ax, dates)

    # D: VaR / CVaR
    ax  = axes[1, 1]
    s_act = res['strat'][res['strat'] != 0]
    confs = [0.90, 0.95, 0.99]
    x     = np.arange(len(confs))
    w_bar = 0.25
    bh_var  = [-np.percentile(r, (1 - q) * 100) for q in confs]
    bh_cvar = [-r[r < np.percentile(r, (1-q)*100)].mean() for q in confs]
    s_var   = [-np.percentile(s_act, (1-q)*100) for q in confs]
    s_cvar  = [-s_act[s_act < np.percentile(s_act, (1-q)*100)].mean() for q in confs]
    ax.bar(x - w_bar,     bh_var,  w_bar, color=c['bench'], alpha=0.65, label='B&H VaR')
    ax.bar(x,             bh_cvar, w_bar, color=c['bench'], alpha=0.35, label='B&H CVaR')
    ax.bar(x + w_bar,     s_var,   w_bar, color=c['model'], alpha=0.65, label='FCI VaR')
    ax.bar(x + 2*w_bar,   s_cvar,  w_bar, color=c['model'], alpha=0.35, label='FCI CVaR')
    ax.set_xticks(x + w_bar / 2)
    ax.set_xticklabels([f'{q:.0%}' for q in confs])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
    ax.set_title('VaR y CVaR Histórico', color=c['bench'])
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.4)

    fig.suptitle('ANÁLISIS DE ATRIBUCIÓN Y RIESGO — FCI Model',
                 color=c['bench'], fontsize=11, y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, 'risk_attribution.png')
    plt.savefig(path, dpi=140, bbox_inches='tight', facecolor=c['bg'])
    plt.close(fig)
    print(f'  [✓] {path}')


def plot_trades(res, out_dir: str):
    """Log de operaciones visual: histograma de duración, TC por trade, scatter."""
    if len(res['trades']) == 0:
        print('  [!] Sin operaciones para visualizar.')
        return
    c     = _setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor(c['bg'])
    trades = res['trades']

    # A: Distribución de TC por operación
    ax = axes[0]
    ax.hist(trades['tc_bps'], bins=20, color=c['model'], alpha=0.75)
    ax.axvline(trades['tc_bps'].mean(), color=c['gold'], lw=1.5, ls='--',
               label=f'Media={trades["tc_bps"].mean():.1f}bps')
    ax.set_xlabel('TC (bps)', color=c['bench'])
    ax.set_title('Distribución TC por Operación', color=c['bench'])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4)

    # B: Delta peso por operación
    ax = axes[1]
    colors_t = [c['bull'] if d > 0 else c['bear'] for d in trades['delta_w']]
    ax.bar(range(len(trades)), trades['delta_w'] * 100, color=colors_t, alpha=0.80)
    ax.axhline(0, color='#8b949e', lw=0.6)
    ax.set_xlabel('Nº operación', color=c['bench'])
    ax.set_ylabel('Δ Peso (%)', color=c['bench'])
    ax.set_title('Delta Peso por Operación', color=c['bench'])
    ax.grid(True, alpha=0.4)

    # C: Scatter prob_bull vs Δpeso
    ax = axes[2]
    sc = ax.scatter(trades['prob_bull'], trades['delta_w'] * 100,
                    c=trades['tc_bps'], cmap='RdYlGn', s=60, alpha=0.80)
    plt.colorbar(sc, ax=ax, label='TC (bps)')
    ax.axhline(0, color='#8b949e', lw=0.6, ls='--')
    ax.axvline(Config.BULL_THRESH, color='#8b949e', lw=0.6, ls='--')
    ax.set_xlabel('P(Bull) en momento del trade', color=c['bench'])
    ax.set_ylabel('Δ Peso (%)', color=c['bench'])
    ax.set_title('P(Bull) vs Cambio de Posición', color=c['bench'])
    ax.grid(True, alpha=0.4)

    fig.suptitle(f'OPERACIONES — {len(trades)} trades  |  '
                 f'TC medio={trades["tc_bps"].mean():.1f}bps  |  '
                 f'TC total={trades["tc_bps"].sum()/10000:.2%}',
                 color=c['bench'], fontsize=10, y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, 'trades.png')
    plt.savefig(path, dpi=140, bbox_inches='tight', facecolor=c['bg'])
    plt.close(fig)
    print(f'  [✓] {path}')


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 8 — EXPORTACIÓN CSV
# ══════════════════════════════════════════════════════════════════════════════

def export_signals(df, res, prob_bull, cond_vol, out_dir: str):
    """
    Exporta señales diarias completas a CSV.

    Columnas:
      date, price, log_ret, prob_bull, ewma_vol, vs_mult,
      signal_bull, w_target, w_applied, strat_ret, nav, drawdown,
      tc_cost_daily, cash_ret_daily
    """
    vs_mult = np.minimum(Config.VOL_TARGET / np.maximum(cond_vol, Config.VOL_FLOOR), 1.0)
    out = pd.DataFrame({
        'price':          df['Price'],
        'log_ret':        df['LogRet'],
        'prob_bull':      np.round(prob_bull, 6),
        'ewma_vol_ann':   np.round(cond_vol, 6),
        'vs_multiplier':  np.round(vs_mult, 6),
        'signal_bull':    (prob_bull > Config.BULL_THRESH).astype(int),
        'w_target':       np.round(res['w_tgt'], 6),
        'w_applied':      np.round(res['w_exec'], 6),
        'strat_ret':      np.round(res['strat'], 8),
        'nav':            np.round(res['nav'], 6),
        'drawdown':       np.round(res['dd'], 6),
        'tc_cost_bps':    np.round(res['tc_cost'] * 10_000, 4),
        'cash_ret_daily': np.round(res['cash_ret'], 8),
    }, index=df.index)
    path = os.path.join(out_dir, 'fci_signals.xlsx')
    out.to_excel(path, index_label='date')
    print(f'  [✓] {path}  ({len(out)} rows)')
    return out


def export_trades(res, out_dir: str):
    """Exporta el log de operaciones a CSV."""
    if len(res['trades']) == 0:
        print('  [!] Sin operaciones.')
        return
    path = os.path.join(out_dir, 'fci_trades.csv')
    res['trades'].to_csv(path, index=False)
    print(f'  [✓] {path}  ({len(res["trades"])} operaciones)')


def export_metrics(m, bs, out_dir: str):
    """Exporta la ficha de métricas a CSV."""
    sig = '***' if bs['p_val'] < 0.01 else ('**' if bs['p_val'] < 0.05 else 'n.s.')
    rows = [
        ('MODELO',       'Señal',              'BOCPD'),
        ('MODELO',       'Lambda',             f'{Config.LAM_BOCPD}d'),
        ('MODELO',       'Vol target',         f'{Config.VOL_TARGET:.0%}'),
        ('MODELO',       'EWMA span',          f'{Config.EWMA_SPAN}d'),
        ('MODELO',       'Bull threshold',     f'{Config.BULL_THRESH:.0%}'),
        ('EJECUCION',    'Rebalanceo',         Config.REBAL_FREQ),
        ('EJECUCION',    'Min cambio',         f'{Config.MIN_CHANGE:.0%}'),
        ('EJECUCION',    'TC (bps)',            f'{Config.TC_BPS}'),
        ('EJECUCION',    'Cash return',        f'{Config.CASH_ANN:.0%}'),
        ('EJECUCION',    'Burn-in',            f'{Config.MIN_TRAIN}d'),
        ('PERFORMANCE',  'Sharpe Ratio',       f'{m["sr"]:+.4f}  {sig}'),
        ('PERFORMANCE',  'CI 90%',             f'[{bs["p5"]:+.3f},{bs["p95"]:+.3f}]'),
        ('PERFORMANCE',  'p-val SR>0',         f'{bs["p_val"]:.4f}'),
        ('PERFORMANCE',  'Retorno anualizado', f'{m["mu"]:.4%}'),
        ('PERFORMANCE',  'Volatilidad',        f'{m["sig"]:.4%}'),
        ('PERFORMANCE',  'Max Drawdown',       f'{m["mdd"]:.4%}'),
        ('PERFORMANCE',  'Calmar',             f'{m["cal"]:.4f}'),
        ('PERFORMANCE',  'Sortino',            f'{m["sor"]:.4f}'),
        ('PERFORMANCE',  'Ulcer Index',        f'{m["ulcer"]:.6f}'),
        ('PERFORMANCE',  'Hit Rate',           f'{m["hit"]:.4f}'),
        ('PERFORMANCE',  'IC',                 f'{m["ic"]:.4f}'),
        ('OPERACIONES',  'Switches / año',     f'{m["sw_yr"]:.2f}'),
        ('OPERACIONES',  'N trades total',     f'{m["n_trades"]}'),
        ('OPERACIONES',  '% tiempo equity',    f'{m["avg_w"]:.4%}'),
        ('OPERACIONES',  'TC total',           f'{m["tc_total"]:.4%}'),
        ('OPERACIONES',  'Cash total',         f'{m["cash_total"]:.4%}'),
        ('OPERACIONES',  'NAV final',          f'{m["nav_x"]:.4f}'),
        ('OPERACIONES',  'CAGR',               f'{m["cagr"]:.4%}'),
    ]
    df_out = pd.DataFrame(rows, columns=['Categoria', 'Parametro', 'Valor'])
    path   = os.path.join(out_dir, 'fci_metrics.csv')
    df_out.to_csv(path, index=False, encoding='utf-8-sig')
    print(f'  [✓] {path}')


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 9 — SEÑAL EN VIVO (uso diario en producción)
# ══════════════════════════════════════════════════════════════════════════════

class LiveSignal:
    """
    Interfaz para uso diario en producción.

    USO TÍPICO:
        from merval_fci_production import LiveSignal, Config

        ls = LiveSignal()
        ls.load_history('base_kalman.xlsx')
        signal = ls.get_signal()

        print(f"Fecha       : {signal['date']}")
        print(f"P(Bull)     : {signal['prob_bull']:.3f}")
        print(f"Vol EWMA    : {signal['ewma_vol']:.1%}")
        print(f"Peso objetivo: {signal['w_target']:.1%}")
        print(f"Régimen     : {signal['regime']}")
        print(f"Acción      : {signal['action']}")

    El método get_signal() retorna la señal para HOY basada en
    los datos hasta AYER — garantía de causalidad en producción.
    """

    def __init__(self, config: Config = None):
        self.cfg       = config or Config()
        self.prob_bull = None
        self.cond_vol  = None
        self.df        = None
        self._fitted   = False

    def load_history(self, path: str = None, sheet: str = None):
        """Carga datos históricos y computa la señal hasta el último dato."""
        path  = path  or self.cfg.DATA_PATH
        sheet = sheet or self.cfg.SHEET_NAME
        self.df = load_data(path, sheet)
        self._compute()
        return self

    def _compute(self):
        r              = self.df['LogRet'].values
        self.cond_vol  = ewma_vol(r)
        self.prob_bull = bocpd_filter(r, lam=self.cfg.LAM_BOCPD)
        self.prob_bull[:self.cfg.MIN_TRAIN] = 0.5   # burn-in
        self._fitted   = True

    def get_signal(self, prev_weight: float = None) -> dict:
        """
        Retorna la señal para el próximo día hábil.

        Params:
          prev_weight: peso actual del portfolio (para calcular acción)

        Retorna dict con:
          date          : fecha del último dato disponible
          prob_bull     : P(bull) según BOCPD
          ewma_vol      : volatilidad condicional EWMA (ann.)
          vs_multiplier : factor de Volatility Scaling
          w_target      : peso objetivo (señal × VS)
          regime        : 'BULL' o 'BEAR'
          action        : 'HOLD' / 'BUY_XX%' / 'SELL_XX%'
          execute_rebal : True si viernes (día de rebalanceo)
        """
        if not self._fitted:
            raise RuntimeError("Llamar load_history() primero.")

        last_t    = -1
        pb_t      = float(self.prob_bull[last_t])
        ev_t      = float(self.cond_vol[last_t])
        ev_t_safe = max(ev_t, self.cfg.VOL_FLOOR)
        vs_mult   = min(self.cfg.VOL_TARGET / ev_t_safe, self.cfg.LEVERAGE_CAP)

        # Lógica de régimen con histéresis
        if pb_t > self.cfg.THRESH_BUY:
            regime = 'BULL (convicción alta)'
            w_tgt  = vs_mult
        elif pb_t < self.cfg.THRESH_SELL:
            regime = 'BEAR (convicción alta)'
            w_tgt  = 0.0
        else:
            regime = f'ZONA GRIS ({self.cfg.THRESH_SELL:.0%}<pb<{self.cfg.THRESH_BUY:.0%}) — mantener posición previa'
            w_tgt  = None   # depende de la posición actual del portfolio
        last_date = self.df.index[-1]
        days_to_fri = (4 - last_date.weekday()) % 7
        next_rebal  = last_date + pd.Timedelta(days=days_to_fri if days_to_fri > 0 else 7)

        action = 'HOLD'
        if prev_weight is not None and w_tgt is not None:
            dw = w_tgt - prev_weight
            if abs(dw) >= self.cfg.MIN_CHANGE:
                action = f"{'BUY' if dw > 0 else 'SELL'} {abs(dw):.0%}"
            next_rebal_is_today = last_date.weekday() == 4
            execute = next_rebal_is_today and abs(dw) >= self.cfg.MIN_CHANGE
        else:
            execute = last_date.weekday() == 4

        return {
            'date':             last_date.date(),
            'prob_bull':        round(pb_t, 4),
            'ewma_vol':         round(ev_t, 4),
            'vs_multiplier':    round(vs_mult, 4),
            'w_target':         round(w_tgt, 4) if w_tgt is not None else (round(prev_weight, 4) if prev_weight is not None else 'mantener_prev'),
            'regime':           regime,
            'thresh_buy':       self.cfg.THRESH_BUY,
            'thresh_sell':      self.cfg.THRESH_SELL,
            'cooldown_days':    self.cfg.COOLDOWN_DAYS,
            'action':           action,
            'execute_rebal':    execute,
            'next_rebal_date':  next_rebal.date(),
        }

    def update(self, new_price: float, new_date: pd.Timestamp = None):
        """
        Actualiza el modelo con un nuevo dato de precio.
        Útil para actualización diaria sin recargar todo el histórico.
        """
        if not self._fitted:
            raise RuntimeError("Llamar load_history() primero.")

        date      = new_date or pd.Timestamp.today().normalize()
        log_ret   = np.log(new_price / self.df['Price'].iloc[-1])
        new_row   = pd.DataFrame({'Price': [new_price], 'LogRet': [log_ret]},
                                  index=[date])
        self.df   = pd.concat([self.df, new_row])
        self._compute()
        return self.get_signal()


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 10 — BACKTEST COMPLETO (modo script)
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(data_path: str = Config.DATA_PATH,
                 out_dir:   str = Config.OUTPUT_DIR,
                 verbose:   bool = True) -> dict:
    """
    Ejecuta el backtest completo y genera todos los outputs.

    Retorna dict con: df, prob_bull, cond_vol, res, metrics, bs
    """
    os.makedirs(out_dir, exist_ok=True)

    if verbose:
        _banner()

    # ── 1. Datos ───────────────────────────────────────────────────────────
    if verbose:
        print("PASO 1 — CARGA DE DATOS")
        print("─" * 60)
    df = load_data(data_path)
    r  = df['LogRet'].values
    dates = df.index
    if verbose:
        print(f"  Serie  : {dates[0].date()} → {dates[-1].date()}")
        print(f"  N obs  : {len(r)}")
        print(f"  μ_ann  : {r.mean()*Config.ANN:.2%}")
        print(f"  σ_ann  : {r.std()*Config.ANN**0.5:.2%}")
        print(f"  Skew   : {stats.skew(r):.3f}")
        print(f"  Kurt   : {stats.kurtosis(r):.1f}")

    # ── 2. Señal BOCPD ─────────────────────────────────────────────────────
    if verbose:
        print()
        print("PASO 2 — SEÑAL BOCPD")
        print("─" * 60)
        print(f"  λ={Config.LAM_BOCPD}d  prior NIG  poda={Config.BOCPD_PRUNE}", end=' ', flush=True)
        t0 = __import__('time').time()
    pb = bocpd_filter(r)
    pb[:Config.MIN_TRAIN] = 0.5   # burn-in
    if verbose:
        print(f"→ {__import__('time').time()-t0:.1f}s  "
              f"P(bull)_mean={pb[Config.MIN_TRAIN:].mean():.3f}")

    # ── 3. Volatilidad ─────────────────────────────────────────────────────
    if verbose:
        print()
        print("PASO 3 — VOLATILIDAD EWMA")
        print("─" * 60)
    ev = ewma_vol(r)
    if verbose:
        print(f"  EWMA span={Config.EWMA_SPAN}d")
        print(f"  mean={ev[Config.MIN_TRAIN:].mean():.1%}  "
              f"p10={np.percentile(ev[Config.MIN_TRAIN:],10):.1%}  "
              f"p90={np.percentile(ev[Config.MIN_TRAIN:],90):.1%}")
        print(f"  Pct tiempo VS<1 (vol>target): "
              f"{(ev[Config.MIN_TRAIN:] > Config.VOL_TARGET).mean():.1%}")

    # ── 4. Simulación ──────────────────────────────────────────────────────
    if verbose:
        print()
        print("PASO 4 — SIMULACIÓN FCI")
        print("─" * 60)
    res = simulate(r, pb, ev, dates)
    m   = compute_metrics(res, r)
    if verbose:
        print(f"  Sharpe      : {m['sr']:+.3f}")
        print(f"  Max DD      : {m['mdd']:.1%}")
        print(f"  Retorno ann.: {m['mu']:.1%}")
        print(f"  Volatilidad : {m['sig']:.1%}")
        print(f"  Calmar      : {m['cal']:.3f}")
        print(f"  Sortino     : {m['sor']:.3f}")
        print(f"  Switches/yr : {m['sw_yr']:.1f}  ({m['n_trades']} operaciones)")
        print(f"  % Equity    : {m['avg_w']:.1%}")
        print(f"  TC total    : {m['tc_total']:.1%}")
        print(f"  Cash total  : {m['cash_total']:.1%}")
        print(f"  NAV final   : ×{m['nav_x']:.2f}")

    # ── 5. Bootstrap ───────────────────────────────────────────────────────
    if verbose:
        print()
        print("PASO 5 — INFERENCIA BOOTSTRAP")
        print("─" * 60)
    bs  = bootstrap_sharpe(res['strat'])
    sig = '***' if bs['p_val']<0.01 else ('**' if bs['p_val']<0.05 else 'n.s.')
    if verbose:
        print(f"  SR observado : {bs['obs']:+.3f}")
        print(f"  CI 50%       : [{bs['p25']:+.3f}, {bs['p75']:+.3f}]")
        print(f"  CI 90%       : [{bs['p5']:+.3f}, {bs['p95']:+.3f}]")
        print(f"  p-val (SR>0) : {bs['p_val']:.4f}  {sig}")

    # ── 6. Señal actual ────────────────────────────────────────────────────
    if verbose:
        print()
        print("PASO 6 — SEÑAL ACTUAL (último dato disponible)")
        print("─" * 60)
        ls   = LiveSignal()
        ls.df = df; ls.prob_bull = pb; ls.cond_vol = ev; ls._fitted = True
        sig_today = ls.get_signal()
        print(f"  Fecha             : {sig_today['date']}")
        print(f"  P(Bull) BOCPD     : {sig_today['prob_bull']:.4f}")
        print(f"  Vol EWMA          : {sig_today['ewma_vol']:.2%}")
        print(f"  VS multiplicador  : {sig_today['vs_multiplier']:.4f}")
        print(f"  Peso objetivo     : {sig_today['w_target']}")
        print(f"  Régimen           : {sig_today['regime']}")
        print(f"  Umbral BUY        : {sig_today['thresh_buy']:.0%}  |  Umbral SELL: {sig_today['thresh_sell']:.0%}")
        print(f"  Cooldown          : {sig_today['cooldown_days']}d post-sell")
        print(f"  Próximo rebal.    : {sig_today['next_rebal_date']}")

    # ── 7. Figuras ─────────────────────────────────────────────────────────
    if verbose:
        print()
        print("PASO 7 — VISUALIZACIÓN")
        print("─" * 60)
    plot_backtest(df, res, m, bs, out_dir)
    plot_attribution(df, res, out_dir)
    plot_trades(res, out_dir)

    # ── 8. CSVs ────────────────────────────────────────────────────────────
    if verbose:
        print()
        print("PASO 8 — EXPORTACIÓN")
        print("─" * 60)
    export_signals(df, res, pb, ev, out_dir)
    export_trades(res, out_dir)
    export_metrics(m, bs, out_dir)

    # ── Resumen ────────────────────────────────────────────────────────────
    if verbose:
        _footer(m, bs, sig)

    return dict(df=df, prob_bull=pb, cond_vol=ev, res=res, metrics=m, bs=bs)

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 11 — UTILIDADES DE CONSOLA
# ══════════════════════════════════════════════════════════════════════════════

def compare_versions(data_path: str = Config.DATA_PATH,
                     out_dir:   str = Config.OUTPUT_DIR) -> None:
    os.makedirs(out_dir, exist_ok=True)
    df = load_data(data_path)
    r  = df['LogRet'].values
    dates = df.index

    print()
    print("╔" + "═"*68 + "╗")
    print("║" + " COMPARACIÓN v1 vs v2 ".center(68) + "║")
    print("╚" + "═"*68 + "╝")
    print()

    # Señal común
    t0 = __import__('time').time()
    print("  Computando BOCPD...", end=' ', flush=True)
    pb = bocpd_filter(r); pb[:Config.MIN_TRAIN] = 0.5
    ev = ewma_vol(r)
    print(f"OK ({__import__('time').time()-t0:.1f}s)")

    # v1: simétrico 55/55 sin cooldown
    from copy import deepcopy
    cfg_v1 = deepcopy(Config)
    cfg_v1.THRESH_BUY   = 0.55
    cfg_v1.THRESH_SELL  = 0.55
    cfg_v1.COOLDOWN_DAYS = 0

    # Simular v1 manualmente
    w_v1 = np.zeros(len(r)); prev = 0.0
    for t in range(1, len(r)):
        ev_t = max(ev[t-1], Config.VOL_FLOOR)
        vs   = min(Config.VOL_TARGET / ev_t, Config.LEVERAGE_CAP)
        w_raw = vs if pb[t-1] > 0.55 else 0.0
        if abs(w_raw - prev) < Config.MIN_CHANGE: w_raw = prev
        w_v1[t] = w_raw; prev = w_raw
    rebal_set = set(pd.date_range(dates[0], dates[-1], freq=Config.REBAL_FREQ))
    w_v1_exec = np.zeros(len(r)); pw = 0.0
    for t in range(len(r)):
        if dates[t] in rebal_set or t == 0: pw = w_v1[t]
        w_v1_exec[t] = pw
    tc1 = np.abs(np.diff(np.concatenate([[0.], w_v1_exec]))) * Config.TC
    s_v1 = w_v1_exec * r - tc1 + (1 - w_v1_exec) * Config.CASH_ANN / Config.ANN

    # Simular v2 con la función estándar
    res_v2 = simulate(r, pb, ev, dates)
    s_v2   = res_v2['strat']

    def quick_metrics(s):
        mu  = s.mean() * Config.ANN; sg = s.std() * Config.ANN**0.5
        sr  = mu / (sg + 1e-10)
        cum = np.cumprod(1+s)
        mdd = ((cum-np.maximum.accumulate(cum))/np.maximum.accumulate(cum)).min()
        sw  = np.sum(np.abs(np.diff(s != 0))) / (len(s)/Config.ANN)
        tc_ = np.abs(np.diff(np.concatenate([[0.], (s>0).astype(float)]))).sum()
        return dict(sr=sr, mu=mu, mdd=mdd, sw=sw, nav=cum[-1])

    m1 = quick_metrics(s_v1); m2 = quick_metrics(s_v2)

    # Bootstrap ambas versiones
    def boot(s, N=Config.N_BOOT, bl=Config.BLOCK_SIZE):
        T=len(s); nb=int(np.ceil(T/bl)); rng=np.random.default_rng(42)
        sb=np.zeros(N)
        for b in range(N):
            st=rng.integers(0,T,nb)
            bt=np.concatenate([np.roll(s,-ss)[:bl] for ss in st])[:T]
            sb[b]=bt.mean()*Config.ANN/(bt.std()*Config.ANN**0.5+1e-10)
        obs=s.mean()*Config.ANN/(s.std()*Config.ANN**0.5+1e-10)
        return dict(obs=obs, p5=np.percentile(sb,5), p95=np.percentile(sb,95),
                    pval=(sb<=0).mean())

    b1 = boot(s_v1); b2 = boot(s_v2)

    print()
    W = 72
    print("  " + "─"*W)
    print(f"  {'Métrica':<28}  {'v1 (55/55, sin CD)':>20}  {'v2 (70/35, CD14d) ★':>20}")
    print("  " + "─"*W)
    rows = [
        ("Sharpe Ratio",
         f"{b1['obs']:+.3f}",
         f"{b2['obs']:+.3f}"),
        ("CI 90% Sharpe",
         f"[{b1['p5']:+.2f},{b1['p95']:+.2f}]",
         f"[{b2['p5']:+.2f},{b2['p95']:+.2f}]"),
        ("p-val SR>0",
         f"{b1['pval']:.4f} {'**' if b1['pval']<0.05 else 'n.s.'}",
         f"{b2['pval']:.4f} {'***' if b2['pval']<0.001 else '**' if b2['pval']<0.01 else '*'}"),
        ("Max Drawdown",      f"{m1['mdd']:.1%}",  f"{m2['mdd']:.1%}"),
        ("Retorno anualizado", f"{m1['mu']:.1%}",   f"{m2['mu']:.1%}"),
        ("NAV final",         f"×{m1['nav']:.2f}", f"×{m2['nav']:.2f}"),
        ("Switches / año",    f"{m1['sw']:.1f}",   f"{res_v2['trades'].__len__()/(len(r)/Config.ANN):.1f}"),
    ]
    for name, v1_val, v2_val in rows:
        print(f"  {name:<28}  {v1_val:>20}  {v2_val:>20}")
    print("  " + "─"*W)
    print()
    bh_sr = r.mean()*Config.ANN/(r.std()*Config.ANN**0.5)
    print(f"  Referencia Buy & Hold: SR={bh_sr:+.3f}  NAV=×{np.cumprod(1+r)[-1]:.2f}")
    print()

    # Figura comparación
    c = _setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(20, 10))
    fig.patch.set_facecolor(c['bg'])
    nav_bh = np.cumprod(1+r)
    nav_v1 = np.cumprod(1+s_v1)
    nav_v2 = np.cumprod(1+s_v2)

    # A: NAV
    ax = axes[0,0]
    ax.plot(dates, nav_bh, color=c['bench'], lw=0.8, ls='--', alpha=0.5, label='Buy & Hold')
    ax.plot(dates, nav_v1, color=c['gold'],  lw=1.2, label=f'v1  SR={b1["obs"]:+.3f}')
    ax.plot(dates, nav_v2, color=c['model2'],lw=1.8, label=f'v2  SR={b2["obs"]:+.3f} ★')
    ax.set_yscale('log'); ax.set_title('NAV Comparado (log)', color=c['bench'])
    ax.legend(fontsize=8); ax.grid(True, alpha=0.4)

    # B: Drawdown
    ax = axes[0,1]
    dd_v1 = (nav_v1-np.maximum.accumulate(nav_v1))/np.maximum.accumulate(nav_v1)
    dd_v2 = (nav_v2-np.maximum.accumulate(nav_v2))/np.maximum.accumulate(nav_v2)
    ax.fill_between(dates, dd_v1.astype(float), 0, color=c['gold'],  alpha=0.25,
                    label=f'v1  MDD={dd_v1.min():.0%}')
    ax.fill_between(dates, dd_v2.astype(float), 0, color=c['model2'],alpha=0.30,
                    label=f'v2  MDD={dd_v2.min():.0%}')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{x:.0%}'))
    ax.set_title('Drawdown Comparado', color=c['bench'])
    ax.legend(fontsize=8); ax.grid(True, alpha=0.4)

    # C: Bootstrap distribution
    ax = axes[1,0]
    bins = np.linspace(-1.0, 2.5, 70)
    bv1 = bootstrap_sharpe(s_v1); bv2 = bootstrap_sharpe(s_v2)
    ax.hist(bv1['dist'], bins=bins, color=c['gold'],   alpha=0.45, density=True,
            label=f'v1  p={bv1["p_val"]:.3f}')
    ax.hist(bv2['dist'], bins=bins, color=c['model2'], alpha=0.50, density=True,
            label=f'v2  p={bv2["p_val"]:.4f}')
    ax.axvline(bv1['obs'], color=c['gold'],   lw=1.8)
    ax.axvline(bv2['obs'], color=c['model2'], lw=1.8)
    ax.axvline(0, color=c['bear'], lw=1.0, ls='--', alpha=0.6)
    ax.set_xlabel('Sharpe Bootstrap', color=c['bench'])
    ax.set_title('Distribución Bootstrap Sharpe', color=c['bench'])
    ax.legend(fontsize=8); ax.grid(True, alpha=0.4)

    # D: Peso comparado (zona de tiempo)
    ax = axes[1,1]
    ax.fill_between(dates, 0, w_v1_exec, color=c['gold'],   alpha=0.35, label='v1 peso')
    ax.fill_between(dates, 0, res_v2['w_exec'], color=c['model2'], alpha=0.40, label='v2 peso')
    ax.set_ylim(0, 1.1); ax.set_ylabel('Peso equity', color=c['bench'])
    ax.set_title(f'Peso Portfolio — v1 ({(w_v1_exec>0).mean():.0%} tiempo) vs '
                 f'v2 ({(res_v2["w_exec"]>0).mean():.0%} tiempo)', color=c['bench'])
    ax.legend(fontsize=8); ax.grid(True, alpha=0.4)

    fig.suptitle('COMPARACIÓN v1 (simétrico 55/55) vs v2 (histéresis 70/35 + cooldown 14d)',
                 color=c['bench'], fontsize=11, y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, 'fci_v1_vs_v2_comparison.png')
    plt.savefig(path, dpi=140, bbox_inches='tight', facecolor=c['bg'])
    plt.close(fig)
    print(f"  [✓] {path}")


def _banner():
    print()
    print("╔" + "═"*68 + "╗")
    print("║" + " MERVAL CCL/BADLAR — MODELO FCI v2 ".center(68) + "║")
    print("║" + " BOCPD × VS20% × Histéresis 70/35 × Cooldown 14d ".center(68) + "║")
    print("╚" + "═"*68 + "╝")
    print()
    print(f"  Parámetros activos:")
    print(f"  ├ Señal       : BOCPD  λ={Config.LAM_BOCPD}d")
    print(f"  ├ VS target   : {Config.VOL_TARGET:.0%}  EWMA span={Config.EWMA_SPAN}d")
    print(f"  ├ Thresh BUY  : {Config.THRESH_BUY:.0%}")
    print(f"  ├ Thresh SELL : {Config.THRESH_SELL:.0%}")
    print(f"  ├ Cooldown    : {Config.COOLDOWN_DAYS}d post-sell  (v1: 0d)")
    print(f"  ├ Rebalanceo  : {Config.REBAL_FREQ}  umbral={Config.MIN_CHANGE:.0%}")
    print(f"  ├ TC          : {Config.TC_BPS}bps one-way")
    print(f"  ├ Cash return : {Config.CASH_ANN:.0%} anual")
    print(f"  └ Burn-in     : {Config.MIN_TRAIN}d")
    print()


def _footer(m, bs, sig):
    print()
    print("╔" + "═"*68 + "╗")
    print("║" + " RESUMEN ".center(68) + "║")
    print("╚" + "═"*68 + "╝")
    print()
    print(f"  Sharpe     : {m['sr']:+.3f}  CI90%=[{bs['p5']:+.3f},{bs['p95']:+.3f}]  "
          f"p={bs['p_val']:.4f} {sig}")
    print(f"  Retorno    : {m['mu']:+.1%} anualizado")
    print(f"  Max DD     : {m['mdd']:.1%}")
    print(f"  Calmar     : {m['cal']:.3f}")
    print(f"  NAV final  : ×{m['nav_x']:.1f}")
    print(f"  Switches   : {m['sw_yr']:.1f}/año")
    print()
    print("╔" + "═"*68 + "╗")
    print("║" + " ANÁLISIS COMPLETADO ".center(68) + "║")
    print("╚" + "═"*68 + "╝")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Modelo FCI v2 Merval — BOCPD × VS × Histéresis × Cooldown')
    parser.add_argument('--data',    default=Config.DATA_PATH,
                        help='Ruta al archivo Excel de datos')
    parser.add_argument('--sheet',   default=Config.SHEET_NAME,
                        help='Nombre de la hoja en el Excel')
    parser.add_argument('--output',  default=Config.OUTPUT_DIR,
                        help='Directorio de salida para figuras y CSVs')
    parser.add_argument('--signal',  action='store_true',
                        help='Solo imprimir la señal del día, sin backtest completo')
    parser.add_argument('--prev-weight', type=float, default=None, metavar='W',
                        help='Peso actual del portfolio (0.0–1.0) para mostrar posición previa en zona gris')
    parser.add_argument('--compare', action='store_true',
                        help='Comparar v1 (simétrico 55/55) vs v2 (histéresis 70/35 + CD14)')
    args = parser.parse_args()

    if args.signal:
        ls = LiveSignal()
        ls.load_history(args.data, args.sheet)
        s  = ls.get_signal(prev_weight=args.prev_weight)
        print()
        print("SEÑAL ACTUAL — FCI v2")
        print("─" * 45)
        for k, v in s.items():
            print(f"  {k:<24}: {v}")
        print()
    elif args.compare:
        Config.DATA_PATH  = args.data
        Config.SHEET_NAME = args.sheet
        Config.OUTPUT_DIR = args.output
        compare_versions(data_path=args.data, out_dir=args.output)
    else:
        Config.DATA_PATH  = args.data
        Config.SHEET_NAME = args.sheet
        Config.OUTPUT_DIR = args.output
        result = run_backtest(data_path=args.data, out_dir=args.output)

        strategy_returns = pd.Series(
            result['res']['strat'],
            index=result['df'].index,
            name='strategy'
        )

        bh_returns = pd.Series(
            result['df']['LogRet'].values,
            index=result['df'].index,
            name='buy_and_hold'
        )
# ══════════════════════════════════════════════════════════════════════════════
# Quant Stats Report
# ══════════════════════════════════════════════════════════════════════════════

