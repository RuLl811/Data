# -*- coding: utf-8 -*-
"""
=====================================================================================
 PROYECCIÓN DE LA TASA DE INTERÉS REAL TAMAR — MODELO VASICEK DE 3 REGÍMENES
=====================================================================================

-------------------------------------------------------------------------------------
NOTA METODOLÓGICA (leer antes de tocar parámetros)
-------------------------------------------------------------------------------------
1) OBJETO MODELADO. Se modela exclusivamente la TASA REAL EX-ANTE MENSUAL de la TAMAR:
        r_real_t = (1 + TAMAR_TEM_t) / (1 + E[inflación]_t) - 1
   El usuario provee la inflación por separado para recomponer la nominal aguas abajo.
   Como proxy de E[inflación] se usa la inflación alineada un mes (forward del PF a 30d);
   si se dispone de una serie de expectativas (REM), reemplazar la columna y recalibrar.

2) UNIDADES — DISTINCIÓN CLAVE.
   - Los CENTROS (θ) son medias reales SOSTENIDAS por régimen: rango acotado (~ -0,25 a
     +0,40 %/mes ≈ -3% a +5% anual).
   - La BANDA (σ) refleja los TICKS mensuales: históricamente la real se movió ~ -2 a +4 %/m.
   - Ambas cosas son históricas, en estadísticos distintos. NO confundir el tick mensual
     (transitorio) con el nivel sostenido (lo que define el régimen).

3) TRES REGÍMENES (anclados a la historia, no a supuestos forward):
   - real negativo   ~ 2020-2021 / 2026
   - Neutral                     ~ promedio 2003-2019
   - real positivo             ~ 2025 (desinflación)
   Cada uno fija θ (centro) y un multiplicador de σ (dispersión); el real alto vino
   históricamente con más volatilidad, de ahí el multiplicador creciente.

5) ESTIMACIÓN. OLS sobre la forma discretizada de Vasicek:
        Δr_t = a + b·r_t + ε_t   ->   κ = -b/dt ,  θ = -a/b ,  σ = std(ε)/√dt
   κ se corrige por el sesgo AL ALZA de muestra corta (Marriott-Pope sobre el AR(1)).

6) INCERTIDUMBRE. Además de la incertidumbre de TRAYECTORIA (Monte Carlo), se PROPAGA la
   incertidumbre de PARÁMETROS: cada path sortea (κ,σ) de su distribución bootstrap
   paramétrica. Así la banda se ABRE con el horizonte (los paths de κ bajo revierten lento),
   en vez de aplanarse como con (κ,σ) fijos.

LIMITACIONES
   - Muestra de régimen corta (~25 meses): IC de κ amplio.
   - Vasicek no acota colas: paths extremos de real compuesta poco plausibles (cola, <5%).
   - El modelo es de UN régimen por escenario; los saltos discretos (parada súbita) requieren
     una capa Markov-switching, no incluida aquí.
=====================================================================================
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy.stats import kstest

# =====================================================================================
# CONFIGURACIÓN
# =====================================================================================
@dataclass
class Config:
    # --- Datos ---
    file_path: str = fr"C:\Users\lr110574\PycharmProjects\Estrategias\Estacionalidad Badlar\Badlar.xlsx"
    sheet: str = "Tamar"
    output_xlsx: str = "tasa_real_vasicek.xlsx"
    output_png: str = "tasa_real_vasicek.png"
    cal_start: str = "2024-06-01"
    exclude_cepo: Tuple[str, str] = ("2021-12-20", "2024-05-03")  # quiebre estructural
    # --- Horizonte y simulación (TODO MENSUAL) ---
    horizon_end: str = "2029-10-31"
    bond_life_m: int = 27                          # ventana de estimacion
    n_sim: int = 10_000                             # cantidad de simulaciones
    n_boot: int = 1_500
    dt: float = 1.0 / 12.0
    bias_correct: bool = True
    seed: int = 12345
    # --- Regímenes: θ en %/MES (centro sostenido) + multiplicador de σ (dispersión) ---
    regimes: Dict[str, dict] = field(default_factory=lambda: {
        "Real negativa":  dict(theta_m=-0.012, sig_mult=0.6, color="#C62828"),  # ≈ -3%
        "Neutral":     dict(theta_m=+0.0008, sig_mult=1.0, color="#1565C0"),  # ≈ +1%
        "Real positiva": dict(theta_m=+0.0020, sig_mult=1.4, color="#2E7D32")})  # ≈ +2%})


# =====================================================================================
# 1. DATOS
# =====================================================================================
def load_real_rate(cfg: Config) -> pd.DataFrame:
    """Carga la hoja y construye la tasa real ex-ante MENSUAL (decimal, %/mes)."""
    try:
        raw = pd.read_excel(cfg.file_path, sheet_name=cfg.sheet)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"No se encontró {cfg.file_path}") from e
    raw.columns = [c.strip().lower() for c in raw.columns]
    dcol = next(c for c in raw.columns if "date" in c or "fecha" in c)
    tcol = next(c for c in raw.columns if "tem" in c)         # TAMAR efectiva mensual
    icol = next(c for c in raw.columns if "infl" in c)        # inflación mensual (proxy expectativa)
    df = raw[[dcol, tcol, icol]].copy()
    df.columns = ["date", "tem", "infl"]
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna().sort_values("date")
    monthly = df.set_index("date").resample("ME").last().dropna()
    monthly["real"] = (1 + monthly["tem"]) / (1 + monthly["infl"]) - 1   # real ex-ante mensual
    return monthly


def calibration_sample(monthly: pd.DataFrame, cfg: Config) -> np.ndarray:
    """Muestra de régimen: excluye el quiebre estructural y arranca en cal_start."""
    s = monthly[~((monthly.index >= cfg.exclude_cepo[0]) & (monthly.index <= cfg.exclude_cepo[1]))]
    return s[s.index >= cfg.cal_start]["real"].values


# =====================================================================================
# 2. CALIBRACIÓN (Vasicek por OLS discretizado + corrección de sesgo)
# =====================================================================================
def fit_vasicek(rates: np.ndarray, dt: float, bias_correct: bool = True
                ) -> Tuple[float, float, float]:
    """
    OLS sobre  Δr = a + b·r  ->  κ = -b/dt ,  θ = -a/b ,  σ = std(resid)/√dt.
    Corrección de sesgo de muestra corta sobre el AR(1) φ = exp(-κ·dt) (Marriott-Pope).
    """
    dr, r0 = np.diff(rates), rates[:-1]
    b1, b0 = np.polyfit(r0, dr, 1)                 # pendiente, intercepto
    kappa = -b1 / dt
    theta = -b0 / b1
    if bias_correct and kappa > 0:
        phi = np.exp(-kappa * dt)
        n = len(rates)
        phi_corr = min(phi - (-(1 + 3 * phi) / n), 0.999)   # corrige sesgo AL ALZA de κ
        kappa = -np.log(phi_corr) / dt
    resid = dr - (b0 + b1 * r0)
    sigma = float(np.std(resid, ddof=2) / np.sqrt(dt))
    return float(kappa), float(theta), sigma


def ks_residual_test(rates: np.ndarray, kappa: float, theta: float, sigma: float,
                     dt: float) -> float:
    """KS de normalidad sobre residuos estandarizados (devuelve p-valor)."""
    dr, r0 = np.diff(rates), rates[:-1]
    resid = (dr - kappa * (theta - r0) * dt) / (sigma * np.sqrt(dt))
    return float(kstest(resid, "norm", args=(resid.mean(), resid.std()))[1])


def bootstrap_params(rates: np.ndarray, kappa: float, theta: float, sigma: float,
                     cfg: Config) -> np.ndarray:
    """
    Bootstrap PARAMÉTRICO: re-simula del modelo ajustado y re-estima (κ,σ).
    Válido para series AR (a diferencia del resample i.i.d.). Devuelve pares (κ,σ)
    recentrados a los estimadores puntuales (preserva dispersión y correlación).
    """
    n = len(rates)
    out = []
    for _ in range(cfg.n_boot):
        path = simulate_vasicek(rates[0], kappa, theta, sigma, n - 1, 1, cfg.dt)[0]
        try:
            k, _, s = fit_vasicek(path, cfg.dt, bias_correct=True)
            if np.isfinite(k) and np.isfinite(s) and k > 0 and s > 0:
                out.append((k, s))
        except Exception:
            continue
    pairs = np.array(out)
    pairs[:, 0] += kappa - pairs[:, 0].mean()      # recentrar κ
    pairs[:, 1] += sigma - pairs[:, 1].mean()      # recentrar σ
    pairs[:, 0] = np.clip(pairs[:, 0], 0.05, None)
    pairs[:, 1] = np.clip(pairs[:, 1], 1e-4, None)
    return pairs


# =====================================================================================
# 3. SIMULACIÓN (discretización EXACTA de Vasicek; κ,σ escalares o por path)
# =====================================================================================
def simulate_vasicek(r0: float, kappa, theta: float, sigma, steps: int, n_sim: int,
                     dt: float) -> np.ndarray:
    """
    Transición exacta:  r_{t+1} = θ + (r_t-θ)·e^{-κdt} + σ·√((1-e^{-2κdt})/(2κ))·Z.
    Gaussiana -> admite valores negativos. κ y σ pueden ser escalares (común) o arrays
    de largo n_sim (un (κ,σ) por path -> propagación de incertidumbre de parámetros).
    """
    kappa = np.asarray(kappa, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    ek = np.exp(-kappa * dt)
    step_sd = sigma * np.sqrt((1 - ek ** 2) / (2 * kappa))      # std exacta por paso
    r = np.empty((n_sim, steps + 1))
    r[:, 0] = r0
    for t in range(1, steps + 1):
        r[:, t] = theta + (r[:, t - 1] - theta) * ek + step_sd * np.random.standard_normal(n_sim)
    return r


# =====================================================================================
# 4. ESCENARIOS
# =====================================================================================
def run_scenarios(r0: float, kvec: np.ndarray, svec: np.ndarray, steps: int,
                  cfg: Config) -> Dict[str, dict]:
    """Simula cada régimen (θ propio, σ escalado) propagando (κ,σ) por path. Todo en %/mes."""
    res = {}
    for name, c in cfg.regimes.items():
        sim = simulate_vasicek(r0, kvec, c["theta_m"], svec * c["sig_mult"],
                               steps, cfg.n_sim, cfg.dt)[:, 1:]            # (n_sim, steps)
        # tasa real "sostenida" = media geométrica MENSUAL sobre la vida del bono
        sustained_m = np.prod(1 + sim[:, :cfg.bond_life_m], axis=1) ** (1 / cfg.bond_life_m) - 1
        res[name] = dict(
            color=c["color"], theta_m=c["theta_m"],
            mean=sim.mean(0),
            p05=np.percentile(sim, 5, 0),  p95=np.percentile(sim, 95, 0),
            p25=np.percentile(sim, 25, 0), p75=np.percentile(sim, 75, 0),
            paths=sim[:30],                 # muestra de paths representativos
            sustained=sustained_m,          # %/mes
        )
    return res

# =====================================================================================
# 5. GRÁFICOS — TODO EN %/MES
# =====================================================================================
def make_plots(monthly: pd.DataFrame, idx: pd.DatetimeIndex, res: Dict[str, dict],
               cfg: Config) -> None:
    pf = PercentFormatter(xmax=1.0, decimals=1)
    hist = monthly["real"].iloc[-36:]               # últimos 3 años de historia
    fig, ax = plt.subplots(2, 2, figsize=(15, 9))

    # (0,0) Historia + proyección (media + banda 5-95) — continuidad
    a = ax[0, 0]
    a.plot(hist.index, hist.values, color="black", lw=1.3, label="Real histórica")
    for name, r in res.items():
        a.plot(idx, r["mean"], color=r["color"], lw=2, label=f"{name} (media)")
        a.fill_between(idx, r["p05"], r["p95"], color=r["color"], alpha=0.08)
    a.axhline(0, color="grey", lw=0.6, ls="--")
    a.axvline(idx[0], color="grey", lw=0.8, ls=":")
    a.set_title("Historia + proyección — tasa real MENSUAL (media y banda 5–95%)")
    a.set_ylabel("tasa real (%/mes)"); a.yaxis.set_major_formatter(pf); a.legend(fontsize=8)

    # (0,1) Abanico por régimen: media + IQR + 5-95
    a = ax[0, 1]
    for name, r in res.items():
        a.plot(idx, r["mean"], color=r["color"], lw=2, label=name)
        a.fill_between(idx, r["p25"], r["p75"], color=r["color"], alpha=0.22)   # IQR
        a.fill_between(idx, r["p05"], r["p95"], color=r["color"], alpha=0.07)   # 5-95
    for ref in (-0.02, 0.04):                       # guías -2% / +4% mensual
        a.axhline(ref, color="grey", ls=":", lw=0.8)
    a.axhline(0, color="grey", lw=0.6, ls="--")
    a.set_title("Abanico por régimen — IQR (oscuro) y 5–95% (claro)\nguías -2% / +4% mensual")
    a.set_ylabel("tasa real (%/mes)"); a.yaxis.set_major_formatter(pf); a.legend(fontsize=8)

    # (1,0) Senderos individuales representativos (spaghetti) — textura
    a = ax[1, 0]
    for name, r in res.items():
        a.plot(idx, r["paths"].T, color=r["color"], lw=0.5, alpha=0.25)
        a.plot(idx, r["mean"], color=r["color"], lw=2.2, label=f"{name} (media)")
    a.axhline(0, color="grey", lw=0.6, ls="--")
    a.set_title("Senderos individuales representativos (30 por régimen)")
    a.set_ylabel("tasa real (%/mes)"); a.yaxis.set_major_formatter(pf); a.legend(fontsize=8)

    # (1,1) Distribución de la real SOSTENIDA (media geom. mensual sobre la vida del bono)
    a = ax[1, 1]
    data = [res[n]["sustained"] for n in res]
    bp = a.boxplot(data, whis=(5, 95), showfliers=False, patch_artist=True,
                   tick_labels=[n.split(" (")[0] for n in res])
    for patch, name in zip(bp["boxes"], res):
        patch.set_facecolor(res[name]["color"]); patch.set_alpha(0.4)
    for med in bp["medians"]:
        med.set_color("black")
    a.axhline(0, color="grey", lw=0.6, ls="--")
    a.set_title(f"Tasa real SOSTENIDA (media geom. mensual, vida bono {cfg.bond_life_m}m)\n"
                "caja = IQR · bigotes = 5–95%")
    a.set_ylabel("tasa real (%/mes)"); a.yaxis.set_major_formatter(pf)

    fig.suptitle("Tasa de interés REAL TAMAR — Vasicek de 3 regímenes (unidades mensuales)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(cfg.output_png, dpi=130)
    print(f"Gráfico guardado: {cfg.output_png}")


# =====================================================================================
# 6. EXPORTACIÓN
# =====================================================================================
def export_excel(idx: pd.DatetimeIndex, res: Dict[str, dict], cfg: Config) -> None:
    senderos = pd.DataFrame(index=idx)
    for name, r in res.items():
        senderos[f"{name} | media"] = r["mean"]
        senderos[f"{name} | p05"] = r["p05"]
        senderos[f"{name} | p25"] = r["p25"]
        senderos[f"{name} | p75"] = r["p75"]
        senderos[f"{name} | p95"] = r["p95"]
    sostenida = pd.DataFrame({n: res[n]["sustained"] for n in res})
    with pd.ExcelWriter(cfg.output_xlsx) as xl:
        senderos.to_excel(xl, sheet_name="Sendero_mensual")
        sostenida.describe(percentiles=[.05, .25, .5, .75, .95]).to_excel(
            xl, sheet_name="Real_sostenida_mensual")
    print(f"Excel guardado: {cfg.output_xlsx}")


# =====================================================================================
# MAIN
# =====================================================================================
def main(cfg: Config = Config()) -> None:
    np.random.seed(cfg.seed)
    pct = lambda x: f"{x:+.3%}"

    monthly = load_real_rate(cfg)
    real = calibration_sample(monthly, cfg)
    r0 = float(monthly["real"].iloc[-1])
    steps = (pd.Period(cfg.horizon_end, "M") - monthly.index[-1].to_period("M")).n
    idx = pd.date_range(monthly.index[-1], periods=steps + 1, freq="ME")[1:]

    # --- Calibración de la DINÁMICA (κ,σ); θ se fija por régimen ---
    kappa, theta_free, sigma = fit_vasicek(real, cfg.dt, cfg.bias_correct)
    ks_p = ks_residual_test(real, kappa, theta_free, sigma, cfg.dt)
    print("=" * 70)
    print(f"Muestra de régimen: {len(real)} meses desde {cfg.cal_start}")
    print(f"r0 (última real)   : {pct(r0)} /mes")
    print(f"Vasicek libre      : κ={kappa:.2f}  θ={pct(theta_free)}/mes  σ={sigma:.4f}")
    print(f"  std mensual estac.: {sigma*np.sqrt(1/(2*kappa)):.3%}/mes (histórica {real.std():.3%}/mes)")
    print(f"  KS residuos p-val : {ks_p:.2f}  ({'normal' if ks_p>0.05 else 'no normal'})")

    # --- Incertidumbre de parámetros: sorteo (κ,σ) por path ---
    pairs = bootstrap_params(real, kappa, theta_free, sigma, cfg)
    draw = pairs[np.random.randint(0, len(pairs), cfg.n_sim)]
    kvec, svec = draw[:, 0], draw[:, 1]
    print(f"Bootstrap (n={len(pairs)}): κ 5–95%=[{np.percentile(pairs[:,0],5):.2f}, "
          f"{np.percentile(pairs[:,0],95):.2f}]  σ 5–95%=[{np.percentile(pairs[:,1],5):.4f}, "
          f"{np.percentile(pairs[:,1],95):.4f}]")
    print("=" * 70)

    # --- Escenarios ---
    res = run_scenarios(r0, kvec, svec, steps, cfg)
    print(f"{'Régimen':<24}{'θ (%/m)':>10}{'real sostenida mensual (%/m)':>34}")
    print(f"{'':<24}{'':>10}{'IQR (típico)':>22}{'5–95% (cola)':>12}")
    for name, r in res.items():
        s = r["sustained"]
        print(f"{name:<24}{r['theta_m']:>+9.3%}"
              f"   [{np.percentile(s,25):>+6.3%}, {np.percentile(s,75):>+6.3%}]"
              f"  [{np.percentile(s,5):>+6.3%}, {np.percentile(s,95):>+6.3%}]")
    print("=" * 70)

    make_plots(monthly, idx, res, cfg)
    export_excel(idx, res, cfg)


if __name__ == "__main__":
    main()
