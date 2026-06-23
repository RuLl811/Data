# -*- coding: utf-8 -*-
"""
=====================================================================================
 VALUACIÓN DE LA OPCIÓN EMBEBIDA EN UN BONO DUAL (CER / TAMAR) — SISTEMA INTEGRAL
=====================================================================================
Importa la TRAYECTORIA de real_TAMAR (Excel), toma los inputs de TIR/términos del dual,
y devuelve el VALOR DE LA OPCIÓN con su descomposición (intrínseco / valor tiempo),
P(ITM) y, opcionalmente, el diagnóstico de fair-value contra el mercado.

-------------------------------------------------------------------------------------
QUÉ ES LO QUE SE VALÚA
-------------------------------------------------------------------------------------
El dual paga a vencimiento  max(pata_CER, pata_TAMAR) = pata_CER + max(0, pata_TAMAR − pata_CER).
El término max(·) es una OPCIÓN DE CANJE (europea). En términos REALES la pata CER es
determinística ((1+spread_CER)^t) y la pata TAMAR es estocástica (real_TAMAR acumulado +
margen). => la opción es una CALL sobre el real_TAMAR acumulado, strike = pata CER.

-------------------------------------------------------------------------------------
AUDITORÍA DE COMPLETITUD — LO QUE EL SISTEMA ASUME / LO QUE NO HAY QUE OLVIDAR
-------------------------------------------------------------------------------------
1) MEDIDA (crítico). Las trayectorias importadas están en medida REAL-WORLD (drift =
   historia/regímenes). El precio de mercado de la opción es RISK-NEUTRAL. Por eso el
   output se etiqueta como VALOR ESPERADO real-world, NO como precio de arbitraje. La cuña
   entre ambos = prima de riesgo de la pata TAMAR. Para juzgar cara/barata NO se compara el
   nivel directo: se compara VOL IMPLÍCITA (del premium de mercado) vs VOL DEL MODELO,
   fijando el forward con las curvas (bloque fair-value).
2) DESCUENTO. El payoff es a vencimiento -> se trae a hoy con una TIR REAL (real_discount_rate).
3) REAL vs NOMINAL. La trayectoria DEBE ser real_TAMAR mensual (así la inflación se cancela y
   la pata CER es determinística). Si fueran TAMAR nominales, faltaría la trayectoria de
   inflación para ambas patas -> el sistema valida que el nivel sea plausible de 'real'.
4) HORIZONTE. La trayectoria debe cubrir la vida del bono; se truncan los primeros bond_life_m
   meses y se valida que alcancen.
5) PESOS DE RÉGIMEN. Si se importan varias hojas (una por régimen), el valor 'del' dual es el
   blend con probabilidades de régimen (regime_weights). Sin pesos -> equiponderado + se
   reportan los condicionales por régimen igual.
6) TÉRMINOS DEL DUAL ≠ COMPARABLES. El strike es la pata CER del PROPIO dual (su spread
   contractual), y la pata TAMAR usa el margen contractual del dual. Los precios de mercado
   (dual y CER puro) son inputs SEPARADOS, solo para el fair-value.
7) EUROPEA. Se asume max evaluado a vencimiento. Si hubiera observación intermedia, no aplica.
8) VOL DEL MODELO. Si la trayectoria viene de un modelo de UN régimen, subestima la cola de
   cambio de régimen -> la opción se verá barata sistemáticamente. Importar sims con
   regímenes/saltos para el juicio de fair-value.
=====================================================================================
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq


# =====================================================================================
# CONFIGURACIÓN
# =====================================================================================
@dataclass
class Config:
    # --- Trayectoria (Excel de input) ---
    traj_path: str = "trayectoria_tamar.xlsx"
    traj_sheet_prefix: str = "sims_"          # hojas a leer (una por régimen)
    months_in_rows: bool = True               # orientación: meses en filas, paths en columnas
    has_index_col: bool = True                # 1ra columna = fechas/índice (se descarta)
    paths_are_monthly_real: bool = True        # valores = real MENSUAL (no anual, no nominal)

    # --- Términos del DUAL (contractuales) ---
    bond_life_m: int = 27
    cer_real_spread: float = 0.07             # pata CER del dual: CER + spread  -> STRIKE
    tamar_margin: float = 0.069               # pata TAMAR del dual: TAMAR + margen

    # --- TIR / descuento ---
    real_discount_rate: float = 0.07          # TIR real para traer el payoff a hoy

    # --- Pesos de régimen para el blend (None = equiponderado) ---
    regime_weights: Optional[Dict[str, float]] = None

    # --- Comparación con mercado (opcional, fair-value). None = se omite. ---
    market_option_premium: Optional[float] = None  # premium que cobra el mercado (= P_dual − P_CERpuro), real
    capital: float = 1.0                            # base; opción se expresa por unidad de capital


# =====================================================================================
# 1. IMPORTACIÓN DE TRAYECTORIAS
# =====================================================================================
def load_trajectories(cfg: Config) -> Dict[str, np.ndarray]:
    """Lee las hojas de paths y devuelve {regimen: matriz (n_sim, n_meses)} en real MENSUAL."""
    xls = pd.ExcelFile(cfg.traj_path)
    sheets = [s for s in xls.sheet_names if s.startswith(cfg.traj_sheet_prefix)]
    if not sheets:
        raise ValueError(f"No hay hojas con prefijo '{cfg.traj_sheet_prefix}' en {cfg.traj_path}")
    out = {}
    for sh in sheets:
        df = pd.read_excel(xls, sheet_name=sh)
        if cfg.has_index_col:
            df = df.iloc[:, 1:]                       # descarta columna índice
        arr = df.to_numpy(dtype=float)
        if cfg.months_in_rows:
            arr = arr.T                               # -> (n_sim, n_meses)
        # validación de unidades: real mensual plausible (|valor| < 0.2/m)
        if cfg.paths_are_monthly_real and np.nanmedian(np.abs(arr)) > 0.2:
            raise ValueError(f"Hoja '{sh}': valores no parecen real MENSUAL (¿anualizado/nominal?).")
        out[sh] = arr
    return out


# =====================================================================================
# 2. VALUACIÓN DE LA OPCIÓN (real-world, descontada)
# =====================================================================================
def value_option(real_paths: np.ndarray, cfg: Config) -> dict:
    """
    Opción de canje sobre el real_TAMAR acumulado, strike = pata CER.
    Devuelve valores DESCONTADOS por unidad de capital y la descomposición.
    """
    H = cfg.bond_life_m
    if real_paths.shape[1] < H:
        raise ValueError(f"Trayectoria con {real_paths.shape[1]} meses < vida del bono {H}.")
    paths = real_paths[:, :H]                                   # truncar al horizonte del bono
    t = H / 12.0
    df = 1.0 / (1.0 + cfg.real_discount_rate) ** t              # factor de descuento real

    # patas como FACTOR real terminal (capitaliza mensual)
    leg_tamar = np.prod(1.0 + paths + cfg.tamar_margin / 12.0, axis=1)   # estocástica
    leg_cer = (1.0 + cfg.cer_real_spread) ** t                          # determinística (strike)

    payoff = np.maximum(leg_tamar - leg_cer, 0.0)
    option_pv = payoff.mean() * df
    intrinsic_pv = max(leg_tamar.mean() - leg_cer, 0.0) * df            # forward/moneyness
    time_value = option_pv - intrinsic_pv
    p_itm = float((leg_tamar > leg_cer).mean())

    # vol del modelo del factor TAMAR terminal (lognormal-equiv, anualizada) -> para fair-value
    vol_model = float(np.std(np.log(leg_tamar)) / np.sqrt(t))

    return dict(option_pv=option_pv * cfg.capital, intrinsic_pv=intrinsic_pv * cfg.capital,
                time_value=time_value * cfg.capital, p_itm=p_itm,
                fwd_tamar=float(leg_tamar.mean()), strike=float(leg_cer),
                vol_model=vol_model, df=df, t=t)


# =====================================================================================
# 3. FAIR-VALUE: vol implícita del premium de mercado vs vol del modelo
# =====================================================================================
def black76_call(F: float, K: float, sigma: float, t: float, df: float) -> float:
    if sigma <= 0:
        return df * max(F - K, 0.0)
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    return df * (F * norm.cdf(d1) - K * norm.cdf(d2))


def implied_vol(premium: float, F: float, K: float, t: float, df: float) -> Optional[float]:
    """Despeja la vol que reproduce el premium de mercado, fijando el forward con las curvas."""
    intrinsic = df * max(F - K, 0.0)
    if premium <= intrinsic + 1e-9:
        return 0.0                                   # premium ≤ intrínseco -> vol implícita ~0
    try:
        return float(brentq(lambda s: black76_call(F, K, s, t, df) - premium, 1e-4, 5.0))
    except ValueError:
        return None


# =====================================================================================
# MAIN
# =====================================================================================
def main(cfg: Config = Config()) -> None:
    trajs = load_trajectories(cfg)
    names = list(trajs.keys())
    weights = cfg.regime_weights or {n: 1.0 / len(names) for n in names}
    wsum = sum(weights.get(n, 0) for n in names)
    weights = {n: weights.get(n, 0) / wsum for n in names}     # normalizar

    print("=" * 78)
    print(f"DUAL: max(CER+{cfg.cer_real_spread:.2%} , TAMAR+{cfg.tamar_margin:.2%})  | "
          f"vida {cfg.bond_life_m}m | descuento real {cfg.real_discount_rate:.2%}")
    print("Valores por unidad de capital, DESCONTADOS, en términos REALES (medida real-world).")
    print("=" * 78)
    print(f"{'Régimen':<16}{'peso':>7}{'P(ITM)':>9}{'Opción':>10}{'Intríns.':>10}{'V.tiempo':>10}{'vol_mod':>9}")

    blended = dict(option_pv=0.0, intrinsic_pv=0.0, time_value=0.0, p_itm=0.0)
    per = {}
    for n in names:
        r = value_option(trajs[n], cfg)
        per[n] = r
        w = weights[n]
        for k in blended:
            blended[k] += w * r[k]
        print(f"{n.replace(cfg.traj_sheet_prefix,''):<16}{w:>7.0%}{r['p_itm']:>8.0%}"
              f"{r['option_pv']:>10.4f}{r['intrinsic_pv']:>10.4f}{r['time_value']:>10.4f}{r['vol_model']:>9.1%}")
    print("-" * 78)
    print(f"{'BLEND':<16}{'100%':>7}{blended['p_itm']:>8.0%}"
          f"{blended['option_pv']:>10.4f}{blended['intrinsic_pv']:>10.4f}{blended['time_value']:>10.4f}")

    # --- Fair-value opcional ---
    if cfg.market_option_premium is not None:
        F = sum(weights[n] * per[n]['fwd_tamar'] for n in names)   # forward TAMAR blended
        K = per[names[0]]['strike']; t = per[names[0]]['t']; df = per[names[0]]['df']
        vol_mod = sum(weights[n] * per[n]['vol_model'] for n in names)
        iv = implied_vol(cfg.market_option_premium, F, K, t, df)
        print("\n" + "=" * 78); print("FAIR-VALUE (vol implícita de mercado vs vol del modelo)")
        print(f"  Premium de mercado      : {cfg.market_option_premium:.4f}")
        print(f"  Valor modelo (blend)    : {blended['option_pv']:.4f}   <- real-world, NO comparar nivel directo")
        print(f"  Forward TAMAR / strike  : {F:.4f} / {K:.4f}")
        if iv is None:
            print("  Vol implícita           : no converge (premium fuera de rango)")
        else:
            print(f"  Vol implícita (mercado) : {iv:.1%}")
            print(f"  Vol modelo (real_TAMAR) : {vol_mod:.1%}")
            verdict = ("opción BARATA (dual barato): el mercado pricea menos vol que el modelo"
                       if iv < vol_mod else
                       "opción CARA (o el mercado pricea cola de régimen que el modelo no capta)")
            print(f"  Veredicto               : {verdict}")
        print("  Recordatorio: la cuña real-world vs risk-neutral es prima de riesgo, no mispricing.")
    print("=" * 78)


if __name__ == "__main__":
    main()
