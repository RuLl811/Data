"""
================================================================================
SISTEMA INTEGRADO DE PRONÓSTICO ELECTORAL — ARGENTINA 2027
SARIMAX-ICG + MODELO ELECTORAL XGB (v7) + MONTE CARLO
VERSIÓN UNIVARIADA (solo ICG)
================================================================================

Este módulo conecta dos modelos para producir una DISTRIBUCIÓN COMPLETA
del pronóstico electoral 2027, en lugar de escenarios manuales:

    1) SARIMAX-ICG: proyecta la trayectoria del ICG mes a mes hasta
       sep-2027, con incertidumbre dinámica.

    2) Modelo electoral XGB monótono (v7): mapea ICG → % voto.

    3) Monte Carlo: simula N trayectorias del ICG, propaga cada una por
       el modelo electoral, y agrega el error del propio modelo electoral.
       Resultado: distribución completa del voto 2027.

--------------------------------------------------------------------------------
POR QUÉ ESTA VERSIÓN ES UNIVARIADA (decisión documentada)
--------------------------------------------------------------------------------
Las versiones previas del sistema integrado simulaban TAMBIÉN el IPC mensual
y lo pasaban como segundo input al modelo electoral. Se removió el IPC por
tres razones, en orden de importancia:

  1. El modelo electoral XGB v7, tras imponer monotonic constraints
     (ICG creciente, IPC decreciente), dejó de usar el IPC: la grilla 2D
     mostró predicción idéntica para todo el rango de IPC. El "efecto
     inflación" que el GBM libre capturaba era confounding por casos
     peronistas de alta inflación, no señal causal. El ICG ya absorbe el
     canal inflacionario (cuando la inflación sube, la confianza cae).

  2. El SARIMAX-IPC tenía skill NEGATIVO en backtesting (-18.7%): proyectaba
     peor que el naïve. Mantenerlo en producción era arrastrar un componente
     roto que no hacía daño solo porque el modelo electoral lo ignoraba —
     una mina latente para futuras reespecificaciones.

  3. Higiene y honestidad del modelo: el sistema con IPC APARENTABA ser un
     modelo de dos factores sin serlo. Un sistema que declara lo que hace
     (univariado del ICG) es más barato de auditar y más defendible ante
     un comité.

CONCLUSIÓN: el pronóstico electoral 2027 depende, en los hechos, de una sola
variable proyectable: el ICG (Índice de Confianza en el Gobierno, UTDT).
El IPC fue evaluado exhaustivamente y removido por decisión razonada.

NOTA: el modelo electoral XGB v7 sigue siendo formalmente bivariado en disco
(fue entrenado con ICG e IPC). Para alimentarlo desde este sistema univariado
se le pasa el IPC mediana histórica como valor constante placebo — irrelevante
para el output dado que el modelo no lo usa, pero necesario para la firma de
la función predict(). Ver `IPC_PLACEBO` más abajo.
--------------------------------------------------------------------------------

COMPONENTES Y VALIDACIÓN:
    SARIMAX-ICG : ARIMA(1,1,1) — mejor en backtesting expanding window
                  24m (MAE 0.134, skill +5.9% vs naïve).
    Modelo XGB  : ver modelo_electoral_produccion_v7.py
                  MAE LOO 2.90 pp | MAE expanding 3.64 pp.

FUENTES DE INCERTIDUMBRE PROPAGADAS:
    1. Incertidumbre de la trayectoria del ICG (la dominante)
    2. Error de pronóstico del modelo electoral (sigma = RMSE expanding)

LIMITACIONES:
    - El SARIMAX-ICG asume que la dinámica histórica del ICG se mantiene.
      Un shock político/económico estructural no está modelado.
    - Horizonte largo (17 meses): la proyección del ICG está fuertemente
      dominada por reversión a la media. La incertidumbre es alta y honesta.
    - No incorpora encuestas de intención de voto (no disponibles aún para
      2027). Cuando se publiquen, deberían combinarse con este sistema.

Autor   : Ruben + Claude
Versión : v2.0 (sistema integrado univariado ICG, mayo 2026)
================================================================================
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Importamos el modelo electoral v7
import sys
sys.path.insert(0, '.')
from modelo_electoral_produccion_v7 import (
    cargar_datos, construir_dataset, ajustar_modelo as ajustar_modelo_electoral,
)


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

CONFIG = {
    'data_path'          : 'base_elecciones.xlsx',

    # === Fecha objetivo y ventana electoral ===
    'fecha_eleccion'     : '2027-10-31',
    'ventana_electoral_m': 6,        # meses previos que promedia el modelo

    # === SARIMAX-ICG ===
    'sarimax_icg_order'   : (1, 1, 1),
    'sarimax_icg_seasonal': (0, 0, 0, 12),

    # === Monte Carlo ===
    'n_simulaciones'     : 10000,
    'random_seed'        : 42,

    # === Error del modelo electoral (RMSE expanding window de v7) ===
    'sigma_modelo_elec'  : 0.0364,

    # === Umbrales de eventos electorales ===
    'umbral_gana_1ra'    : 0.45,     # ≥45% gana cómodo en 1ra vuelta
    'umbral_competitivo' : 0.40,
    'umbral_riesgo'      : 0.35,
}

# Valor placebo del IPC para alimentar la firma bivariada del modelo XGB v7.
# El modelo no usa el IPC (ver docstring), así que este valor es irrelevante
# para el output. Se setea en el pipeline a partir de la mediana histórica.
IPC_PLACEBO: float = None


# ============================================================================
# MODELO DE SERIE DE TIEMPO — ICG
# ============================================================================

def cargar_serie_icg(path: str) -> pd.Series:
    """Carga y normaliza la serie mensual del ICG."""
    df = pd.read_excel(path, sheet_name='base')
    df['Dates'] = pd.to_datetime(df['Dates'])
    df = df.sort_values('Dates').reset_index(drop=True)
    icg = df.set_index('Dates')['icg'].dropna().resample('MS').mean().interpolate()
    return icg


def cargar_ipc_placebo(path: str) -> float:
    """
    Mediana histórica del IPC mensual. Solo se usa como valor constante
    para satisfacer la firma bivariada del modelo electoral XGB v7,
    que NO usa el IPC (ver docstring del módulo).
    """
    df = pd.read_excel(path, sheet_name='base')
    return float(df['ipc_mom'].dropna().median())


def ajustar_sarimax(serie: pd.Series, order: tuple, seasonal: tuple):
    """Ajusta un SARIMAX a una serie."""
    return SARIMAX(serie, order=order, seasonal_order=seasonal,
                   enforce_stationarity=False,
                   enforce_invertibility=False).fit(disp=False)


def backtesting_sarimax(serie: pd.Series, order: tuple, seasonal: tuple,
                         n_test: int = 24) -> dict:
    """Valida un SARIMAX con expanding window."""
    errs = []
    for t in range(len(serie) - n_test, len(serie)):
        train = serie.iloc[:t]
        try:
            fit = ajustar_sarimax(train, order, seasonal)
            pred = fit.forecast(1).iloc[0]
            errs.append(abs(serie.iloc[t] - pred))
        except Exception:
            errs.append(np.nan)
    errs_naive = [abs(serie.iloc[t] - serie.iloc[t-1])
                  for t in range(len(serie) - n_test, len(serie))]
    mae = np.nanmean(errs)
    mae_naive = np.mean(errs_naive)
    return {
        'mae'      : mae,
        'rmse'     : np.sqrt(np.nanmean(np.array(errs)**2)),
        'mae_naive': mae_naive,
        'skill'    : (mae_naive - mae) / mae_naive * 100,
    }


# ============================================================================
# HORIZONTE DE PROYECCIÓN
# ============================================================================

def calcular_horizonte(serie_icg: pd.Series) -> dict:
    """Calcula meses desde la última observación hasta la ventana electoral."""
    ultima_fecha   = serie_icg.index.max()
    fecha_eleccion = pd.Timestamp(CONFIG['fecha_eleccion'])
    # La ventana electoral termina el mes anterior a la elección
    ventana_fin    = fecha_eleccion - pd.DateOffset(months=1)
    ventana_inicio = ventana_fin - pd.DateOffset(months=CONFIG['ventana_electoral_m'] - 1)

    h_total = ((ventana_fin.year - ultima_fecha.year) * 12 +
               (ventana_fin.month - ultima_fecha.month))

    meses_a_inicio_ventana = ((ventana_inicio.year - ultima_fecha.year) * 12 +
                              (ventana_inicio.month - ultima_fecha.month))
    idx_inicio = meses_a_inicio_ventana - 1   # -1 porque proyección[0] = mes+1
    idx_fin    = h_total                       # exclusivo

    return {
        'ultima_fecha'   : ultima_fecha,
        'ventana_inicio' : ventana_inicio,
        'ventana_fin'    : ventana_fin,
        'h_total'        : h_total,
        'idx_ventana'    : slice(idx_inicio, idx_fin),
    }


# ============================================================================
# SIMULACIÓN MONTE CARLO
# ============================================================================

@dataclass
class ResultadoSimulacion:
    # Input simulado
    icg_ventana: np.ndarray
    # Outputs
    pred_sin_ruido: np.ndarray   # solo incertidumbre del ICG
    pred_total: np.ndarray       # + error del modelo electoral
    # Resúmenes
    horizonte: dict
    backtesting_icg: dict
    ipc_placebo: float
    estadisticas: dict = field(default_factory=dict)
    probabilidades: dict = field(default_factory=dict)
    sensibilidad_icg: pd.DataFrame = None   # tabla ICG fijo → pronóstico


def analisis_sensibilidad_icg(modelo_elec, ipc_placebo: float,
                               sigma_modelo: float,
                               icg_grid: list = None,
                               n_sim_ruido: int = 20000,
                               seed: int = 42) -> pd.DataFrame:
    """
    Análisis de sensibilidad: ¿cómo cambia el pronóstico electoral si el
    ICG promedio 6m de la ventana electoral tomara un valor FIJO conocido?

    A diferencia del Monte Carlo (que propaga la incertidumbre de NO saber
    el ICG futuro), esto responde la pregunta condicional:
        "SI el ICG promedio termina siendo X, ¿qué resultado esperamos?"

    Es la tabla que el comité necesita para monitoreo: a medida que se
    acerque 2027 y el ICG observado se acerque a un valor, esta tabla dice
    directamente qué pronóstico implica.

    Para cada nivel de ICG se reporta:
      - Pronóstico puntual (determinístico, del modelo electoral)
      - Distribución agregando SOLO el error del modelo electoral
        (sigma_modelo), NO la incertidumbre del ICG (que aquí es dato)
      - Probabilidades de los eventos electorales relevantes
    """
    if icg_grid is None:
        # Grilla que cubre el rango histórico + algo de extrapolación
        icg_grid = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]

    rng = np.random.default_rng(seed)
    u_gana  = CONFIG['umbral_gana_1ra']
    u_comp  = CONFIG['umbral_competitivo']
    u_riesg = CONFIG['umbral_riesgo']

    rows = []
    for icg in icg_grid:
        # Pronóstico puntual del modelo electoral
        punto = float(modelo_elec.predict(np.array([[icg, ipc_placebo]]))[0])

        # Distribución condicional: ICG fijo, solo error del modelo electoral
        ruido = rng.normal(0, sigma_modelo, n_sim_ruido)
        dist  = np.clip(punto + ruido, 0, 1)

        rows.append({
            'icg_6m'        : icg,
            'pronostico'    : punto,
            'p5'            : float(np.percentile(dist, 5)),
            'p50'           : float(np.percentile(dist, 50)),
            'p95'           : float(np.percentile(dist, 95)),
            'p_gana_1ra'    : float((dist >= u_gana).mean()),
            'p_competitivo' : float(((dist >= u_comp) & (dist < u_gana)).mean()),
            'p_zona_riesgo' : float(((dist >= u_riesg) & (dist < u_comp)).mean()),
            'p_derrota'     : float((dist < u_riesg).mean()),
        })

    return pd.DataFrame(rows)


def simular(path: str = None) -> ResultadoSimulacion:
    """Pipeline completo de simulación univariada."""
    global IPC_PLACEBO

    if path is None:
        path = CONFIG['data_path']

    rng = np.random.default_rng(CONFIG['random_seed'])

    # --- 1. Modelo electoral ---
    df_base, df_elec = cargar_datos(path)
    df_modelo  = construir_dataset(df_base, df_elec)
    modelo_elec = ajustar_modelo_electoral(df_modelo)

    # --- 2. Serie ICG y SARIMAX ---
    serie_icg = cargar_serie_icg(path)
    IPC_PLACEBO = cargar_ipc_placebo(path)

    bt_icg = backtesting_sarimax(serie_icg, CONFIG['sarimax_icg_order'],
                                  CONFIG['sarimax_icg_seasonal'])
    fit_icg = ajustar_sarimax(serie_icg, CONFIG['sarimax_icg_order'],
                               CONFIG['sarimax_icg_seasonal'])

    # --- 3. Horizonte ---
    horizonte = calcular_horizonte(serie_icg)
    H = horizonte['h_total']

    # --- 4. Simular trayectorias del ICG ---
    # IMPORTANTE: fit.simulate() de statsmodels NO usa el `rng` local de
    # numpy.default_rng — hay que pasarle random_state explícitamente, o el
    # sistema NO es reproducible (cada corrida con los mismos datos daría
    # un resultado distinto). Ver auditoría / CHECK 6.
    N = CONFIG['n_simulaciones']
    sim_icg = np.asarray(
        fit_icg.simulate(nsimulations=H, repetitions=N, anchor='end',
                         random_state=CONFIG['random_seed'])
    ).reshape(H, -1)

    # --- 5. Extraer ventana electoral (promedio 6m) ---
    icg_ventana = sim_icg[horizonte['idx_ventana'], :].mean(axis=0)

    # --- 6. Propagar por el modelo electoral ---
    # El modelo XGB v7 tiene firma bivariada (ICG, IPC) pero NO usa el IPC.
    # Se le pasa IPC_PLACEBO constante para satisfacer la firma.
    X_sim = np.column_stack([icg_ventana,
                             np.full(len(icg_ventana), IPC_PLACEBO)])
    pred_sin_ruido = modelo_elec.predict(X_sim)

    # --- 7. Agregar error del modelo electoral ---
    ruido = rng.normal(0, CONFIG['sigma_modelo_elec'], len(pred_sin_ruido))
    pred_total = np.clip(pred_sin_ruido + ruido, 0, 1)

    # --- 8. Estadísticas ---
    def resumen(arr):
        return {
            'media'  : float(arr.mean()),
            'mediana': float(np.median(arr)),
            'std'    : float(arr.std()),
            'p5'     : float(np.percentile(arr, 5)),
            'p25'    : float(np.percentile(arr, 25)),
            'p50'    : float(np.percentile(arr, 50)),
            'p75'    : float(np.percentile(arr, 75)),
            'p95'    : float(np.percentile(arr, 95)),
        }

    estadisticas = {
        'icg_ventana'    : resumen(icg_ventana),
        'pred_sin_ruido' : resumen(pred_sin_ruido),
        'pred_total'     : resumen(pred_total),
    }

    # --- 9. Probabilidades de eventos ---
    u_gana  = CONFIG['umbral_gana_1ra']
    u_comp  = CONFIG['umbral_competitivo']
    u_riesg = CONFIG['umbral_riesgo']
    probabilidades = {
        'p_gana_1ra'    : float((pred_total >= u_gana).mean()),
        'p_competitivo' : float(((pred_total >= u_comp) & (pred_total < u_gana)).mean()),
        'p_zona_riesgo' : float(((pred_total >= u_riesg) & (pred_total < u_comp)).mean()),
        'p_derrota'     : float((pred_total < u_riesg).mean()),
        'p_mayor_40'    : float((pred_total > 0.40).mean()),
        'p_mayor_35'    : float((pred_total > 0.35).mean()),
    }

    # --- 10. Análisis de sensibilidad al ICG ---
    # Tabla condicional: "SI el ICG promedio fuera X, ¿qué pronóstico?"
    sensibilidad = analisis_sensibilidad_icg(
        modelo_elec, IPC_PLACEBO, CONFIG['sigma_modelo_elec'],
        seed=CONFIG['random_seed'],
    )

    return ResultadoSimulacion(
        icg_ventana=icg_ventana,
        pred_sin_ruido=pred_sin_ruido, pred_total=pred_total,
        horizonte=horizonte, backtesting_icg=bt_icg, ipc_placebo=IPC_PLACEBO,
        estadisticas=estadisticas, probabilidades=probabilidades,
        sensibilidad_icg=sensibilidad,
    )


# ============================================================================
# REPORTING
# ============================================================================

def print_header(text, char='='):
    print(); print(char*78); print(text); print(char*78)


def fmt_pct(x, dec=1):
    return f"{x*100:.{dec}f}%"


def reporte(res: ResultadoSimulacion):
    print_header('SISTEMA INTEGRADO DE PRONÓSTICO ELECTORAL — ARGENTINA 2027')
    print(f"SARIMAX-ICG + Modelo electoral XGB (v7) + Monte Carlo")
    print(f"Versión      : UNIVARIADA (solo ICG — ver docstring del módulo)")
    print(f"Simulaciones : {CONFIG['n_simulaciones']:,}")
    print(f"Fecha elección target : {CONFIG['fecha_eleccion']}")

    print_header('1. HORIZONTE DE PROYECCIÓN', '-')
    h = res.horizonte
    print(f"Última observación disponible : {h['ultima_fecha'].strftime('%Y-%m')}")
    print(f"Ventana electoral             : {h['ventana_inicio'].strftime('%Y-%m')} "
          f"a {h['ventana_fin'].strftime('%Y-%m')}")
    print(f"Horizonte total de proyección : {h['h_total']} meses")

    print_header('2. VALIDACIÓN DEL SARIMAX-ICG (backtesting expanding 24m)', '-')
    print(f"SARIMAX-ICG  ARIMA{CONFIG['sarimax_icg_order']}:")
    print(f"  MAE   : {res.backtesting_icg['mae']:.4f}")
    print(f"  Naïve : {res.backtesting_icg['mae_naive']:.4f}")
    print(f"  Skill : {res.backtesting_icg['skill']:+.1f}%")
    print()
    print(f"Nota: el sistema es univariado. El IPC fue removido por decisión")
    print(f"      documentada (ver docstring). El modelo electoral XGB v7 no")
    print(f"      usa el IPC; se le pasa un valor placebo constante de")
    print(f"      {fmt_pct(res.ipc_placebo, 2)} (mediana histórica) solo para satisfacer")
    print(f"      su firma bivariada.")

    print_header('3. DISTRIBUCIÓN SIMULADA DEL ICG (ventana electoral)', '-')
    icg = res.estadisticas['icg_ventana']
    print(f"ICG promedio 6m (abr-sep 2027):")
    print(f"  Mediana : {icg['mediana']:.3f}")
    print(f"  Media   : {icg['media']:.3f}")
    print(f"  Std     : {icg['std']:.3f}")
    print(f"  IC 90%  : [{icg['p5']:.3f}, {icg['p95']:.3f}]")

    print_header('4. DISTRIBUCIÓN DEL PRONÓSTICO ELECTORAL 2027', '-')
    sr = res.estadisticas['pred_sin_ruido']
    tot = res.estadisticas['pred_total']
    print(f"A) Solo incertidumbre del ICG:")
    print(f"   Mediana : {fmt_pct(sr['mediana'])}   IC 90%: "
          f"[{fmt_pct(sr['p5'])}, {fmt_pct(sr['p95'])}]")
    print()
    print(f"B) Incertidumbre TOTAL (ICG + error del modelo electoral):")
    print(f"   Media   : {fmt_pct(tot['media'])}")
    print(f"   Mediana : {fmt_pct(tot['mediana'])}")
    print(f"   Std     : {tot['std']*100:.1f} pp")
    print(f"   P5      : {fmt_pct(tot['p5'])}")
    print(f"   P25     : {fmt_pct(tot['p25'])}")
    print(f"   P50     : {fmt_pct(tot['p50'])}")
    print(f"   P75     : {fmt_pct(tot['p75'])}")
    print(f"   P95     : {fmt_pct(tot['p95'])}")

    print_header('5. PROBABILIDADES DE EVENTOS ELECTORALES', '-')
    p = res.probabilidades
    print(f"P(oficialismo ≥ 45%  — gana 1ra vuelta cómodo) : {fmt_pct(p['p_gana_1ra'])}")
    print(f"P(oficialismo 40-45% — competitivo)            : {fmt_pct(p['p_competitivo'])}")
    print(f"P(oficialismo 35-40% — zona de riesgo)         : {fmt_pct(p['p_zona_riesgo'])}")
    print(f"P(oficialismo < 35%  — derrota probable)       : {fmt_pct(p['p_derrota'])}")
    print()
    print(f"P(oficialismo > 40%) : {fmt_pct(p['p_mayor_40'])}")
    print(f"P(oficialismo > 35%) : {fmt_pct(p['p_mayor_35'])}")

    print_header('6. ANÁLISIS DE SENSIBILIDAD AL ICG', '-')
    print("Pregunta condicional: SI el ICG promedio 6m de la ventana")
    print("electoral (abr-sep 2027) terminara siendo un valor FIJO conocido,")
    print("¿qué pronóstico implica? (incertidumbre = solo error del modelo")
    print("electoral; NO incluye incertidumbre sobre el ICG, que aquí es dato)")
    print()
    s = res.sensibilidad_icg
    print(f"{'ICG 6m':>7}{'Pronóst.':>10}{'IC90%':>17}"
          f"{'P(gana)':>9}{'P(comp)':>9}{'P(riesgo)':>11}{'P(derrota)':>12}")
    print('-'*76)
    for _, r in s.iterrows():
        ic = f"[{r['p5']*100:.1f}, {r['p95']*100:.1f}]"
        print(f"{r['icg_6m']:>7.2f}{r['pronostico']*100:>9.1f}%{ic:>17}"
              f"{r['p_gana_1ra']*100:>8.0f}%{r['p_competitivo']*100:>8.0f}%"
              f"{r['p_zona_riesgo']*100:>10.0f}%{r['p_derrota']*100:>11.0f}%")
    print('-'*76)
    # Identificar el ICG umbral para ganar en 1ra vuelta
    gana = s[s['pronostico'] >= CONFIG['umbral_gana_1ra']]
    if len(gana) > 0:
        icg_umbral = gana.iloc[0]['icg_6m']
        print(f"→ El oficialismo alcanza el umbral de 1ra vuelta ({CONFIG['umbral_gana_1ra']*100:.0f}%)")
        print(f"  con ICG promedio 6m ≥ {icg_umbral:.2f}")
    derrota = s[s['pronostico'] < CONFIG['umbral_riesgo']]
    if len(derrota) > 0:
        icg_derrota = derrota.iloc[-1]['icg_6m']
        print(f"→ El oficialismo cae en zona de derrota (<{CONFIG['umbral_riesgo']*100:.0f}%)")
        print(f"  con ICG promedio 6m ≤ {icg_derrota:.2f}")
    print()
    print("Uso para monitoreo: a medida que avance 2026-2027 y se observe el")
    print("ICG real, ubicar el valor en esta tabla da el pronóstico implícito")
    print("sin necesidad de re-correr el Monte Carlo completo.")

    print_header('7. LECTURA ESTRATÉGICA', '-')
    tot = res.estadisticas['pred_total']
    p = res.probabilidades
    msg = [
        f"Pronóstico central 2027: {fmt_pct(tot['mediana'])} "
        f"(rango intercuartil {fmt_pct(tot['p25'])}–{fmt_pct(tot['p75'])})",
        "",
        "Lo que dice el sistema:",
        f"  • El pronóstico NO es un número, es una distribución. La mediana",
        f"    de {fmt_pct(tot['mediana'])} viene con un IC 90% de ~{(tot['p95']-tot['p5'])*100:.0f} puntos de ancho.",
        f"  • La probabilidad de ganar en 1ra vuelta ({fmt_pct(p['p_gana_1ra'])}) y la de",
        f"    derrota ({fmt_pct(p['p_derrota'])}) son comparables: el resultado es",
        f"    genuinamente abierto a {res.horizonte['h_total']} meses vista.",
        f"  • Toda la incertidumbre proyectable proviene de la trayectoria",
        f"    del ICG. No es una limitación del sistema: es la realidad de",
        f"    que no sabemos cómo va a evolucionar la confianza en el",
        f"    gobierno a {res.horizonte['h_total']} meses.",
        "",
        "Implicancias para gestión de activos:",
        f"  • NO hay base para un trade direccional fuerte sobre el resultado",
        f"    electoral hoy. El sistema asigna probabilidad relevante a los",
        f"    cuatro escenarios.",
        f"  • El valor del sistema es el MONITOREO: a medida que se acerque",
        f"    2027 y entren datos de ICG, la distribución se va a estrechar.",
        f"    El momento de tomar exposición es cuando P(evento) supere un",
        f"    umbral de convicción (ej: P(gana 1ra) > 60%).",
        f"  • Re-correr este pipeline mensualmente. Cada dato nuevo de ICG",
        f"    actualiza toda la distribución.",
        "",
        "Limitaciones:",
        f"  • SARIMAX-ICG asume continuidad de la dinámica histórica. Un",
        f"    shock estructural (crisis, evento político mayor) no está",
        f"    modelado y rompería la proyección.",
        f"  • A {res.horizonte['h_total']} meses, la proyección del ICG es básicamente",
        f"    reversión a la media con bandas anchas. Es honesto pero poco",
        f"    informativo: el sistema gana precisión recién en 2027.",
        f"  • Cuando se publiquen encuestas de intención de voto para 2027,",
        f"    deberían combinarse con este sistema (no reemplazarlo).",
    ]
    print('\n'.join(msg))


# ============================================================================
# EXPORTACIÓN
# ============================================================================

def exportar(res: ResultadoSimulacion, path: str = 'sistema_integrado_outputs.xlsx'):
    percentiles = list(range(1, 100))
    dist_df = pd.DataFrame({
        'percentil'      : percentiles,
        'pred_total'     : [np.percentile(res.pred_total, q) for q in percentiles],
        'pred_sin_ruido' : [np.percentile(res.pred_sin_ruido, q) for q in percentiles],
        'icg_ventana'    : [np.percentile(res.icg_ventana, q) for q in percentiles],
    })

    resumen_df = pd.DataFrame({
        'metrica': [
            'n_simulaciones', 'horizonte_meses', 'version',
            'SARIMAX-ICG MAE backtest', 'SARIMAX-ICG skill (%)',
            'IPC placebo (no usado por el modelo)',
            'ICG ventana — mediana', 'ICG ventana — P5', 'ICG ventana — P95',
            'Pred electoral — media (%)', 'Pred electoral — mediana (%)',
            'Pred electoral — std (pp)',
            'Pred electoral — P5 (%)', 'Pred electoral — P25 (%)',
            'Pred electoral — P50 (%)', 'Pred electoral — P75 (%)',
            'Pred electoral — P95 (%)',
            'P(gana 1ra vuelta)', 'P(competitivo)',
            'P(zona riesgo)', 'P(derrota)',
        ],
        'valor': [
            CONFIG['n_simulaciones'], res.horizonte['h_total'], 'univariada ICG',
            res.backtesting_icg['mae'], res.backtesting_icg['skill'],
            res.ipc_placebo,
            res.estadisticas['icg_ventana']['mediana'],
            res.estadisticas['icg_ventana']['p5'],
            res.estadisticas['icg_ventana']['p95'],
            res.estadisticas['pred_total']['media']*100,
            res.estadisticas['pred_total']['mediana']*100,
            res.estadisticas['pred_total']['std']*100,
            res.estadisticas['pred_total']['p5']*100,
            res.estadisticas['pred_total']['p25']*100,
            res.estadisticas['pred_total']['p50']*100,
            res.estadisticas['pred_total']['p75']*100,
            res.estadisticas['pred_total']['p95']*100,
            res.probabilidades['p_gana_1ra'],
            res.probabilidades['p_competitivo'],
            res.probabilidades['p_zona_riesgo'],
            res.probabilidades['p_derrota'],
        ],
    })

    sim_df = pd.DataFrame({
        'icg_ventana'   : res.icg_ventana[:1000],
        'pred_sin_ruido': res.pred_sin_ruido[:1000],
        'pred_total'    : res.pred_total[:1000],
    })

    with pd.ExcelWriter(path, engine='openpyxl') as w:
        resumen_df.to_excel(           w, sheet_name='resumen',           index=False)
        dist_df.to_excel(              w, sheet_name='distribucion',      index=False)
        res.sensibilidad_icg.to_excel( w, sheet_name='sensibilidad_icg',  index=False)
        sim_df.to_excel(               w, sheet_name='simulaciones_1000', index=False)

    print(f"\n→ Resultados exportados a: {path}")


# ============================================================================
# PIPELINE
# ============================================================================

def main():
    resultado = simular()
    reporte(resultado)
    exportar(resultado)
    return resultado


if __name__ == '__main__':
    main()
