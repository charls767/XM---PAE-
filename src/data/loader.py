"""Carga de datos de la Central Guavio: API de XM (Sinergox) + NOAA, o generador sintético.

La fuente se elige en config/config.yaml (data.source), con la variable GUAVIO_DATA_SOURCE
o con --source. La fuente "xm" está implementada en src/data/xm_loader.py.

Sobre el generador sintético:

Los datos son SINTÉTICOS: reproducen el orden de magnitud y las relaciones físicas
y de mercado esperadas (clima -> lluvia -> aportes -> embalse -> generación -> oferta,
y carga -> degradación -> vibración/temperatura -> fallas), pero no son datos
oficiales de XM ni de Enel Colombia. Sirven para desarrollar y probar el pipeline
antes de conectarlo a las fuentes públicas (XM/SIMEM, IDEAM, NOAA).

Uso:
    python -m src.data.loader --source xm --force          # descarga/actualiza desde la API de XM
    python -m src.data.loader --source synthetic --force   # regenera los datos sintéticos
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import data_source, is_xm_source, load_config, path_of, set_data_source
from src.utils.logging_utils import get_logger

log = get_logger(__name__)

# Anclas mensuales aproximadas del índice Niño 3.4 (°C). Siguen la secuencia
# histórica de episodios ENSO de forma aproximada y terminan con El Niño 2026
# en fortalecimiento. Son insumo del generador, no una serie oficial.
NINO34_ANCHORS = {
    "2016-01-01": 2.5, "2016-04-01": 1.3, "2016-07-01": -0.3, "2016-10-01": -0.7,
    "2017-01-01": -0.4, "2017-05-01": 0.3, "2017-10-01": -0.6, "2018-01-01": -0.9,
    "2018-05-01": -0.2, "2018-10-01": 0.8, "2019-02-01": 0.8, "2019-06-01": 0.5,
    "2019-10-01": 0.3, "2020-02-01": 0.5, "2020-06-01": -0.2, "2020-11-01": -1.3,
    "2021-03-01": -0.8, "2021-07-01": -0.4, "2021-12-01": -1.0, "2022-04-01": -1.1,
    "2022-09-01": -1.0, "2023-01-01": -0.7, "2023-05-01": 0.5, "2023-09-01": 1.6,
    "2023-12-01": 2.0, "2024-03-01": 1.1, "2024-06-01": 0.2, "2024-11-01": -0.4,
    "2025-02-01": -0.6, "2025-06-01": -0.1, "2025-10-01": -0.5, "2026-01-01": -0.3,
    "2026-04-01": 0.4, "2026-06-01": 1.0, "2026-08-01": 1.8, "2026-09-30": 2.3,
}

# Nivel de referencia del precio de bolsa (COP/kWh) por año, para dar tendencia.
PRICE_BASE_ANCHORS = {
    "2016-01-01": 260.0, "2017-01-01": 130.0, "2018-01-01": 120.0, "2019-01-01": 190.0,
    "2020-01-01": 210.0, "2021-01-01": 200.0, "2022-01-01": 260.0, "2023-01-01": 420.0,
    "2024-01-01": 480.0, "2025-01-01": 330.0, "2026-01-01": 300.0, "2026-09-30": 320.0,
}

# Episodio seco adicional (sequía de 2024 que llevó al racionamiento de agua en Bogotá).
DRY_EPISODES = [("2024-07-15", "2024-10-31", 0.55)]

# Constantes físicas para convertir energía de aportes a caudal equivalente.
HEAD_M = 1100.0          # caída neta aproximada (m), orden de magnitud de Guavio
EFFICIENCY = 0.90
GWH_DAY_PER_M3S = 1000 * 9.81 * HEAD_M * EFFICIENCY * 86400 / 3.6e12


@dataclass
class PlantState:
    volume_gwh: float
    degradation: float = 0.05
    days_since_maintenance: int = 0
    outage_days_left: int = 0
    maintenance_days_left: int = 0
    next_maintenance_in: int = 180
    log_bid_noise: float = 0.0
    wear_rate: float = 1.0


def _interp_daily(anchors: dict[str, float], index: pd.DatetimeIndex) -> np.ndarray:
    s = pd.Series(anchors)
    s.index = pd.to_datetime(s.index)
    s = s.reindex(s.index.union(index)).interpolate(method="time").ffill().bfill()
    return s.reindex(index).to_numpy()


def _ar1(n: int, phi: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    out = np.zeros(n)
    eps = rng.normal(0, sigma, n)
    for i in range(1, n):
        out[i] = phi * out[i - 1] + eps[i]
    return out


def _sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_synthetic_data(cfg: dict | None = None) -> pd.DataFrame:
    """Genera la serie diaria sintética de la central y su contexto de mercado."""
    cfg = cfg or load_config()
    sc = cfg["synthetic"]
    rng = np.random.default_rng(sc["seed"])
    idx = pd.date_range(sc["start_date"], sc["end_date"], freq="D")
    n = len(idx)
    doy = idx.dayofyear.to_numpy()
    years = (idx - idx[0]).days.to_numpy() / 365.25

    # --- Clima -------------------------------------------------------------
    nino = _interp_daily(NINO34_ANCHORS, idx) + _ar1(n, 0.97, 0.03, rng)
    nino_lag60 = pd.Series(nino).shift(60).bfill().to_numpy()
    # Régimen monomodal (vertiente oriental): seco dic-mar, húmedo may-ago.
    clim_precip = np.clip(6.0 + 5.0 * np.sin(2 * np.pi * (doy - 100) / 365.25), 0.6, None)
    enso_factor = np.exp(-0.20 * nino_lag60)
    dry = np.ones(n)
    for start, end, factor in DRY_EPISODES:
        dry[(idx >= start) & (idx <= end)] = factor
    precip = clim_precip * enso_factor * dry * rng.gamma(shape=1.4, scale=1 / 1.4, size=n)
    precip[rng.random(n) < 0.18] = 0.0
    temp_amb = 18.0 - 1.2 * np.sin(2 * np.pi * (doy - 100) / 365.25) + 0.45 * nino + rng.normal(0, 0.6, n)

    # --- Hidrología: aportes por embalse lineal (rápido + base) ----------------
    p = pd.Series(precip)
    fast = p.ewm(span=5, adjust=False).mean().to_numpy()
    slow = p.ewm(span=60, adjust=False).mean().to_numpy()
    raw = 0.35 * fast + 0.65 * slow
    aportes = raw * sc["mean_inflow_gwh_day"] / raw.mean() * rng.lognormal(0, 0.08, n)
    aportes = np.clip(aportes, 0.4, None)
    clim_aportes = pd.Series(aportes).groupby(doy).transform("mean").rolling(15, center=True, min_periods=1).mean().to_numpy()
    aportes_pct = 100 * aportes / clim_aportes
    hydro30 = pd.Series(aportes / clim_aportes).rolling(30, min_periods=1).mean().to_numpy()

    # --- Mercado (exógeno a la central) --------------------------------------
    weekday = idx.dayofweek.to_numpy()
    wd_factor = np.where(weekday == 6, 0.86, np.where(weekday == 5, 0.94, 1.0))
    demanda = 185.0 * (1.024 ** years) * wd_factor * (1 + 0.02 * np.sin(2 * np.pi * (doy - 30) / 365.25))
    demanda = demanda * rng.normal(1, 0.012, n)
    price_base = _interp_daily(PRICE_BASE_ANCHORS, idx)
    precio_bolsa = price_base * np.exp(-1.1 * (hydro30 - 1)) * np.exp(0.12 * nino) * np.exp(_ar1(n, 0.9, 0.06, rng))
    oct24 = (idx >= "2024-09-20") & (idx <= "2024-10-31")
    precio_bolsa[oct24] *= 1.9
    precio_bolsa = np.clip(precio_bolsa, 80, 1600)
    frac_minimo = np.clip(_sigmoid(-1.0 + 4.0 * (hydro30 - 1)) + rng.normal(0, 0.03, n), 0, 1)
    month_index = (idx.year - idx[0].year) * 12 + idx.month - 1
    precio_minimo = sc["min_bid_2016_cop_kwh"] * (1 + sc["min_bid_annual_growth"]) ** (month_index.to_numpy() / 12)

    # Expectativa de aportes futuros del agente (perfecta + ruido): el agente
    # conoce su cuenca mejor que el dato publicado. Esto hace que el pronóstico
    # de aportes tenga valor para explicar la oferta.
    fut14 = pd.Series(aportes).rolling(14, min_periods=1).mean().shift(-14).ffill().to_numpy()
    clim14 = pd.Series(clim_aportes).rolling(14, min_periods=1).mean().shift(-14).ffill().to_numpy()
    expected_ratio = (fut14 / clim14) * rng.lognormal(0, 0.10, n)

    # --- Simulación día a día de la central -----------------------------------
    cap_mw = sc["installed_capacity_mw"]
    unit_mw = cap_mw / sc["n_units"]
    vcap = sc["useful_capacity_gwh"]
    state = PlantState(volume_gwh=sc["initial_volume_pct"] * vcap, next_maintenance_in=sc["maintenance_interval_days"])
    month_regime = rng.normal(0, 0.12, month_index.max() + 1)
    # Límite de despacho por restricciones de red (fracción de la disponibilidad).
    dispatch_cap = np.clip(0.50 + _ar1(n, 0.95, 0.03, rng), 0.30, 0.85)

    rec = {k: np.zeros(n) for k in (
        "volumen_util_gwh", "vertimientos_gwh", "generacion_gwh", "disponibilidad_mw",
        "unidades_disponibles", "precio_oferta_cop_kwh", "oferta_al_minimo", "vibracion_mm_s",
        "temp_cojinete_c", "indice_degradacion", "horas_desde_mant", "mantenimiento_programado",
        "falla_forzada")}

    for i in range(n):
        v_pct = state.volume_gwh / vcap
        # Disponibilidad por mantenimiento programado o falla forzada.
        units_out = 0
        maint_today = 0
        if state.maintenance_days_left > 0:
            units_out += 1
            maint_today = 1
            state.maintenance_days_left -= 1
        if state.outage_days_left > 0:
            units_out += 1
            state.outage_days_left -= 1
        units = sc["n_units"] - units_out
        avail_mw = units * unit_mw

        # Oferta del agente: valor del agua según embalse, aportes esperados y mercado.
        logit_floor = -0.6 + 9.0 * (v_pct - 0.80) + 3.0 * (expected_ratio[i] - 1.0) + 1.0 * (frac_minimo[i] - 0.5)
        is_floor = rng.random() < _sigmoid(logit_floor)
        state.log_bid_noise = 0.75 * state.log_bid_noise + rng.normal(0, 0.08)
        log_y = (0.55 + 2.4 * (0.75 - v_pct) + 1.6 * (1.0 - expected_ratio[i]) + 0.18 * nino[i]
                 + 0.55 * np.log(precio_bolsa[i - 1 if i else 0] / price_base[i]) + month_regime[month_index[i]]
                 + state.log_bid_noise)
        y_norm = 1.0 if is_floor else 1.05 + np.exp(np.clip(log_y, -3, 2.3))
        bid = y_norm * precio_minimo[i]

        # Generación: despacho por mérito simplificado.
        merit = 0.6 if bid < precio_bolsa[i] else -0.6
        f = _sigmoid(-1.1 + 1.6 * (v_pct - 0.60) + merit + (0.9 if is_floor else 0.0))
        gen = (0.12 + 0.78 * f) * avail_mw * 24 / 1000 * rng.normal(1, 0.04)
        gen = min(gen, dispatch_cap[i] * avail_mw * 24 / 1000)
        gen = max(0.0, min(gen, state.volume_gwh + aportes[i] - 0.03 * vcap))
        new_v = state.volume_gwh + aportes[i] - gen
        spill = 0.0
        if new_v > vcap:
            spill = new_v - vcap
            new_v = vcap
        state.volume_gwh = max(new_v, 0.0)

        # Condición de la máquina: degradación, vibración, temperatura, fallas.
        load = gen / (cap_mw * 24 / 1000)
        state.degradation = min(1.0, state.degradation + state.wear_rate * 0.0024 * (0.4 + load) + abs(rng.normal(0, 0.0010)))
        vib = 1.8 + 3.6 * state.degradation ** 1.5 + 0.7 * load + rng.normal(0, 0.15)
        temp_bear = 46 + 13 * load + 11 * state.degradation ** 1.3 + 0.35 * (temp_amb[i] - 18) + rng.normal(0, 0.8)
        hazard = 0.0003 + 3.0 * state.degradation ** 5
        failure = state.outage_days_left == 0 and rng.random() < hazard
        if failure:
            state.outage_days_left = int(rng.integers(2, 7))
            state.degradation = 0.35
        state.days_since_maintenance += 1
        state.next_maintenance_in -= 1
        if state.next_maintenance_in <= 0 and state.maintenance_days_left == 0:
            state.maintenance_days_left = sc["maintenance_duration_days"]
            state.degradation = 0.05
            state.wear_rate = float(rng.lognormal(0, 0.35))
            state.days_since_maintenance = 0
            state.next_maintenance_in = sc["maintenance_interval_days"] + int(rng.integers(-10, 11))

        rec["volumen_util_gwh"][i] = state.volume_gwh
        rec["vertimientos_gwh"][i] = spill
        rec["generacion_gwh"][i] = gen
        rec["disponibilidad_mw"][i] = avail_mw
        rec["unidades_disponibles"][i] = units
        rec["precio_oferta_cop_kwh"][i] = bid
        rec["oferta_al_minimo"][i] = int(is_floor)
        rec["vibracion_mm_s"][i] = vib
        rec["temp_cojinete_c"][i] = temp_bear
        rec["indice_degradacion"][i] = state.degradation
        rec["horas_desde_mant"][i] = state.days_since_maintenance * 24
        rec["mantenimiento_programado"][i] = maint_today
        rec["falla_forzada"][i] = int(failure)

    df = pd.DataFrame({
        "fecha": idx,
        "nino34": nino,
        "precipitacion_mm": precip,
        "temp_ambiente_c": temp_amb,
        "aportes_gwh": aportes,
        "aportes_pct_media": aportes_pct,
        "caudal_m3s": aportes / GWH_DAY_PER_M3S,
        **rec,
        "demanda_sin_gwh": demanda,
        "precio_bolsa_cop_kwh": precio_bolsa,
        "precio_minimo_cop_kwh": precio_minimo,
        "fraccion_plantas_minimo": frac_minimo,
    })
    df["volumen_util_pct"] = 100 * df["volumen_util_gwh"] / vcap
    df["nivel_embalse_msnm"] = 1540 + 90 * np.sqrt(df["volumen_util_gwh"] / vcap)
    df["generacion_mw_media"] = df["generacion_gwh"] * 1000 / 24
    df["capacidad_util_gwh"] = float(vcap)
    for col in ("oferta_al_minimo", "unidades_disponibles", "mantenimiento_programado", "falla_forzada"):
        df[col] = df[col].astype(int)
    df["es_sintetico"] = True
    return df


def load_data(force: bool = False, source: str | None = None) -> pd.DataFrame:
    """Carga los datos crudos de la fuente activa; los descarga/genera si no existen o si force=True."""
    if source:
        set_data_source(source)
    src = data_source()
    path = path_of("raw_data")
    if path.exists() and not force:
        return pd.read_parquet(path)
    if is_xm_source(src):
        from src.data.xm_loader import build_xm_dataset

        log.info("Descargando datos de la API de XM (Sinergox) y NOAA...")
        df = build_xm_dataset()
    else:
        log.info("Generando datos sintéticos de Guavio...")
        df = generate_synthetic_data()
    df.to_parquet(path, index=False)
    log.info("Guardado %s (%d filas, %s a %s)", path, len(df), df.fecha.min().date(), df.fecha.max().date())
    return df


def has_machine_data(df: pd.DataFrame) -> bool:
    """True si el dataset trae variables de condición de máquina (solo la fuente sintética)."""
    return {"vibracion_mm_s", "temp_cojinete_c", "falla_forzada"}.issubset(df.columns)


def main() -> None:
    parser = argparse.ArgumentParser(description="Carga los datos de la Central Guavio (XM o sintéticos).")
    parser.add_argument("--source", choices=["xm", "synthetic"], help="Fuente de datos (por defecto, la de config).")
    parser.add_argument("--force", action="store_true", help="Vuelve a descargar/generar aunque el archivo exista.")
    args = parser.parse_args()
    df = load_data(force=args.force, source=args.source)
    print(df.describe().T[["mean", "min", "max"]].round(2).to_string())


if __name__ == "__main__":
    main()
