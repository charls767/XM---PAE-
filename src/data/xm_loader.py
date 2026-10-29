"""Construye la serie diaria real de la Central Guavio desde la API de XM (Sinergox) y NOAA.

Produce las mismas columnas que el generador sintético para hidrología y mercado, de modo que
el resto del pipeline no cambia. No incluye variables de condición de máquina (vibración,
temperatura de cojinetes): no son públicas, por lo que el modelo de fallas se omite con esta fuente.

Columnas producidas (unidades):
    fecha, nino34 (°C), aportes_gwh, aportes_pct_media (%), caudal_m3s, volumen_util_gwh,
    volumen_util_pct (%), capacidad_util_gwh, vertimientos_gwh, generacion_gwh, generacion_mw_media,
    disponibilidad_mw, unidades_disponibles, precio_oferta_cop_kwh, precio_minimo_cop_kwh,
    oferta_al_minimo, precio_bolsa_cop_kwh, demanda_sin_gwh, fraccion_plantas_minimo, es_sintetico
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import ROOT, load_config
from src.data.noaa import load_nino34
from src.data.xm_client import XMClient, resolve_plant_codes
from src.utils.logging_utils import get_logger

log = get_logger(__name__)
KWH_TO_GWH = 1e-6
KW_TO_MW = 1e-3


def _date_range() -> tuple[pd.Timestamp, pd.Timestamp]:
    cfg = load_config()["xm"]
    start = pd.Timestamp(cfg["start_date"])
    end = cfg.get("end_date", "auto")
    end = pd.Timestamp.today().normalize() - pd.Timedelta(days=1) if end in (None, "auto") else pd.Timestamp(end)
    return start, end


def _series(client: XMClient, key: str, start, end, code: str | None) -> pd.Series:
    spec = load_config()["xm"]["metrics"][key]
    # Recurso: filtro del lado del servidor (código del recurso). Río y embalse: se piden sin filtro y se
    # selecciona el código detectado en los datos, porque XM no usa el mismo código en listados y series.
    filters = [code] if code and spec["entity"] == "Recurso" else []
    s = client.fetch_daily_series(spec["id"], spec["entity"], start, end, filters=filters,
                                  period=spec.get("period", "daily"), how=spec.get("how", "sum"),
                                  code=code if spec["entity"] != "Sistema" else None)
    if s.empty and key not in load_config()["xm"].get("optional", []):
        raise RuntimeError(f"XM no devolvió datos para {spec['id']}/{spec['entity']} ({code}). "
                           "Ejecute 'python -m src.data.xm_client --check' para revisar métricas y códigos.")
    if s.empty:
        log.warning("Variable opcional sin datos en XM: %s (%s)", key, spec["id"])
    return s


def offers_all_resources(client: XMClient, start, end) -> pd.DataFrame:
    """Precio de oferta diario de todos los recursos del SIN (mediana de las 24 horas)."""
    spec = load_config()["xm"]["metrics"]["precio_oferta"]
    df = client.fetch(spec["id"], spec["entity"], start, end, filters=[], period=spec.get("period", "hourly"))
    if df.empty:
        return df
    if "hora" in df:
        df = df.groupby(["fecha", "codigo"])["valor"].median().reset_index()
    return df


def price_floor(offers: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """Precio mínimo de oferta por día.

    proxy: mediana mensual del menor precio positivo ofertado en el SIN cada día (el piso
           regulatorio es mensual, así que se toma un valor por mes).
    csv:   valores mensuales cargados por el usuario (columnas mes, precio_cop_kwh).
    """
    pf = load_config()["xm"]["price_floor"]
    months = index.to_period("M")
    if pf.get("source", "proxy") == "csv":
        tab = pd.read_csv(ROOT / pf["csv_path"])
        m = tab.set_index(pd.PeriodIndex(tab["mes"], freq="M"))["precio_cop_kwh"]
        out = pd.Series(months.map(m), index=index, dtype=float)
    else:
        valid = offers[offers["valor"] >= pf.get("min_valid_cop_kwh", 1.0)]
        daily_min = valid.groupby("fecha")["valor"].min()
        monthly = daily_min.groupby(daily_min.index.to_period("M")).median()
        out = pd.Series(months.map(monthly), index=index, dtype=float)
    # Meses sin dato (p. ej. el mes en curso): se mantiene el último valor conocido.
    return out.ffill().bfill().rename("precio_minimo_cop_kwh")


def fraction_at_floor(offers: pd.DataFrame, floor: pd.Series, eps: float) -> pd.Series:
    """Fracción de recursos del SIN que ofertaron al precio mínimo cada día."""
    d = offers[offers["valor"] > 0].copy()
    d["piso"] = d["fecha"].map(floor)
    d["al_min"] = d["valor"] <= d["piso"] * (1 + eps)
    return d.groupby("fecha")["al_min"].mean().rename("fraccion_plantas_minimo")


def build_xm_dataset(client: XMClient | None = None, start=None, end=None) -> pd.DataFrame:
    """Descarga y arma la serie diaria de Guavio. Usa la caché local para los meses ya descargados."""
    cfg = load_config()
    xcfg = cfg["xm"]
    client = client or XMClient()
    s0, e0 = _date_range()
    start, end = pd.Timestamp(start or s0), pd.Timestamp(end or e0)
    codes = resolve_plant_codes(client, end=end)
    log.info("Construyendo dataset XM %s a %s para %s", start.date(), end.date(), codes)
    idx = pd.date_range(start, end, freq="D", name="fecha")

    def get(key: str, code_key: str | None) -> pd.Series:
        code = codes[code_key] if code_key else None
        return _series(client, key, start, end, code).reindex(idx)

    aportes = get("aportes", "rio") * KWH_TO_GWH
    aportes_media = get("aportes_media", "rio") * KWH_TO_GWH
    caudal = get("caudal", "rio")
    vol = get("volumen_util", "embalse") * KWH_TO_GWH
    cap = get("capacidad_util", "embalse") * KWH_TO_GWH
    vert = get("vertimientos", "embalse") * KWH_TO_GWH
    gen = get("generacion", "recurso") * KWH_TO_GWH
    disp = get("disponibilidad", "recurso") * KW_TO_MW
    bid = get("precio_oferta", "recurso")
    bolsa = get("precio_bolsa", None)
    demanda = get("demanda", None) * KWH_TO_GWH

    offers = offers_all_resources(client, start, end)
    floor = price_floor(offers, idx) if not offers.empty else pd.Series(np.nan, index=idx)
    eps = cfg["oferta_model"]["floor_epsilon"]
    frac = fraction_at_floor(offers, floor, eps).reindex(idx) if not offers.empty else pd.Series(np.nan, index=idx)
    nino = load_nino34(start, end).reindex(idx)

    plant = xcfg["plant"]
    df = pd.DataFrame(index=idx)
    df["nino34"] = nino
    df["aportes_gwh"] = aportes.interpolate(limit=3, limit_area="inside")
    df["aportes_pct_media"] = 100 * df["aportes_gwh"] / aportes_media if aportes_media.notna().any() else np.nan
    df["caudal_m3s"] = caudal
    df["capacidad_util_gwh"] = cap.ffill().bfill()
    df["volumen_util_gwh"] = vol.interpolate(limit=3, limit_area="inside")
    df["volumen_util_pct"] = 100 * df["volumen_util_gwh"] / df["capacidad_util_gwh"]
    df["vertimientos_gwh"] = vert.fillna(0.0) if vert.notna().any() else 0.0
    df["generacion_gwh"] = gen
    df["generacion_mw_media"] = gen * 1000 / 24
    df["disponibilidad_mw"] = disp.ffill() if disp.notna().any() else float(plant["installed_capacity_mw"])
    unit_mw = plant["installed_capacity_mw"] / plant["n_units"]
    df["unidades_disponibles"] = np.clip(np.round(df["disponibilidad_mw"] / unit_mw), 0, plant["n_units"])
    df["precio_oferta_cop_kwh"] = bid
    df["precio_minimo_cop_kwh"] = floor
    df["oferta_al_minimo"] = (bid <= floor * (1 + eps)).astype(float).where(bid.notna())
    df["precio_bolsa_cop_kwh"] = bolsa
    df["demanda_sin_gwh"] = demanda
    df["fraccion_plantas_minimo"] = frac
    df["es_sintetico"] = False
    df = df.reset_index()

    # Se recorta al último día con hidrología publicada (aportes y volumen útil).
    last = df.loc[df["aportes_gwh"].notna() & df["volumen_util_gwh"].notna(), "fecha"].max()
    if pd.isna(last):
        raise RuntimeError("XM no devolvió aportes ni volumen útil para Guavio en el rango solicitado.")
    df = df[df["fecha"] <= last].reset_index(drop=True)

    # Si falta la media histórica de XM, se calcula con los propios datos (mismo día del año).
    if df["aportes_pct_media"].isna().all():
        clim = df.groupby(df["fecha"].dt.dayofyear)["aportes_gwh"].transform("mean")
        df["aportes_pct_media"] = 100 * df["aportes_gwh"] / clim
    _sanity(df)
    return df


def _sanity(df: pd.DataFrame) -> None:
    """Advertencias sobre unidades o datos inconsistentes."""
    med = df["aportes_gwh"].median()
    if not 0.05 < med < 200:
        log.warning("Aportes medianos de %.3f GWh/día: revise la unidad de AporEner (se asume kWh).", med)
    if (df["volumen_util_pct"] > 105).mean() > 0.05:
        log.warning("Volumen útil > 105 %% en más del 5 %% de los días: revise VoluUtilDiarEner/CapaUtilDiarEner.")
    gmax = df["generacion_gwh"].max()
    cap_day = load_config()["xm"]["plant"]["installed_capacity_mw"] * 24 / 1000
    if gmax > 1.2 * cap_day:
        log.warning("Generación máxima %.1f GWh/día supera la capacidad (%.1f): revise la unidad de Gene.",
                    gmax, cap_day)
    n_bids = df["precio_oferta_cop_kwh"].notna().sum()
    last_bid = df.loc[df["precio_oferta_cop_kwh"].notna(), "fecha"].max()
    log.info("Dataset XM: %d días, %d con oferta publicada (última: %s), %.1f %% de días al mínimo.",
             len(df), n_bids, last_bid.date() if pd.notna(last_bid) else "ninguna",
             100 * df["oferta_al_minimo"].mean())
