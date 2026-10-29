"""Índice Niño 3.4 de NOAA/CPC (anomalía semanal de temperatura superficial del mar).

Fuente: https://www.cpc.ncep.noaa.gov/data/indices/wksst9120.for (semanal, base 1991-2020).
Respaldo: https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt (ONI, trimestral móvil).

La serie se lleva a resolución diaria con relleno hacia adelante (cada valor se usa
solo desde su semana de publicación), para no introducir información futura.
"""
from __future__ import annotations

import re

import pandas as pd

from src.config import xm_cache_dir
from src.utils.logging_utils import get_logger

log = get_logger(__name__)

WEEKLY_URL = "https://www.cpc.ncep.noaa.gov/data/indices/wksst9120.for"
ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
_LINE = re.compile(r"^\s*(\d{2}[A-Z]{3}\d{4})\s+(.*)$")
_NUM = re.compile(r"-?\d+\.\d")
_SEASON_MID_MONTH = {"DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6, "JJA": 7, "JAS": 8,
                     "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12}


def parse_weekly(text: str) -> pd.Series:
    """Parsea el archivo semanal de CPC. Devuelve la anomalía Niño 3.4 indexada por fecha."""
    dates, values = [], []
    for line in text.splitlines():
        m = _LINE.match(line)
        if not m:
            continue
        nums = _NUM.findall(m.group(2))
        if len(nums) < 8:
            continue
        # Orden: Niño1+2 (SST, SSTA), Niño3 (SST, SSTA), Niño3.4 (SST, SSTA), Niño4 (SST, SSTA)
        dates.append(pd.to_datetime(m.group(1), format="%d%b%Y"))
        values.append(float(nums[5]))
    return pd.Series(values, index=pd.DatetimeIndex(dates), name="nino34").sort_index()


def parse_oni(text: str) -> pd.Series:
    """Parsea oni.ascii.txt (SEAS YR TOTAL ANOM) y fecha cada valor en el mes central."""
    dates, values = [], []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[0] in _SEASON_MID_MONTH:
            dates.append(pd.Timestamp(int(parts[1]), _SEASON_MID_MONTH[parts[0]], 15))
            values.append(float(parts[3]))
    return pd.Series(values, index=pd.DatetimeIndex(dates), name="nino34").sort_index()


def _download(url: str, timeout: float = 60.0) -> str:
    import requests

    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "guavio-xai/1.0"})
    resp.raise_for_status()
    return resp.text


def load_nino34(start, end, force: bool = False) -> pd.Series:
    """Serie diaria del Niño 3.4 entre start y end (relleno hacia adelante)."""
    cache = xm_cache_dir() / "noaa_nino34.csv"
    fresh = cache.exists() and (pd.Timestamp.now() - pd.Timestamp(cache.stat().st_mtime, unit="s")).days < 3
    if fresh and not force:
        s = pd.read_csv(cache, index_col=0, parse_dates=True).iloc[:, 0]
    else:
        try:
            s = parse_weekly(_download(WEEKLY_URL))
            if s.empty:
                raise ValueError("archivo semanal vacío")
        except Exception as exc:
            log.warning("No se pudo leer el Niño 3.4 semanal (%s); se usa el ONI mensual.", exc)
            try:
                s = parse_oni(_download(ONI_URL))
            except Exception as exc2:
                if cache.exists():
                    log.warning("NOAA no disponible (%s); se usa la copia en caché.", exc2)
                    s = pd.read_csv(cache, index_col=0, parse_dates=True).iloc[:, 0]
                else:
                    raise RuntimeError(f"No fue posible descargar el índice Niño 3.4 de NOAA: {exc2}") from exc2
        s.to_frame("nino34").to_csv(cache)
    idx = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="D")
    daily = s.reindex(s.index.union(idx)).sort_index().ffill().reindex(idx)
    return daily.bfill(limit=10).rename("nino34")
