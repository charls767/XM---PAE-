"""Carga de configuración y rutas del proyecto."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


@lru_cache(maxsize=1)
def load_config(path: str | None = None) -> dict[str, Any]:
    """Lee config/config.yaml (o la ruta en la variable GUAVIO_CONFIG)."""
    cfg_path = Path(path or os.getenv("GUAVIO_CONFIG", ROOT / "config" / "config.yaml"))
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    with open(cfg_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def data_source() -> str:
    """Fuente de datos activa: variable GUAVIO_DATA_SOURCE o data.source de la configuración."""
    src = os.getenv("GUAVIO_DATA_SOURCE") or load_config().get("data", {}).get("source", "synthetic")
    src = src.strip().lower()
    if src not in ("xm", "synthetic", "xm_mock"):
        raise ValueError(f"Fuente de datos desconocida: {src!r} (use 'xm' o 'synthetic')")
    return src


def is_xm_source(source: str | None = None) -> bool:
    """True para la API real de XM y para su simulador de pruebas (xm_mock)."""
    return (source or data_source()).startswith("xm")


def xm_cache_dir() -> Path:
    """Caché de la API de XM y NOAA, separada por fuente (el simulador nunca toca la caché real)."""
    base = load_config().get("xm", {}).get("cache_dir", "data/external/xm_cache")
    p = ROOT / (base if data_source() == "xm" else f"{base}_{data_source()}")
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_data_source(source: str) -> None:
    """Fija la fuente para el proceso actual (y los subprocesos)."""
    os.environ["GUAVIO_DATA_SOURCE"] = source
    data_source()


def path_of(key: str) -> Path:
    """Ruta absoluta de una entrada de `paths` en la configuración."""
    p = ROOT / load_config()["paths"][key].format(source=data_source())
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def models_dir() -> Path:
    p = ROOT / load_config()["paths"]["models_dir"].format(source=data_source())
    p.mkdir(parents=True, exist_ok=True)
    return p


def figures_dir() -> Path:
    p = ROOT / load_config()["paths"]["figures_dir"].format(source=data_source())
    p.mkdir(parents=True, exist_ok=True)
    return p


def metrics_dir() -> Path:
    p = ROOT / load_config()["paths"]["metrics_dir"].format(source=data_source())
    p.mkdir(parents=True, exist_ok=True)
    return p
