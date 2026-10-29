"""Ejecuta el pipeline completo: datos -> entrenamiento -> explicabilidad.

Uso:
    python scripts/run_pipeline.py --source xm --force          # descarga/actualiza desde la API de XM
    python scripts/run_pipeline.py --source synthetic --force   # usa el simulador (sin internet)
    python scripts/run_pipeline.py                              # fuente de config, datos existentes
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import data_source, set_data_source  # noqa: E402
from src.data.loader import has_machine_data, load_data  # noqa: E402
from src.explainability.explainer import run_all  # noqa: E402
from src.models.train import train_aportes, train_falla, train_oferta  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402

log = get_logger("pipeline")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline completo del prototipo Guavio.")
    parser.add_argument("--source", choices=["xm", "synthetic"], help="Fuente de datos (por defecto, la de config).")
    parser.add_argument("--force", action="store_true", help="Vuelve a descargar/generar los datos.")
    parser.add_argument("--skip-shap", action="store_true", help="Omite las figuras SHAP.")
    args = parser.parse_args()
    t0 = time.time()
    if args.source:
        set_data_source(args.source)
    log.info("1/4 Datos (fuente: %s)", data_source())
    df = load_data(force=args.force)
    log.info("2/4 Modelo de aportes (incluye pronósticos walk-forward)")
    train_aportes(df)
    log.info("3/4 Modelo de oferta en dos etapas")
    train_oferta(df)
    if has_machine_data(df):
        train_falla(df)
    else:
        log.info("Modelo de fallas omitido: la fuente no trae datos de condición de máquina.")
    if not args.skip_shap:
        log.info("4/4 Explicabilidad (SHAP y LIME)")
        run_all()
    log.info("Pipeline terminado en %.0f s. Lance el dashboard con: streamlit run app.py", time.time() - t0)
    if data_source() != "xm" or args.source:
        log.info("Si la fuente no es la de config/config.yaml, lance el dashboard con la misma fuente: "
                 "GUAVIO_DATA_SOURCE=%s streamlit run app.py", data_source())


if __name__ == "__main__":
    main()
