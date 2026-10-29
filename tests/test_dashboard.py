"""Prueba de humo del dashboard con el arnés de pruebas de Streamlit."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_app_runs_without_exceptions(trained):
    if not trained:
        pytest.skip("Modelos no entrenados")
    import os

    from streamlit.testing.v1 import AppTest

    os.environ.pop("GUAVIO_DEFAULT_SOURCE", None)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
    at.run()
    assert not at.exception
    assert "Mantenimiento predictivo" in [t.label for t in at.tabs]
    at.date_input[0].set_value("2024-10-15")
    at.run()
    assert not at.exception
    [s for s in at.selectbox if s.label == "Escenario hidrológico"][0].set_value("El Niño fuerte")
    at.run()
    assert not at.exception


def test_scenario_lowers_forecast(trained):
    if not trained:
        pytest.skip("Modelos no entrenados")
    import pandas as pd

    from dashboards.data_access import forecast_path, load_bundle, offer_estimate

    b = load_bundle()
    day = pd.Timestamp("2026-09-30")
    base = forecast_path(b, day - pd.Timedelta(days=2), 1.0)
    dry = forecast_path(b, day - pd.Timedelta(days=2), 0.7)
    assert (dry.p50 < base.p50).all()
    o = offer_estimate(b, day, 0.7)
    assert o is not None and 0 <= o["p_floor"] <= 1


def test_app_runs_with_xm_source():
    """Si hay modelos entrenados con el simulador de la API (python -m tests.xm_mock), el dashboard los usa."""
    import os

    from src.config import models_dir, set_data_source

    prev = os.environ.get("GUAVIO_DATA_SOURCE")
    set_data_source("xm_mock")
    try:
        if not (models_dir() / "aportes_models.joblib").exists():
            pytest.skip("Sin modelos de la fuente xm_mock")
        from streamlit.testing.v1 import AppTest

        os.environ.pop("GUAVIO_DEFAULT_SOURCE", None)
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
        at.run()
        assert not at.exception
        labels = [t.label for t in at.tabs]
        assert "Explicabilidad y desempeño" in labels
        assert "Mantenimiento predictivo" not in labels  # la fuente XM no trae datos de máquina
    finally:
        os.environ["GUAVIO_DATA_SOURCE"] = prev or "synthetic"
        os.environ.pop("GUAVIO_DEFAULT_SOURCE", None)
