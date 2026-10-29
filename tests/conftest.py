import os
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Las pruebas de modelos y dashboard usan la fuente sintética (no requieren internet).
# Para probar con otra fuente entrenada: GUAVIO_TEST_SOURCE=xm python -m pytest
os.environ["GUAVIO_DATA_SOURCE"] = os.getenv("GUAVIO_TEST_SOURCE", "synthetic")

from src.config import load_config, models_dir, path_of  # noqa: E402
from src.data.loader import generate_synthetic_data  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def df(cfg) -> pd.DataFrame:
    """Datos sintéticos generados en memoria (no toca data/raw)."""
    return generate_synthetic_data(cfg)


@pytest.fixture(scope="session")
def trained() -> bool:
    return (models_dir() / "aportes_models.joblib").exists() and path_of("features_oferta").exists()
