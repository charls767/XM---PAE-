"""Transformaciones del objetivo del regresor de oferta.

Viven en un módulo propio para que los modelos guardados con joblib se puedan
cargar desde cualquier punto de entrada (CLI, dashboard, pruebas).
"""
from __future__ import annotations

import numpy as np


def log_premium(y):
    """log de la prima del precio normalizado sobre el mínimo: log(y - 1)."""
    return np.log(np.maximum(np.asarray(y, dtype=float) - 1.0, 1e-3))


def inv_log_premium(z):
    return 1.0 + np.exp(np.asarray(z, dtype=float))
