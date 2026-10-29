"""Métricas de evaluación para regresión, clasificación, cuantiles y terciles."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def _clean(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    return y_true[mask], y_pred[mask]


def nse(y_true, y_pred) -> float:
    """Eficiencia de Nash-Sutcliffe (1 = perfecto, 0 = igual a la media)."""
    y, p = _clean(y_true, y_pred)
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def kge(y_true, y_pred) -> float:
    """Eficiencia de Kling-Gupta (1 = perfecto)."""
    y, p = _clean(y_true, y_pred)
    r = np.corrcoef(y, p)[0, 1]
    alpha = p.std() / y.std()
    beta = p.mean() / y.mean()
    return float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def wape(y_true, y_pred) -> float:
    y, p = _clean(y_true, y_pred)
    return float(100 * np.sum(np.abs(y - p)) / np.sum(np.abs(y)))


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y, p = _clean(y_true, y_pred)
    mae = mean_absolute_error(y, p)
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    mape = float(100 * np.mean(np.abs((y - p) / np.where(y == 0, np.nan, y))))
    return {
        "n": int(len(y)),
        "MAE": float(mae),
        "RMSE": rmse,
        "MAPE_%": mape,
        "WAPE_%": wape(y, p),
        "R2": float(r2_score(y, p)),
        "Pearson": float(np.corrcoef(y, p)[0, 1]) if len(y) > 2 else float("nan"),
    }


def hydro_metrics(y_true, y_pred, y_clim, y_persist) -> dict[str, float]:
    """Métricas hidrológicas y skill score MAE frente a climatología y persistencia."""
    m = regression_metrics(y_true, y_pred)
    y, p = _clean(y_true, y_pred)
    m["NSE"] = nse(y_true, y_pred)
    m["KGE"] = kge(y_true, y_pred)
    mae = m["MAE"]
    m["skill_vs_clim"] = float(1 - mae / mean_absolute_error(*_clean(y_true, y_clim)))
    m["skill_vs_persist"] = float(1 - mae / mean_absolute_error(*_clean(y_true, y_persist)))
    return m


def pinball_loss(y_true, y_pred, q: float) -> float:
    y, p = _clean(y_true, y_pred)
    diff = y - p
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def interval_coverage(y_true, lower, upper) -> float:
    y = np.asarray(y_true, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    mask = ~(np.isnan(y) | np.isnan(lo) | np.isnan(hi))
    return float(np.mean((y[mask] >= lo[mask]) & (y[mask] <= hi[mask])))


def classification_metrics(y_true, proba, threshold: float) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(proba, dtype=float)
    pred = (p > threshold).astype(int)
    out = {
        "n": int(len(y)),
        "prevalencia": float(y.mean()),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "Brier": float(brier_score_loss(y, p)),
        "umbral": float(threshold),
    }
    if len(np.unique(y)) > 1:
        out["PR_AUC"] = float(average_precision_score(y, p))
        out["ROC_AUC"] = float(roc_auc_score(y, p))
    else:
        out["PR_AUC"] = float("nan")
        out["ROC_AUC"] = float("nan")
    return out


def best_f1_threshold(y_true, proba) -> float:
    """Umbral que maximiza F1 (se usa solo sobre el bloque de validación)."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(proba, dtype=float)
    grid = np.unique(np.quantile(p, np.linspace(0.01, 0.99, 99)))
    if y.sum() == 0:
        return 0.5
    scores = [f1_score(y, (p > t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def tercile_edges(y_train) -> tuple[float, float]:
    y = np.asarray(y_train, dtype=float)
    y = y[~np.isnan(y)]
    return float(np.quantile(y, 1 / 3)), float(np.quantile(y, 2 / 3))


def to_tercile(values, edges: tuple[float, float]) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    return np.digitize(v, edges)  # 0 = bajo, 1 = medio, 2 = alto


def tercile_accuracy(y_true, y_pred, edges: tuple[float, float]) -> float:
    y, p = _clean(y_true, y_pred)
    return float(np.mean(to_tercile(y, edges) == to_tercile(p, edges)))


def tercile_confusion(y_true, y_pred, edges: tuple[float, float]) -> pd.DataFrame:
    y, p = _clean(y_true, y_pred)
    labels = ["Bajo", "Medio", "Alto"]
    t = pd.Categorical([labels[i] for i in to_tercile(y, edges)], categories=labels)
    q = pd.Categorical([labels[i] for i in to_tercile(p, edges)], categories=labels)
    return pd.crosstab(pd.Series(t, name="Observado"), pd.Series(q, name="Predicho"), dropna=False)


def diebold_mariano(err_a, err_b, h: int = 1) -> dict[str, float]:
    """Prueba de Diebold-Mariano con pérdida absoluta. Estadístico < 0 => A mejor que B."""
    a = np.abs(np.asarray(err_a, dtype=float))
    b = np.abs(np.asarray(err_b, dtype=float))
    mask = ~(np.isnan(a) | np.isnan(b))
    d = a[mask] - b[mask]
    n = len(d)
    if n < 10:
        return {"DM": float("nan"), "p_value": float("nan")}
    gamma0 = np.var(d, ddof=0)
    var = gamma0
    for k in range(1, h):
        var += 2 * np.cov(d[k:], d[:-k], ddof=0)[0, 1]
    dm = d.mean() / np.sqrt(var / n)
    p = 2 * (1 - stats.norm.cdf(abs(dm)))
    return {"DM": float(dm), "p_value": float(p)}


def population_stability_index(expected, actual, bins: int = 10) -> float:
    """PSI entre la distribución de entrenamiento (expected) y la reciente (actual)."""
    e = np.asarray(expected, dtype=float)
    a = np.asarray(actual, dtype=float)
    e, a = e[~np.isnan(e)], a[~np.isnan(a)]
    if len(e) == 0 or len(a) == 0:
        return float("nan")
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    pe = np.histogram(e, edges)[0] / len(e)
    pa = np.histogram(a, edges)[0] / len(a)
    pe, pa = np.clip(pe, 1e-4, None), np.clip(pa, 1e-4, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))
