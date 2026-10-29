"""Pruebas de métricas, del modelo de dos etapas y de la explicabilidad.

Las pruebas que requieren modelos entrenados se omiten si no se ha ejecutado
`python scripts/run_pipeline.py`.
"""
import json

import numpy as np
import pandas as pd
import pytest

from src.models import evaluate as ev


def test_metrics_basic():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert ev.nse(y, y) == pytest.approx(1.0)
    assert ev.kge(y, y) == pytest.approx(1.0)
    assert ev.regression_metrics(y, y)["MAE"] == 0
    assert ev.pinball_loss(y, y, 0.9) == 0
    assert ev.interval_coverage(y, y - 1, y + 1) == 1.0


def test_tercile_and_threshold():
    edges = (1.0, 2.0)
    assert list(ev.to_tercile([0.5, 1.5, 2.5], edges)) == [0, 1, 2]
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    assert 0.2 <= ev.best_f1_threshold(y, p) < 0.8
    assert ev.classification_metrics(y, p, 0.5)["F1"] == 1.0


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 2000)
    assert ev.population_stability_index(a, rng.normal(0, 1, 500)) < 0.1
    assert ev.population_stability_index(a, rng.normal(2, 1, 500)) > 0.25


def test_diebold_mariano_sign():
    rng = np.random.default_rng(0)
    e_good, e_bad = rng.normal(0, 0.5, 300), rng.normal(0, 2, 300)
    assert ev.diebold_mariano(e_good, e_bad)["DM"] < 0


@pytest.mark.usefixtures("trained")
def test_two_stage_outputs(trained):
    if not trained:
        pytest.skip("Modelos no entrenados")
    import joblib

    from src.config import models_dir, path_of
    from src.models.train import predict_two_stage

    art = joblib.load(models_dir() / "oferta_E2_observada_mas_pronostico.joblib")
    ds = pd.read_parquet(path_of("features_oferta")).tail(200)
    pred = predict_two_stage(art, ds)
    assert pred["p_floor"].between(0, 1).all()
    assert (pred["y_hat"] >= 1.0).all()
    floor = pred["p_floor"] > art["tau"]
    assert (pred.loc[floor, "y_hat"] == 1.0).all()


def test_metrics_files(trained):
    if not trained:
        pytest.skip("Modelos no entrenados")
    from src.config import metrics_dir

    m = json.load(open(metrics_dir() / "oferta_metrics.json", encoding="utf-8"))
    assert {"E1_observada", "E2_observada_mas_pronostico"} <= set(m["experimentos"])
    val = m["experimentos"]["E2_observada_mas_pronostico"]["val"]
    assert val["clasificador"]["F1"] > 0.4
    a = json.load(open(metrics_dir() / "aportes_metrics.json", encoding="utf-8"))
    assert a["resumen_pruebas"]["NSE"]["test_a"] > 0.3


def test_shap_additivity(trained):
    """Base + suma de SHAP = salida del modelo (en su escala)."""
    if not trained:
        pytest.skip("Modelos no entrenados")
    from src.explainability.explainer import build_explainers, grouped_importance, local_contributions

    ex = build_explainers()
    item = ex["aportes"]
    X = item["X"].tail(20)
    expl = item["explainer"].explain(X)
    pred = item["explainer"].model.predict(X)
    np.testing.assert_allclose(expl.base_values + expl.values.sum(axis=1), pred, rtol=1e-4, atol=1e-4)
    g = grouped_importance(expl)
    assert g["participacion_%"].sum() == pytest.approx(100)
    assert len(local_contributions(expl, -1, top=5)) == 6

    clf = ex["oferta_clasificador"]
    Xo = clf["X"].tail(20)
    e2 = clf["explainer"].explain(Xo)
    margin = clf["explainer"].model.predict(Xo, output_margin=True)
    np.testing.assert_allclose(e2.base_values + e2.values.sum(axis=1), margin, rtol=1e-3, atol=1e-3)
