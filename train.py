"""Entrenamiento de los modelos del prototipo.

Componentes:
  1. aportes  - pronóstico directo de aportes diarios (GWh/día) a 1-15 días, cuantiles P10/P50/P90.
  2. oferta   - modelo de dos etapas (clasificador de oferta al mínimo + regresor del precio
                normalizado), con experimentos de ablación E1/E2/E4.
  3. falla    - riesgo de falla forzada en los próximos 7 días (mantenimiento predictivo).

Uso:
    python -m src.models.train                    # entrena todo
    python -m src.models.train --component aportes
"""
from __future__ import annotations

import argparse
import json
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from src.config import ROOT, load_config, metrics_dir, models_dir, path_of
from src.data.features import (
    FALLA_NON_FEATURES,
    build_aportes_features,
    build_aportes_targets,
    build_falla_dataset,
    build_oferta_dataset,
    climatology,
    oferta_feature_columns,
)
from src.data.loader import load_data
from src.models import evaluate as ev
from src.models.transforms import inv_log_premium, log_premium
from src.utils.logging_utils import get_logger

log = get_logger(__name__)
SEED = 42


# --------------------------------------------------------------------------
# Utilidades de partición temporal
# --------------------------------------------------------------------------
def split_masks(dates: pd.Series) -> dict[str, pd.Series]:
    s = load_config()["splits"]
    return {
        "train": dates <= s["train_end"],
        "val": (dates >= s["val_start"]) & (dates <= s["val_end"]),
        "dev": dates <= s["val_end"],
        "test_a": (dates >= s["test_a_start"]) & (dates <= s["test_a_end"]),
        "test_b": (dates >= s["test_b_start"]) & (dates <= s["test_b_end"]),
    }


def split_label(dates: pd.Series) -> pd.Series:
    m = split_masks(dates)
    lab = pd.Series("separación", index=dates.index)
    for name in ("train", "val", "test_a", "test_b"):
        lab[m[name]] = name
    return lab


def _json_dump(obj: Any, name: str) -> None:
    path = metrics_dir() / name
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=float)
    log.info("Métricas guardadas en %s", path)


# --------------------------------------------------------------------------
# Componente 1: aportes
# --------------------------------------------------------------------------
def _lgbm_quantile(alpha: float) -> LGBMRegressor:
    params = dict(load_config()["aportes_model"]["lgbm_params"])
    return LGBMRegressor(objective="quantile", alpha=alpha, random_state=SEED, **params)


def _fit_horizon_models(X: pd.DataFrame, Y: pd.DataFrame, dates: pd.Series, cutoff: pd.Timestamp,
                        horizons: list[int], alpha: float) -> dict[int, LGBMRegressor]:
    """Entrena un modelo por horizonte con orígenes cuyo objetivo (o+h) ocurre antes del corte."""
    models = {}
    for h in horizons:
        mask = (dates + pd.Timedelta(days=h) < cutoff) & Y[f"y_h{h}"].notna()
        model = _lgbm_quantile(alpha)
        model.fit(X[mask], Y.loc[mask, f"y_h{h}"])
        models[h] = model
    return models


def train_aportes(df: pd.DataFrame) -> dict[str, Any]:
    cfg = load_config()
    acfg, s = cfg["aportes_model"], cfg["splits"]
    horizons, quantiles = acfg["horizons"], acfg["quantiles"]
    dates = df["fecha"]
    masks = split_masks(dates)
    Y = build_aportes_targets(df, horizons)
    clim = climatology(df, "aportes_gwh", s["train_end"])
    train_cut = pd.Timestamp(s["train_end"]) + pd.Timedelta(days=1)
    dev_cut = pd.Timestamp(s["val_end"]) + pd.Timedelta(days=1)

    # 1) Selección del conjunto de variables (base vs. con clima) en validación.
    selection = {}
    for fs, include in (("base", False), ("clima", True)):
        X = build_aportes_features(df, include_climate=include)
        models = _fit_horizon_models(X, Y, dates, train_cut, horizons, 0.5)
        skills = []
        for h in horizons:
            mv = masks["val"] & Y[f"y_h{h}"].notna()
            pred = models[h].predict(X[mv])
            m = ev.hydro_metrics(Y.loc[mv, f"y_h{h}"], pred, clim.shift(-h)[mv], X.loc[mv, "aportes_lag0"])
            skills.append(m["skill_vs_clim"])
        selection[fs] = float(np.mean(skills))
        log.info("Aportes - conjunto '%s': skill medio vs climatología en validación = %.3f", fs, selection[fs])
    feature_set = max(selection, key=selection.get)
    X = build_aportes_features(df, include_climate=feature_set == "clima")
    features = list(X.columns)
    log.info("Aportes - conjunto elegido: %s", feature_set)

    # 2) Calibración conformal del intervalo P10-P90 (CQR): modelos entrenados en train,
    #    corrección estimada en validación y aplicada luego a los modelos finales.
    q_lo, q_hi = min(quantiles), max(quantiles)
    target_cov = q_hi - q_lo
    lo_tr = _fit_horizon_models(X, Y, dates, train_cut, horizons, q_lo)
    hi_tr = _fit_horizon_models(X, Y, dates, train_cut, horizons, q_hi)
    conformal = {}
    for h in horizons:
        mv = masks["val"] & Y[f"y_h{h}"].notna()
        y = Y.loc[mv, f"y_h{h}"].to_numpy()
        score = np.maximum(lo_tr[h].predict(X[mv]) - y, y - hi_tr[h].predict(X[mv]))
        level = min(1.0, np.ceil((len(y) + 1) * target_cov) / len(y))
        conformal[h] = float(np.quantile(score, level))

    # 3) Modelos finales por cuantil entrenados con todo el periodo de desarrollo.
    final = {q: _fit_horizon_models(X, Y, dates, dev_cut, horizons, q) for q in quantiles}

    def _quantile_preds(Xb: pd.DataFrame, h: int) -> dict[float, np.ndarray]:
        p = {q: final[q][h].predict(Xb) for q in quantiles}
        p[q_lo] = np.minimum(p[q_lo] - conformal[h], p[0.5])
        p[q_hi] = np.maximum(p[q_hi] + conformal[h], p[0.5])
        p[q_lo] = np.maximum(p[q_lo], 0.0)
        return p

    # 4) Evaluación en pruebas A y B (y validación de los modelos de desarrollo, como referencia).
    rows = []
    for split in ("test_a", "test_b"):
        for h in horizons:
            mt = masks[split] & Y[f"y_h{h}"].notna()
            if mt.sum() == 0:
                continue
            y = Y.loc[mt, f"y_h{h}"]
            preds = _quantile_preds(X[mt], h)
            m = ev.hydro_metrics(y, preds[0.5], clim.shift(-h)[mt], X.loc[mt, "aportes_lag0"])
            m["cobertura_P10_P90"] = ev.interval_coverage(y, preds[0.1], preds[0.9])
            m["pinball_P10"] = ev.pinball_loss(y, preds[0.1], 0.1)
            m["pinball_P90"] = ev.pinball_loss(y, preds[0.9], 0.9)
            m["MAE_persistencia"] = float(np.mean(np.abs(y - X.loc[mt, "aportes_lag0"])))
            m["MAE_climatologia"] = float(np.mean(np.abs(y - clim.shift(-h)[mt])))
            m["MAE_media7"] = float(np.mean(np.abs(y - X.loc[mt, "aportes_mean7"])))
            m["MAE_estacional_ingenua"] = float(np.nanmean(np.abs(y - df["aportes_gwh"].shift(365 - h)[mt])))
            rows.append({"split": split, "horizonte": h, **m})
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(metrics_dir() / "aportes_metrics_por_horizonte.csv", index=False)

    # 5) Pronósticos walk-forward fuera de muestra (alimentan el modelo de oferta).
    oof = pd.DataFrame(index=dates.values, columns=[f"p50_h{h}" for h in horizons], dtype=float)
    oof.index.name = "fecha"
    cutoffs = pd.date_range(acfg["oof_start"], dates.max() + pd.offsets.YearBegin(1), freq=acfg["oof_refit_freq"])
    for c0, c1 in zip(cutoffs[:-1], cutoffs[1:]):
        models = _fit_horizon_models(X, Y, dates, c0, horizons, 0.5)
        block = (dates >= c0) & (dates < c1)
        if block.sum() == 0:
            continue
        for h in horizons:
            oof.loc[dates[block].values, f"p50_h{h}"] = models[h].predict(X[block])
        log.info("Aportes - walk-forward con corte %s: %d orígenes pronosticados", c0.date(), int(block.sum()))
    oof.reset_index().to_parquet(path_of("aportes_oof"), index=False)

    # 6) Pronósticos de los modelos finales para todos los orígenes (uso en el dashboard).
    final_pred = pd.DataFrame({"fecha": dates})
    for h in horizons:
        preds = _quantile_preds(X, h)
        for q in quantiles:
            final_pred[f"p{int(q * 100)}_h{h}"] = preds[q]
    final_pred["split"] = split_label(dates)
    final_pred.to_parquet(ROOT / "data/processed/aportes_final_predictions.parquet", index=False)
    X.assign(fecha=dates).to_parquet(path_of("features_aportes"), index=False)

    artifact = {"feature_set": feature_set, "features": features, "horizons": horizons,
                "quantiles": quantiles, "models": final, "selection_skill": selection,
                "conformal_correction": conformal}
    joblib.dump(artifact, models_dir() / "aportes_models.joblib")

    summary = {
        "conjunto_variables": feature_set,
        "skill_validacion": selection,
        "resumen_pruebas": metrics_df.groupby("split")[["MAE", "NSE", "KGE", "skill_vs_clim",
                                                       "skill_vs_persist", "cobertura_P10_P90"]].mean().to_dict(),
    }
    _json_dump(summary, "aportes_metrics.json")
    return summary


# --------------------------------------------------------------------------
# Componente 2: oferta en dos etapas
# --------------------------------------------------------------------------
def _with_log_target(model):
    return TransformedTargetRegressor(regressor=model, func=log_premium, inverse_func=inv_log_premium,
                                      check_inverse=False)


def _regressor_candidates() -> dict[str, Any]:
    params = load_config()["oferta_model"]["regressor_params"]
    return {
        "Ridge": _with_log_target(Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])),
        "RandomForest": _with_log_target(RandomForestRegressor(n_estimators=400, min_samples_leaf=3,
                                                               max_features=0.5, n_jobs=-1, random_state=SEED)),
        "XGBoost": _with_log_target(XGBRegressor(random_state=SEED, n_jobs=-1, **params)),
    }


def _clone(model):
    from sklearn.base import clone
    return clone(model)


def _cv_mae(model, X: pd.DataFrame, y: pd.Series) -> float:
    s = load_config()["splits"]
    tss = TimeSeriesSplit(n_splits=s["cv_folds"], gap=s["gap_days"])
    scores = []
    for tr, te in tss.split(X):
        m = _clone(model).fit(X.iloc[tr], y.iloc[tr])
        scores.append(np.mean(np.abs(y.iloc[te] - m.predict(X.iloc[te]))))
    return float(np.mean(scores))


def _fit_two_stage(ds: pd.DataFrame, feats: list[str], masks: dict[str, pd.Series], edges) -> dict[str, Any]:
    cfg = load_config()["oferta_model"]
    tr, va, dev = masks["train"], masks["val"], masks["dev"]
    X, q, y = ds[feats], ds["q_floor"], ds["y_norm"]

    # Etapa 1: clasificador (entrenado en train; calibración y umbral en validación).
    pos = max(int(q[tr].sum()), 1)
    clf = XGBClassifier(random_state=SEED, n_jobs=-1, eval_metric="logloss",
                        scale_pos_weight=(len(q[tr]) - pos) / pos, **cfg["classifier_params"])
    clf.fit(X[tr], q[tr])
    raw_val = clf.predict_proba(X[va])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(raw_val, q[va])
    tau = ev.best_f1_threshold(q[va], iso.predict(raw_val))

    # Etapa 2: regresor sobre días sin oferta al mínimo. Selección por MAE en validación.
    nf = q == 0
    cand_scores = {}
    for name, model in _regressor_candidates().items():
        m = _clone(model).fit(X[tr & nf], y[tr & nf])
        cand_scores[name] = {
            "MAE_validacion": float(np.mean(np.abs(y[va & nf] - m.predict(X[va & nf])))),
            "MAE_cv_train": _cv_mae(model, X[tr & nf], y[tr & nf]),
        }
    best = min(cand_scores, key=lambda k: cand_scores[k]["MAE_validacion"])
    reg = _clone(_regressor_candidates()[best]).fit(X[dev & nf], y[dev & nf])

    return {"classifier": clf, "calibrator": iso, "tau": tau, "regressor": reg, "regressor_name": best,
            "regressor_candidates": cand_scores, "features": feats, "tercile_edges": edges}


def predict_two_stage(artifact: dict[str, Any], X: pd.DataFrame) -> pd.DataFrame:
    """Aplica el modelo de dos etapas. Devuelve probabilidad, precio normalizado y tercil."""
    feats = artifact["features"]
    raw = artifact["classifier"].predict_proba(X[feats])[:, 1]
    p = artifact["calibrator"].predict(raw)
    reg = artifact["regressor"].predict(X[feats])
    reg = np.maximum(reg, 1.0)
    y_hat = np.where(p > artifact["tau"], 1.0, reg)
    return pd.DataFrame({"p_floor": p, "p_floor_raw": raw, "y_reg": reg, "y_hat": y_hat,
                         "tercil_hat": ev.to_tercile(y_hat, artifact["tercile_edges"])}, index=X.index)


def _evaluate_two_stage(ds: pd.DataFrame, pred: pd.DataFrame, mask: pd.Series, art: dict) -> dict[str, Any]:
    q, y = ds["q_floor"][mask], ds["y_norm"][mask]
    p = pred[mask]
    nf = q == 0
    out = {
        "clasificador": ev.classification_metrics(q, p["p_floor"], art["tau"]),
        "regresor_dias_sin_minimo": ev.regression_metrics(y[nf], p["y_reg"][nf]) if nf.sum() > 2 else {},
        "combinado": ev.regression_metrics(y, p["y_hat"]),
    }
    out["combinado"]["acierto_terciles"] = ev.tercile_accuracy(y, p["y_hat"], art["tercile_edges"])
    out["combinado"]["MAE_COP_kWh"] = float(np.mean(np.abs(ds["precio_oferta_cop_kwh"][mask]
                                                           - p["y_hat"] * ds["pmin"][mask])))
    return out


def train_oferta(df: pd.DataFrame) -> dict[str, Any]:
    cfg = load_config()
    ocfg = cfg["oferta_model"]
    oof = pd.read_parquet(path_of("aportes_oof")).set_index("fecha")
    start = pd.Timestamp(cfg["aportes_model"]["oof_start"]) + pd.Timedelta(days=30)
    results, all_preds, artifacts = {}, [], {}

    for exp in ocfg["experiments"]:
        ds = build_oferta_dataset(df, oof, exp)
        feats = oferta_feature_columns(ds)
        ok = (ds["fecha"] >= start) & ds[feats].notna().all(axis=1) & ds["y_norm"].notna()
        ds = ds[ok].copy()
        masks = split_masks(ds["fecha"])
        edges = ev.tercile_edges(ds.loc[masks["train"], "y_norm"])
        art = _fit_two_stage(ds, feats, masks, edges)
        pred = predict_two_stage(art, ds)
        res = {"n_variables": len(feats), "umbral_tau": art["tau"], "regresor": art["regressor_name"],
               "candidatos_regresor": art["regressor_candidates"], "terciles": edges}
        for split in ("val", "test_a", "test_b"):
            res[split] = _evaluate_two_stage(ds, pred, masks[split], art)
        results[exp] = res
        artifacts[exp] = art
        joblib.dump(art, models_dir() / f"oferta_{exp}.joblib")
        out = ds[["fecha", "y_norm", "q_floor", "precio_oferta_cop_kwh", "pmin"]].join(pred)
        out["experimento"] = exp
        out["split"] = split_label(ds["fecha"])
        all_preds.append(out)
        log.info("Oferta %s: F1 test_a=%.3f | MAE comb test_a=%.3f | terciles test_a=%.3f | regresor=%s",
                 exp, res["test_a"]["clasificador"]["F1"], res["test_a"]["combinado"]["MAE"],
                 res["test_a"]["combinado"]["acierto_terciles"], art["regressor_name"])
        if exp == ocfg["default_experiment"]:
            ds.to_parquet(path_of("features_oferta"), index=False)

    preds = pd.concat(all_preds, ignore_index=True)
    preds.to_parquet(ROOT / "data/processed/oferta_predictions.parquet", index=False)

    # Baselines (con las mismas filas del experimento por defecto).
    base_ds = pd.read_parquet(path_of("features_oferta"))
    masks = split_masks(base_ds["fecha"])
    feats = artifacts[ocfg["default_experiment"]]["features"]
    tr, va = masks["train"], masks["val"]
    rate_by_month = base_ds[tr].groupby("mes")["q_floor"].mean()
    p_month = base_ds["mes"].map(rate_by_month).fillna(base_ds["q_floor"][tr].mean())
    tau_month = ev.best_f1_threshold(base_ds["q_floor"][va], p_month[va])
    logit = Pipeline([("s", StandardScaler()), ("m", LogisticRegression(max_iter=2000, class_weight="balanced"))])
    logit.fit(base_ds.loc[tr, feats], base_ds.loc[tr, "q_floor"])
    p_logit = logit.predict_proba(base_ds[feats])[:, 1]
    tau_logit = ev.best_f1_threshold(base_ds["q_floor"][va], p_logit[va.values])
    baselines = {}
    for split in ("test_a", "test_b"):
        m = masks[split]
        y = base_ds["y_norm"][m]
        baselines[split] = {
            "clasificador_tasa_mensual": ev.classification_metrics(base_ds["q_floor"][m], p_month[m], tau_month),
            "clasificador_logistico": ev.classification_metrics(base_ds["q_floor"][m], p_logit[m.values], tau_logit),
            "precio_media_mes_publicado": ev.regression_metrics(y, base_ds["oferta_mes_media"][m]),
        }
        edges = artifacts[ocfg["default_experiment"]]["tercile_edges"]
        baselines[split]["precio_media_mes_publicado"]["acierto_terciles"] = ev.tercile_accuracy(
            y, base_ds["oferta_mes_media"][m], edges)

    # Prueba de Diebold-Mariano: E2 vs E1 (estadístico < 0 => E2 mejor).
    dm = {}
    e1 = preds[preds.experimento == "E1_observada"].set_index("fecha")
    e2 = preds[preds.experimento == "E2_observada_mas_pronostico"].set_index("fecha")
    common = e1.index.intersection(e2.index)
    for split in ("test_a", "test_b"):
        idx = common[e1.loc[common, "split"] == split]
        dm[split] = ev.diebold_mariano(e2.loc[idx, "y_hat"] - e2.loc[idx, "y_norm"],
                                       e1.loc[idx, "y_hat"] - e1.loc[idx, "y_norm"])
    summary = {"experimentos": results, "baselines": baselines, "diebold_mariano_E2_vs_E1": dm}
    _json_dump(summary, "oferta_metrics.json")
    return summary


# --------------------------------------------------------------------------
# Componente complementario: riesgo de falla
# --------------------------------------------------------------------------
def train_falla(df: pd.DataFrame) -> dict[str, Any]:
    cfg = load_config()
    fcfg = cfg["falla_model"]
    ds = build_falla_dataset(df, fcfg["horizon_days"])
    feats = [c for c in ds.columns if c not in FALLA_NON_FEATURES]
    ds = ds[ds[feats].notna().all(axis=1) & ds["y_falla"].notna()].copy()
    ds["y_falla"] = ds["y_falla"].astype(int)
    masks = split_masks(ds["fecha"])
    tr, va = masks["train"], masks["val"]
    pos = max(int(ds.loc[tr, "y_falla"].sum()), 1)
    clf = XGBClassifier(random_state=SEED, n_jobs=-1, eval_metric="logloss",
                        scale_pos_weight=(int(tr.sum()) - pos) / pos, **fcfg["classifier_params"])
    clf.fit(ds.loc[tr, feats], ds.loc[tr, "y_falla"])
    raw = clf.predict_proba(ds[feats])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(raw[va.values], ds.loc[va, "y_falla"])
    p = iso.predict(raw)
    tau = ev.best_f1_threshold(ds.loc[va, "y_falla"], p[va.values])
    res = {"horizonte_dias": fcfg["horizon_days"], "umbral_tau": tau}
    for split in ("val", "test_a", "test_b"):
        m = masks[split].values
        res[split] = ev.classification_metrics(ds["y_falla"].values[m], p[m], tau)
    art = {"classifier": clf, "calibrator": iso, "tau": tau, "features": feats}
    joblib.dump(art, models_dir() / "falla_model.joblib")
    out = ds[["fecha", "y_falla"]].copy()
    out["p_falla"], out["p_falla_raw"] = p, raw
    out["split"] = split_label(ds["fecha"])
    out.to_parquet(ROOT / "data/processed/falla_predictions.parquet", index=False)
    ds.to_parquet(path_of("features_falla"), index=False)
    log.info("Falla: F1 test_a=%.3f | PR-AUC test_a=%.3f | prevalencia=%.3f",
             res["test_a"]["F1"], res["test_a"]["PR_AUC"], res["test_a"]["prevalencia"])
    _json_dump(res, "falla_metrics.json")
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena los modelos del prototipo Guavio.")
    parser.add_argument("--component", choices=["all", "aportes", "oferta", "falla"], default="all")
    args = parser.parse_args()
    df = load_data()
    if args.component in ("all", "aportes"):
        train_aportes(df)
    if args.component in ("all", "oferta"):
        train_oferta(df)
    if args.component in ("all", "falla"):
        train_falla(df)
    log.info("Entrenamiento terminado.")


if __name__ == "__main__":
    main()
