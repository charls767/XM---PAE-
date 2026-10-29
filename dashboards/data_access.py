"""Acceso a datos, modelos, escenarios y reglas de alerta para el dashboard.

Este módulo no depende de Streamlit: se puede probar con pytest y reutilizar en
otros frentes (Dash, API).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import stats

from src.config import data_source, load_config, metrics_dir, models_dir, path_of
from src.data.features import climatology
from src.data.loader import has_machine_data, load_data
from src.models import evaluate as ev
from src.models.train import predict_two_stage

TERCILE_LABELS = ["Bajo", "Medio", "Alto"]


@dataclass
class Alert:
    level: str        # "rojo", "ambar", "info", "gris"
    title: str
    message: str


def artifacts_ready() -> bool:
    needed = [models_dir() / "aportes_models.joblib", path_of("aportes_pred"),
              path_of("oferta_pred"), path_of("features_oferta")]
    return all(p.exists() for p in needed)


def load_bundle() -> dict[str, Any]:
    """Carga todos los insumos que usa el dashboard."""
    cfg = load_config()
    raw = load_data()
    exp = cfg["oferta_model"]["default_experiment"]
    bundle = {
        "cfg": cfg,
        "raw": raw,
        "aportes_pred": pd.read_parquet(path_of("aportes_pred")),
        "aportes_metrics_h": pd.read_csv(metrics_dir() / "aportes_metrics_por_horizonte.csv"),
        "oferta_pred": pd.read_parquet(path_of("oferta_pred")),
        "oferta_ds": pd.read_parquet(path_of("features_oferta")),
        "falla_pred": pd.read_parquet(path_of("falla_pred")) if path_of("falla_pred").exists() else None,
        "oferta_art": joblib.load(models_dir() / f"oferta_{exp}.joblib"),
        "aportes_art": joblib.load(models_dir() / "aportes_models.joblib"),
        "falla_art": joblib.load(models_dir() / "falla_model.joblib")
        if (models_dir() / "falla_model.joblib").exists() else None,
        "source": data_source(),
        "default_experiment": exp,
    }
    for name in ("aportes_metrics", "oferta_metrics", "falla_metrics"):
        path = metrics_dir() / f"{name}.json"
        bundle[name] = json.load(open(path, encoding="utf-8")) if path.exists() else None
    bundle["has_machine"] = bundle["falla_art"] is not None and has_machine_data(raw)
    train_end = cfg["splits"]["train_end"]
    raw = raw.copy()
    raw["clim_aportes"] = climatology(raw, "aportes_gwh", train_end)
    raw["clim_vol_pct"] = climatology(raw, "volumen_util_pct", train_end)
    train = raw[raw.fecha <= train_end]
    doy = train.fecha.dt.dayofyear
    raw["aportes_p30"] = raw.fecha.dt.dayofyear.map(train.groupby(doy)["aportes_gwh"].quantile(0.30)
                                                   .rolling(15, center=True, min_periods=1).mean())
    raw["aportes_p10"] = raw.fecha.dt.dayofyear.map(train.groupby(doy)["aportes_gwh"].quantile(0.10)
                                                   .rolling(15, center=True, min_periods=1).mean())
    bundle["raw"] = raw
    return bundle


# --------------------------------------------------------------------------
# Pronóstico de aportes y escenarios
# --------------------------------------------------------------------------
def forecast_path(bundle: dict, origin: pd.Timestamp, factor: float = 1.0) -> pd.DataFrame:
    """Trayectoria P10/P50/P90 a 1-15 días desde el origen, con factor de escenario."""
    ap = bundle["aportes_pred"].set_index("fecha")
    raw = bundle["raw"].set_index("fecha")
    row = ap.loc[origin]
    horizons = bundle["aportes_art"]["horizons"]
    dates = [origin + pd.Timedelta(days=h) for h in horizons]
    clim_doy = raw.groupby(raw.index.dayofyear)["clim_aportes"].first()
    p30_doy = raw.groupby(raw.index.dayofyear)["aportes_p30"].first()
    out = pd.DataFrame({
        "fecha": dates,
        "horizonte": horizons,
        "p10": [row[f"p10_h{h}"] * factor for h in horizons],
        "p50": [row[f"p50_h{h}"] * factor for h in horizons],
        "p90": [row[f"p90_h{h}"] * factor for h in horizons],
    })
    out["clim"] = [clim_doy.get(d.dayofyear, np.nan) for d in out.fecha]
    out["p30_hist"] = [p30_doy.get(d.dayofyear, np.nan) for d in out.fecha]
    out["observado"] = [raw["aportes_gwh"].get(d, np.nan) for d in out.fecha]
    return out


def deficit_probability(path: pd.DataFrame, horizon_slice: slice = slice(1, 15)) -> float:
    """P(aporte medio del periodo < P30 histórico), suponiendo normalidad a partir de P10-P90."""
    p = path.iloc[horizon_slice]
    mu = p["p50"].mean()
    sigma = max((p["p90"].mean() - p["p10"].mean()) / (2 * 1.2816), 1e-6)
    thr = p["p30_hist"].mean()
    return float(stats.norm.cdf((thr - mu) / sigma))


def apply_scenario(x: pd.DataFrame, factor: float, vcap: float) -> pd.DataFrame:
    """Perturba las variables de hidrología pronosticada (análisis de sensibilidad)."""
    x = x.copy()
    if factor == 1.0 or "pred_aporte_t" not in x:
        return x
    for c in ("pred_aporte_t", "pred_aporte_7d", "pred_aporte_14d"):
        x[c] = x[c] * factor
    x["pred_anomalia_14d"] = (1 + x["pred_anomalia_14d"]) * factor - 1
    extra = (factor - 1) * 15 * x["pred_aporte_14d"] / factor
    x["pred_vol_proy_14d"] = (x["pred_vol_proy_14d"] + 100 * extra / vcap).clip(0, 100)
    return x


def offer_estimate(bundle: dict, day: pd.Timestamp, factor: float = 1.0) -> dict[str, Any] | None:
    ds = bundle["oferta_ds"]
    row = ds[ds.fecha == day]
    if row.empty:
        return None
    vcap = float(bundle["raw"]["capacidad_util_gwh"].iloc[0])
    x = apply_scenario(row, factor, vcap)
    art = bundle["oferta_art"]
    pred = predict_two_stage(art, x).iloc[0]
    pmin = float(row["pmin"].iloc[0])
    return {
        "p_floor": float(pred["p_floor"]),
        "tau": float(art["tau"]),
        "y_hat": float(pred["y_hat"]),
        "y_reg": float(pred["y_reg"]),
        "precio_cop": float(pred["y_hat"] * pmin),
        "precio_reg_cop": float(pred["y_reg"] * pmin),
        "pmin": pmin,
        "tercil": TERCILE_LABELS[int(pred["tercil_hat"])],
        "observado_cop": float(row["precio_oferta_cop_kwh"].iloc[0]),
        "observado_publicado": False,
        "x": x,
    }


# --------------------------------------------------------------------------
# Deriva y alertas
# --------------------------------------------------------------------------
KEY_DRIFT_FEATURES = ["vol_pct", "aportes_mean30", "bolsa_rel_pmin", "nino34_lag1m", "pred_aporte_14d",
                      "frac_min_mean7", "frac_min_mes_publicado"]


def drift_table(bundle: dict, day: pd.Timestamp, window: int = 30) -> pd.DataFrame:
    ds = bundle["oferta_ds"]
    train = ds[ds.fecha <= bundle["cfg"]["splits"]["train_end"]]
    recent = ds[(ds.fecha > day - pd.Timedelta(days=window)) & (ds.fecha <= day)]
    rows = []
    for f in KEY_DRIFT_FEATURES:
        if f not in ds:
            continue
        lo, hi = train[f].quantile(0.01), train[f].quantile(0.99)
        rows.append({"variable": f, "PSI": ev.population_stability_index(train[f], recent[f]),
                     "valor_actual": float(recent[f].iloc[-1]) if len(recent) else np.nan,
                     "rango_entrenamiento": f"{lo:.2f} a {hi:.2f}",
                     "fuera_de_rango": bool(len(recent) and not lo <= recent[f].iloc[-1] <= hi)})
    return pd.DataFrame(rows)


def build_alerts(bundle: dict, day: pd.Timestamp, path: pd.DataFrame, offer: dict | None,
                 offer_prev: dict | None, p_falla: float | None, drift: pd.DataFrame) -> list[Alert]:
    a = bundle["cfg"]["alerts"]
    raw = bundle["raw"].set_index("fecha")
    alerts: list[Alert] = []

    # Déficit de aportes a 14 días (percentil de la media pronosticada frente a la historia)
    mean_p50 = path["p50"].iloc[1:15].mean()
    train = bundle["raw"][bundle["raw"].fecha <= bundle["cfg"]["splits"]["train_end"]]
    window_hist = train[train.fecha.dt.dayofyear.isin([d.dayofyear for d in path.fecha])]["aportes_gwh"]
    pct = float(stats.percentileofscore(window_hist, mean_p50)) if len(window_hist) else 50.0
    anom = 100 * (mean_p50 / path["clim"].iloc[1:15].mean() - 1)
    if pct < a["deficit_red_pct"]:
        alerts.append(Alert("rojo", "Déficit severo de aportes",
                            f"Aportes esperados a 14 días: {anom:+.0f} % frente a la media (percentil {pct:.0f})."))
    elif pct < a["deficit_amber_pct"]:
        alerts.append(Alert("ambar", "Aportes por debajo de lo esperado",
                            f"Aportes esperados a 14 días: {anom:+.0f} % frente a la media (percentil {pct:.0f})."))

    nino = float(raw.loc[day, "nino34"])
    if nino >= a["nino_red"]:
        alerts.append(Alert("rojo", "El Niño fuerte",
                            f"Niño 3.4 = {nino:+.1f} °C: mayor riesgo de déficit en las próximas semanas."))
    elif nino >= a["nino_amber"]:
        alerts.append(Alert("ambar", "Condición El Niño", f"Niño 3.4 = {nino:+.1f} °C."))

    if offer is not None:
        vol_proj = float(offer["x"]["pred_vol_proy_14d"].iloc[0]) if "pred_vol_proy_14d" in offer["x"] else np.nan
        clim_vol = float(raw.loc[day, "clim_vol_pct"])
        if vol_proj < clim_vol - a["low_volume_points"]:
            alerts.append(Alert("ambar", "Embalse bajo su nivel típico",
                                f"Volumen proyectado a 14 días {vol_proj:.0f} % vs. {clim_vol:.0f} % típico de la fecha."))
        if vol_proj >= a["spill_risk_pct"]:
            alerts.append(Alert("info", "Riesgo de vertimiento",
                                f"Volumen proyectado a 14 días {vol_proj:.0f} % de la capacidad útil."))
        if offer_prev is not None:
            crossed = (offer["p_floor"] > offer["tau"]) != (offer_prev["p_floor"] > offer_prev["tau"])
            jump = 100 * abs(offer["p_floor"] - offer_prev["p_floor"])
            if crossed or jump >= a["regime_change_points"]:
                alerts.append(Alert("info", "Cambio de régimen de oferta",
                                    f"Probabilidad de oferta al mínimo pasó de {100 * offer_prev['p_floor']:.0f} % "
                                    f"a {100 * offer['p_floor']:.0f} %."))
        if offer["tercil"] == "Alto" and any(x.level in ("rojo", "ambar") for x in alerts):
            alerts.append(Alert("ambar", "Oferta esperada en tercil alto",
                                "El precio estimado cae en el tercil alto con una alerta hidrológica activa."))

    if p_falla is not None:
        if p_falla >= a["failure_red"]:
            alerts.append(Alert("rojo", "Riesgo alto de falla",
                                f"Probabilidad de falla forzada en 7 días: {100 * p_falla:.0f} %."))
        elif p_falla >= a["failure_amber"]:
            alerts.append(Alert("ambar", "Riesgo de falla en aumento",
                                f"Probabilidad de falla forzada en 7 días: {100 * p_falla:.0f} %."))

    if not drift.empty and ((drift["PSI"] > a["psi_threshold"]) | drift["fuera_de_rango"]).any():
        vars_ = ", ".join(drift.loc[(drift["PSI"] > a["psi_threshold"]) | drift["fuera_de_rango"], "variable"])
        alerts.append(Alert("rojo", "Condiciones fuera del rango de entrenamiento",
                            f"Interpretar con cautela; variables con deriva: {vars_}."))
    return alerts


LEVEL_ORDER = {"rojo": 0, "ambar": 1, "info": 2, "gris": 3}


def top_alert(alerts: list[Alert]) -> Alert | None:
    return sorted(alerts, key=lambda x: LEVEL_ORDER[x.level])[0] if alerts else None
