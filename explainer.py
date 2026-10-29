"""Explicabilidad de los modelos con SHAP (global, local, por familia y por régimen) y LIME.

Escalas de salida de SHAP según el modelo:
  - Aportes (LightGBM cuantil P50): GWh/día.
  - Oferta etapa 1 (XGBoost clasificador): log-odds de ofertar al mínimo (antes de calibrar).
  - Oferta etapa 2 (regresor con objetivo transformado): log de la prima sobre el mínimo, log(y - 1).
  - Falla (XGBoost clasificador): log-odds de falla en los próximos 7 días.

Las explicaciones describen lo que aprendió el modelo, no relaciones causales.

Uso:
    python -m src.explainability.explainer          # genera figuras y tablas en reports/
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from sklearn.compose import TransformedTargetRegressor  # noqa: E402
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor  # noqa: E402
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from src.config import figures_dir, load_config, metrics_dir, models_dir, path_of  # noqa: E402
from src.data.features import family_of  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402

log = get_logger(__name__)

TREE_TYPES = ("XGB", "LGBM", "RandomForest", "ExtraTrees", "GradientBoosting", "DecisionTree")


@dataclass
class ModelExplainer:
    """Envuelve un estimador (posiblemente dentro de Pipeline o TransformedTargetRegressor)."""

    model: Any
    background: pd.DataFrame
    output_name: str = "salida del modelo"
    max_background: int = 200

    def __post_init__(self) -> None:
        est = self.model
        if isinstance(est, TransformedTargetRegressor):
            est = est.regressor_
        self._preproc: Pipeline | None = None
        if isinstance(est, Pipeline):
            self._preproc = est[:-1]
            est = est[-1]
        self.estimator = est
        bg = self.background.sample(min(len(self.background), self.max_background), random_state=0)
        bg_t = self._transform(bg)
        name = type(est).__name__
        if any(t in name for t in TREE_TYPES):
            self.kind = "tree"
            self._explainer = shap.TreeExplainer(est)
        elif isinstance(est, (Ridge, LinearRegression, LogisticRegression)):
            self.kind = "linear"
            self._explainer = shap.LinearExplainer(est, bg_t)
        else:
            self.kind = "kernel"
            predict = est.predict_proba if hasattr(est, "predict_proba") else est.predict
            self._explainer = shap.KernelExplainer(lambda a: np.asarray(predict(a)).reshape(len(a), -1)[:, -1],
                                                   shap.kmeans(bg_t, 20))

    def _transform(self, X: pd.DataFrame) -> np.ndarray | pd.DataFrame:
        return self._preproc.transform(X) if self._preproc is not None else X

    def explain(self, X: pd.DataFrame) -> shap.Explanation:
        Xt = self._transform(X)
        if self.kind == "kernel":
            values = np.asarray(self._explainer.shap_values(Xt, silent=True))
            base = np.full(len(X), float(np.ravel(self._explainer.expected_value)[-1]))
        else:
            exp = self._explainer(Xt)
            values = exp.values
            base = exp.base_values
            if values.ndim == 3:  # clasificadores multiclase / RandomForest con 2 salidas
                values = values[:, :, -1]
                base = np.asarray(base)[:, -1] if np.ndim(base) == 2 else base
        base = np.broadcast_to(np.asarray(base, dtype=float).ravel(), (len(X),)) if np.size(base) == 1 else \
            np.asarray(base, dtype=float).ravel()
        return shap.Explanation(values=np.asarray(values, dtype=float), base_values=base,
                                data=X.to_numpy(dtype=float), feature_names=list(X.columns))


# --------------------------------------------------------------------------
# Tablas
# --------------------------------------------------------------------------
def global_importance(expl: shap.Explanation) -> pd.DataFrame:
    imp = np.abs(expl.values).mean(axis=0)
    df = pd.DataFrame({"variable": expl.feature_names, "shap_medio_abs": imp})
    df["familia"] = df["variable"].map(family_of)
    return df.sort_values("shap_medio_abs", ascending=False).reset_index(drop=True)


def grouped_importance(expl: shap.Explanation) -> pd.DataFrame:
    """Suma de valores SHAP por familia antes del valor absoluto (atenúa la colinealidad entre rezagos)."""
    fams = pd.Series(expl.feature_names).map(family_of)
    grouped = pd.DataFrame(expl.values, columns=expl.feature_names).T.groupby(fams.values).sum().T
    out = grouped.abs().mean().rename("shap_medio_abs").reset_index().rename(columns={"index": "familia"})
    out["participacion_%"] = 100 * out["shap_medio_abs"] / out["shap_medio_abs"].sum()
    return out.sort_values("shap_medio_abs", ascending=False).reset_index(drop=True)


def local_contributions(expl: shap.Explanation, row: int = -1, top: int | None = None) -> pd.DataFrame:
    df = pd.DataFrame({
        "variable": expl.feature_names,
        "valor": expl.data[row],
        "contribucion": expl.values[row],
    })
    df["familia"] = df["variable"].map(family_of)
    df["abs"] = df["contribucion"].abs()
    df = df.sort_values("abs", ascending=False).drop(columns="abs").reset_index(drop=True)
    if top:
        rest = df.iloc[top:]
        df = df.iloc[:top].copy()
        if len(rest):
            df.loc[len(df)] = {"variable": f"otras {len(rest)} variables", "valor": np.nan,
                               "contribucion": rest["contribucion"].sum(), "familia": "Otras"}
    return df


def local_grouped(expl: shap.Explanation, row: int = -1) -> pd.DataFrame:
    df = local_contributions(expl, row)
    return (df.groupby("familia")["contribucion"].sum().reset_index()
              .assign(abs=lambda d: d.contribucion.abs()).sort_values("abs", ascending=False)
              .drop(columns="abs").reset_index(drop=True))


def importance_by_regime(expl: shap.Explanation, regimes: pd.Series) -> pd.DataFrame:
    """Importancia media absoluta por familia en cada régimen (p. ej. húmedo, seco, El Niño)."""
    fams = pd.Series(expl.feature_names).map(family_of)
    vals = pd.DataFrame(expl.values, columns=expl.feature_names).T.groupby(fams.values).sum().T
    vals["regimen"] = np.asarray(regimes)
    return vals.groupby("regimen").agg(lambda s: s.abs().mean()).T


def hydrological_regime(dates: pd.Series, nino: pd.Series) -> pd.Series:
    month = pd.to_datetime(dates).dt.month
    reg = np.where(month.isin([5, 6, 7, 8, 9, 10]), "Húmedo", "Seco")
    reg = np.where(np.asarray(nino) >= 1.0, "El Niño", reg)
    return pd.Series(reg, index=dates.index)


# --------------------------------------------------------------------------
# LIME (comparación local)
# --------------------------------------------------------------------------
def lime_explanation(predict_fn: Callable, X_train: pd.DataFrame, x_row: pd.Series,
                     mode: str = "regression", num_features: int = 8) -> pd.DataFrame:
    from lime.lime_tabular import LimeTabularExplainer

    explainer = LimeTabularExplainer(X_train.to_numpy(dtype=float), feature_names=list(X_train.columns),
                                     mode=mode, discretize_continuous=True, random_state=0)
    exp = explainer.explain_instance(x_row.to_numpy(dtype=float), predict_fn, num_features=num_features)
    return pd.DataFrame(exp.as_list(), columns=["condicion", "peso"])


# --------------------------------------------------------------------------
# Figuras (matplotlib, para el informe)
# --------------------------------------------------------------------------
def save_summary_plot(expl: shap.Explanation, name: str, max_display: int = 15) -> str:
    plt.figure()
    shap.plots.beeswarm(expl, max_display=max_display, show=False)
    path = figures_dir() / f"{name}_shap_summary.png"
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close("all")
    return str(path)


def save_bar_plot(expl: shap.Explanation, name: str, max_display: int = 15) -> str:
    plt.figure()
    shap.plots.bar(expl, max_display=max_display, show=False)
    path = figures_dir() / f"{name}_shap_bar.png"
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close("all")
    return str(path)


def save_waterfall_plot(expl: shap.Explanation, name: str, row: int = -1, max_display: int = 12) -> str:
    plt.figure()
    shap.plots.waterfall(expl[row], max_display=max_display, show=False)
    path = figures_dir() / f"{name}_shap_waterfall.png"
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close("all")
    return str(path)


def save_grouped_plot(grouped: pd.DataFrame, name: str, title: str) -> str:
    fig, ax = plt.subplots(figsize=(7, 3.6))
    g = grouped.sort_values("shap_medio_abs")
    ax.barh(g["familia"], g["shap_medio_abs"], color="#2a6f97")
    ax.set_xlabel("|SHAP| medio agrupado por familia")
    ax.set_title(title, loc="left", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    path = figures_dir() / f"{name}_shap_familias.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return str(path)


def save_dependence_plot(expl: shap.Explanation, feature: str, name: str) -> str:
    plt.figure()
    shap.plots.scatter(expl[:, feature], color=expl, show=False)
    path = figures_dir() / f"{name}_shap_dependencia_{feature}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close("all")
    return str(path)


# --------------------------------------------------------------------------
# Carga de modelos y explicadores listos para usar
# --------------------------------------------------------------------------
def build_explainers() -> dict[str, dict[str, Any]]:
    """Construye los explicadores de los cuatro modelos y devuelve sus matrices X completas.

    No calcula valores SHAP (eso se hace bajo demanda con `.explain`), por lo que es rápido.
    """
    cfg = load_config()
    out: dict[str, dict[str, Any]] = {}

    # Aportes (P50, horizonte 2 = día de la oferta)
    ap = joblib.load(models_dir() / "aportes_models.joblib")
    Xa = pd.read_parquet(path_of("features_aportes"))
    dates_a = Xa.pop("fecha")
    Xa = Xa[ap["features"]]
    ok = Xa.notna().all(axis=1)
    out["aportes"] = {"explainer": ModelExplainer(ap["models"][0.5][2], Xa[ok], "aportes día t (GWh/día)"),
                      "X": Xa[ok], "dates": dates_a[ok], "unidad": "GWh/día",
                      "titulo": "Aportes del día de la oferta (P50)"}

    # Oferta, etapas 1 y 2
    exp_name = cfg["oferta_model"]["default_experiment"]
    of = joblib.load(models_dir() / f"oferta_{exp_name}.joblib")
    ds = pd.read_parquet(path_of("features_oferta"))
    Xo = ds[of["features"]]
    out["oferta_clasificador"] = {"explainer": ModelExplainer(of["classifier"], Xo, "log-odds de oferta al mínimo"),
                                  "X": Xo, "dates": ds["fecha"], "unidad": "log-odds",
                                  "titulo": "Etapa 1: probabilidad de oferta al mínimo"}
    out["oferta_regresor"] = {"explainer": ModelExplainer(of["regressor"], Xo, "log de la prima sobre el mínimo"),
                              "X": Xo, "dates": ds["fecha"], "unidad": "log(prima sobre el mínimo)",
                              "titulo": "Etapa 2: precio de oferta normalizado"}

    # Falla
    fa = joblib.load(models_dir() / "falla_model.joblib")
    dsf = pd.read_parquet(path_of("features_falla"))
    Xf = dsf[fa["features"]]
    out["falla"] = {"explainer": ModelExplainer(fa["classifier"], Xf, "log-odds de falla a 7 días"),
                    "X": Xf, "dates": dsf["fecha"], "unidad": "log-odds",
                    "titulo": "Riesgo de falla en los próximos 7 días"}
    return out


def load_explainers(window_days: int = 730) -> dict[str, dict[str, Any]]:
    """Explicadores con los valores SHAP de la ventana más reciente ya calculados."""
    out = build_explainers()
    for item in out.values():
        recent = item["dates"] > item["dates"].max() - pd.Timedelta(days=window_days)
        item["X"], item["dates"] = item["X"][recent.values], item["dates"][recent.values]
        item["explanation"] = item["explainer"].explain(item["X"])
    return out


def run_all() -> None:
    """Genera figuras y tablas SHAP de todos los modelos para la fecha más reciente."""
    from src.data.loader import load_data

    raw = load_data()
    nino = raw.set_index("fecha")["nino34"]
    explainers = load_explainers()
    titles = {
        "aportes": "Aportes día t (P50): contribución por familia",
        "oferta_clasificador": "Oferta al mínimo (etapa 1): contribución por familia",
        "oferta_regresor": "Precio de oferta (etapa 2): contribución por familia",
        "falla": "Riesgo de falla a 7 días: contribución por familia",
    }
    for name, item in explainers.items():
        expl = item["explanation"]
        save_summary_plot(expl, name)
        save_bar_plot(expl, name)
        save_waterfall_plot(expl, name, row=-1)
        grouped = grouped_importance(expl)
        save_grouped_plot(grouped, name, titles[name])
        global_importance(expl).to_csv(metrics_dir() / f"shap_global_{name}.csv", index=False)
        grouped.to_csv(metrics_dir() / f"shap_familias_{name}.csv", index=False)
        regimes = hydrological_regime(item["dates"], item["dates"].map(nino))
        importance_by_regime(expl, regimes).to_csv(metrics_dir() / f"shap_regimen_{name}.csv")
        local_contributions(expl, -1, top=10).to_csv(metrics_dir() / f"shap_local_{name}.csv", index=False)
        log.info("SHAP %s: %d observaciones explicadas; familia principal = %s",
                 name, len(item["X"]), grouped.iloc[0]["familia"])
    for feat, name in (("vol_pct", "oferta_regresor"), ("pred_aporte_14d", "oferta_clasificador")):
        expl = explainers[name]["explanation"]
        if feat in expl.feature_names:
            save_dependence_plot(expl, feat, name)

    # Comparación local con LIME para el regresor de oferta (último día).
    item = explainers["oferta_regresor"]
    lime_df = lime_explanation(item["explainer"].model.predict, item["X"], item["X"].iloc[-1], "regression")
    lime_df.to_csv(metrics_dir() / "lime_local_oferta_regresor.csv", index=False)
    log.info("Figuras SHAP guardadas en %s", figures_dir())


if __name__ == "__main__":
    run_all()
