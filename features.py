"""Ingeniería de características con vista segura por latencia.

Convenciones de tiempo:
- Modelo de aportes: origen `o` = último día con hidrología publicada. Se predice
  A(o + h). En operación, para ofertar el día t se usa o = t - 2 (hidrología D-1,
  corte de información a las 18:00 de t-1), de modo que A(t) corresponde a h = 2.
- Modelo de oferta: cada fila es el día de la oferta t. Hidrología y mercado
  entran con rezago de 2 días; las ofertas propias solo desde el último lote
  mensual publicado (M+1).
- Modelo de falla: cada fila es el día t (telemetría propia D-0). El objetivo es
  la ocurrencia de una falla forzada en (t, t + H].
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import load_config

# Familias para agrupar SHAP (sección 9 de la propuesta).
FEATURE_FAMILIES = {
    "aportes_": "Hidrología observada",
    "vol_": "Hidrología observada",
    "vert_": "Hidrología observada",
    "pred_": "Hidrología pronosticada",
    "precip_": "Clima",
    "nino": "Clima",
    "temp_amb": "Clima",
    "bolsa_": "Mercado",
    "demanda_": "Mercado",
    "frac_min": "Mercado",
    "oferta_mes": "Oferta propia publicada",
    "pmin": "Mercado",
    "gen_": "Operación",
    "disp_": "Operación",
    "vib_": "Condición de máquina",
    "tcoj_": "Condición de máquina",
    "horas_": "Condición de máquina",
    "carga_": "Operación",
    "doy_": "Calendario",
    "mes": "Calendario",
    "dow": "Calendario",
    "h_": "Calendario",
}


def family_of(feature: str) -> str:
    for prefix, fam in FEATURE_FAMILIES.items():
        if feature.startswith(prefix):
            return fam
    return "Otras"


def _calendar(dates: pd.Series, prefix: str = "") -> pd.DataFrame:
    doy = dates.dt.dayofyear
    return pd.DataFrame({
        f"{prefix}doy_sin": np.sin(2 * np.pi * doy / 365.25),
        f"{prefix}doy_cos": np.cos(2 * np.pi * doy / 365.25),
        f"{prefix}mes": dates.dt.month,
        f"{prefix}dow": dates.dt.dayofweek,
    }, index=dates.index)


def climatology(df: pd.DataFrame, col: str, end_date: str) -> pd.Series:
    """Media histórica por día del año calculada SOLO con datos hasta end_date.

    Se suaviza con ventana circular de 15 días y se devuelve alineada a todas
    las filas de df (incluidas las posteriores a end_date).
    """
    train = df[df["fecha"] <= end_date]
    by_doy = train.groupby(train["fecha"].dt.dayofyear)[col].mean().reindex(range(1, 367)).interpolate().bfill().ffill()
    padded = pd.concat([by_doy.iloc[-7:], by_doy, by_doy.iloc[:7]])
    smooth = padded.rolling(15, center=True).mean().iloc[7:-7]
    smooth.index = range(1, 367)
    return df["fecha"].dt.dayofyear.map(smooth)


# --------------------------------------------------------------------------
# Componente 1: aportes
# --------------------------------------------------------------------------
def build_aportes_features(df: pd.DataFrame, include_climate: bool = True) -> pd.DataFrame:
    """Variables conocidas en el origen o (todas con información <= o)."""
    cfg = load_config()
    a = df["aportes_gwh"]
    X = pd.DataFrame(index=df.index)
    X["aportes_lag0"] = a
    for lag in cfg["features"]["lags"]:
        X[f"aportes_lag{lag}"] = a.shift(lag)
    for w in cfg["features"]["rolling_windows"]:
        X[f"aportes_mean{w}"] = a.rolling(w).mean()
    X["aportes_std7"] = a.rolling(7).std()
    X["aportes_max7"] = a.rolling(7).max()
    X["aportes_tend_7_30"] = X["aportes_mean7"] - X["aportes_mean30"]
    X["vol_pct"] = df["volumen_util_pct"]
    X["vol_delta7"] = df["volumen_util_pct"].diff(7)
    X["vert_sum7"] = df["vertimientos_gwh"].rolling(7).sum()
    X["gen_mean7"] = df["generacion_gwh"].rolling(7).mean()
    X = X.join(_calendar(df["fecha"]))
    if include_climate:
        p = df["precipitacion_mm"]
        X["precip_sum3"] = p.rolling(3).sum()
        X["precip_sum7"] = p.rolling(7).sum()
        X["precip_sum30"] = p.rolling(30).sum()
        for m in cfg["features"]["nino_lags_months"]:
            X[f"nino34_lag{m}m"] = df["nino34"].shift(30 * m)
    return X


def build_aportes_targets(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    a = df["aportes_gwh"]
    return pd.DataFrame({f"y_h{h}": a.shift(-h) for h in horizons}, index=df.index)


# --------------------------------------------------------------------------
# Componente 2: oferta
# --------------------------------------------------------------------------
def last_published_month_stats(df: pd.DataFrame, eps: float) -> pd.DataFrame:
    """Estadísticos de la oferta propia del último mes publicado al corte de t-1.

    El lote del mes m se publica el día 1 del mes m+1. Para el día t, el corte es
    t-1; el lote está disponible si su fecha de publicación <= t-1.
    """
    d = df[["fecha", "precio_oferta_cop_kwh", "precio_minimo_cop_kwh"]].copy()
    d["y_norm"] = d["precio_oferta_cop_kwh"] / d["precio_minimo_cop_kwh"]
    d["floor"] = (d["precio_oferta_cop_kwh"] <= d["precio_minimo_cop_kwh"] * (1 + eps)).astype(int)
    d["period"] = d["fecha"].dt.to_period("M")
    monthly = d.groupby("period").agg(oferta_mes_media=("y_norm", "mean"),
                                      oferta_mes_std=("y_norm", "std"),
                                      oferta_mes_frac_min=("floor", "mean"))
    cutoff = df["fecha"] - pd.Timedelta(days=1)
    # Último mes m tal que primer día de m+1 <= cutoff  <=>  m <= periodo(cutoff) - 1
    avail_period = cutoff.dt.to_period("M") - 1
    return monthly.reindex(avail_period).set_index(df.index)


def build_oferta_dataset(df: pd.DataFrame, aportes_pred: pd.DataFrame | None, experiment: str) -> pd.DataFrame:
    """Dataset del modelo de oferta para el experimento indicado.

    experiment:
      E1_observada: hidrología observada (réplica del artículo).
      E2_observada_mas_pronostico: E1 + aportes pronosticados fuera de muestra.
      E4_cota_superior: E1 + aportes futuros reales (solo diagnóstico).
    aportes_pred: DataFrame indexado por fecha de ORIGEN con columnas p50_h{h}.
    """
    cfg = load_config()
    eps = cfg["oferta_model"]["floor_epsilon"]
    L = 2  # rezago efectivo de hidrología y mercado respecto al día de oferta
    out = pd.DataFrame({"fecha": df["fecha"]})
    a = df["aportes_gwh"]
    for lag in [0, 1, 2, 3, 5, 7]:
        out[f"aportes_lag{lag + L}"] = a.shift(lag + L)
    for w in (7, 30):
        out[f"aportes_mean{w}"] = a.rolling(w).mean().shift(L)
    out["aportes_pct_media_mean7"] = df["aportes_pct_media"].rolling(7).mean().shift(L)
    out["vol_pct"] = df["volumen_util_pct"].shift(L)
    out["vol_delta7"] = df["volumen_util_pct"].diff(7).shift(L)
    out["vert_sum7"] = df["vertimientos_gwh"].rolling(7).sum().shift(L)
    out["gen_mean7"] = df["generacion_gwh"].rolling(7).mean().shift(L)
    out["disp_mw"] = df["disponibilidad_mw"].shift(1)
    out["bolsa_lag"] = df["precio_bolsa_cop_kwh"].shift(L)
    out["bolsa_mean7"] = df["precio_bolsa_cop_kwh"].rolling(7).mean().shift(L)
    out["bolsa_rel_pmin"] = out["bolsa_mean7"] / df["precio_minimo_cop_kwh"]
    out["demanda_mean7"] = df["demanda_sin_gwh"].rolling(7).mean().shift(L)
    out["frac_min_lag"] = df["fraccion_plantas_minimo"].shift(L)
    out["frac_min_mean7"] = df["fraccion_plantas_minimo"].rolling(7).mean().shift(L)
    out["pmin"] = df["precio_minimo_cop_kwh"]
    out["nino34_lag1m"] = df["nino34"].shift(30)
    out = out.join(last_published_month_stats(df, eps))
    out = out.join(_calendar(df["fecha"]))

    if experiment in ("E2_observada_mas_pronostico", "E4_cota_superior"):
        clim = climatology(df, "aportes_gwh", cfg["splits"]["train_end"])
        by_doy = clim.groupby(df["fecha"].dt.dayofyear).first()
        # Media climatológica de los días t..t+13 (calendario conocido de antemano, sin datos futuros).
        clim14 = sum((df["fecha"] + pd.Timedelta(days=k)).dt.dayofyear.map(by_doy) for k in range(14)) / 14
        if experiment == "E4_cota_superior":
            fut = pd.DataFrame({f"p50_h{h}": a.shift(-h) for h in range(1, 16)}, index=df.index)
        else:
            fut = aportes_pred.reindex(df["fecha"]).set_index(df.index)
        # Pronósticos emitidos en el origen t-2 -> se desplazan 2 filas.
        f = fut.shift(L)
        out["pred_aporte_t"] = f["p50_h2"]
        out["pred_aporte_7d"] = f[[f"p50_h{h}" for h in range(2, 9)]].mean(axis=1)
        out["pred_aporte_14d"] = f[[f"p50_h{h}" for h in range(2, 16)]].mean(axis=1)
        out["pred_anomalia_14d"] = out["pred_aporte_14d"] / clim14 - 1
        sum15 = f[[f"p50_h{h}" for h in range(1, 16)]].sum(axis=1)
        vcap = df["capacidad_util_gwh"]
        vol_proj = (df["volumen_util_gwh"].shift(L) + sum15 - 15 * out["gen_mean7"]).clip(lower=0) / vcap
        out["pred_vol_proy_14d"] = 100 * vol_proj.clip(upper=1.0)

    out["y_norm"] = df["precio_oferta_cop_kwh"] / df["precio_minimo_cop_kwh"]
    out["q_floor"] = (df["precio_oferta_cop_kwh"] <= df["precio_minimo_cop_kwh"] * (1 + eps)).astype(int)
    out["precio_oferta_cop_kwh"] = df["precio_oferta_cop_kwh"]
    return out


OFERTA_NON_FEATURES = ["fecha", "y_norm", "q_floor", "precio_oferta_cop_kwh"]


def oferta_feature_columns(dataset: pd.DataFrame) -> list[str]:
    return [c for c in dataset.columns if c not in OFERTA_NON_FEATURES]


# --------------------------------------------------------------------------
# Modelo complementario: riesgo de falla (mantenimiento predictivo)
# --------------------------------------------------------------------------
def build_falla_dataset(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    out = pd.DataFrame({"fecha": df["fecha"]})
    vib, tc = df["vibracion_mm_s"], df["temp_cojinete_c"]
    load = df["generacion_gwh"] / (df["disponibilidad_mw"].clip(lower=1) * 24 / 1000)
    out["vib_actual"] = vib
    out["vib_mean7"] = vib.rolling(7).mean()
    out["vib_mean30"] = vib.rolling(30).mean()
    out["vib_max7"] = vib.rolling(7).max()
    out["vib_pend7"] = vib.rolling(7).mean() - vib.rolling(7).mean().shift(7)
    out["tcoj_actual"] = tc
    out["tcoj_mean7"] = tc.rolling(7).mean()
    out["tcoj_max7"] = tc.rolling(7).max()
    out["tcoj_pend7"] = tc.rolling(7).mean() - tc.rolling(7).mean().shift(7)
    out["carga_actual"] = load
    out["carga_mean7"] = load.rolling(7).mean()
    out["vib_por_carga"] = out["vib_mean7"] / (0.3 + out["carga_mean7"])
    out["tcoj_menos_amb"] = out["tcoj_mean7"] - df["temp_ambiente_c"].rolling(7).mean()
    out["horas_desde_mant"] = df["horas_desde_mant"]
    out["temp_amb_mean7"] = df["temp_ambiente_c"].rolling(7).mean()
    out["disp_unidades"] = df["unidades_disponibles"]
    # Objetivo: falla forzada en (t, t + horizon]
    fut = sum(df["falla_forzada"].shift(-k) for k in range(1, horizon + 1))
    out["y_falla"] = (fut > 0).astype(float)
    out.loc[fut.isna(), "y_falla"] = np.nan
    return out


FALLA_NON_FEATURES = ["fecha", "y_falla"]
