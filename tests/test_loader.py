import numpy as np
import pandas as pd

from src.data.loader import generate_synthetic_data

REQUIRED = [
    "fecha", "nino34", "precipitacion_mm", "aportes_gwh", "caudal_m3s", "volumen_util_gwh", "volumen_util_pct",
    "nivel_embalse_msnm", "vertimientos_gwh", "generacion_gwh", "generacion_mw_media", "disponibilidad_mw",
    "precio_bolsa_cop_kwh", "precio_minimo_cop_kwh", "precio_oferta_cop_kwh", "oferta_al_minimo",
    "vibracion_mm_s", "temp_cojinete_c", "horas_desde_mant", "falla_forzada", "mantenimiento_programado",
]


def test_columns_and_window(df, cfg):
    assert set(REQUIRED).issubset(df.columns)
    assert df["fecha"].min() == pd.Timestamp(cfg["synthetic"]["start_date"])
    assert df["fecha"].max() == pd.Timestamp("2026-09-30")
    assert df["fecha"].is_monotonic_increasing and df["fecha"].diff().dropna().dt.days.eq(1).all()
    assert not df[REQUIRED].isna().any().any()


def test_physical_bounds(df, cfg):
    cap = cfg["synthetic"]["useful_capacity_gwh"]
    assert (df["volumen_util_gwh"] >= 0).all() and (df["volumen_util_gwh"] <= cap + 1e-6).all()
    assert (df["aportes_gwh"] > 0).all()
    max_daily = cfg["synthetic"]["installed_capacity_mw"] * 24 / 1000
    assert (df["generacion_gwh"] <= max_daily + 1e-6).all()
    assert (df["vertimientos_gwh"] >= 0).all()
    assert df["disponibilidad_mw"].between(0, cfg["synthetic"]["installed_capacity_mw"]).all()


def test_mass_balance(df):
    v = df["volumen_util_gwh"]
    resid = v.diff() - (df["aportes_gwh"] - df["generacion_gwh"] - df["vertimientos_gwh"])
    assert resid.iloc[1:].abs().max() < 1e-6


def test_bids_and_floor(df, cfg):
    eps = cfg["oferta_model"]["floor_epsilon"]
    floor = df["precio_oferta_cop_kwh"] <= df["precio_minimo_cop_kwh"] * (1 + eps)
    assert (floor.astype(int) == df["oferta_al_minimo"]).all()
    assert 0.05 < df["oferta_al_minimo"].mean() < 0.5


def test_reproducible(cfg):
    a = generate_synthetic_data(cfg)
    b = generate_synthetic_data(cfg)
    pd.testing.assert_frame_equal(a, b)


def test_failures_and_maintenance(df):
    assert df["falla_forzada"].sum() >= 10
    assert df["mantenimiento_programado"].sum() > 0
    # La vibración es mayor en los 7 días previos a una falla que en promedio.
    pre = sum(df["falla_forzada"].shift(-k).fillna(0) for k in range(1, 8)) > 0
    assert df.loc[pre, "vibracion_mm_s"].mean() > df["vibracion_mm_s"].mean()


def test_enso_dry_link(df):
    mo = df.set_index("fecha")[["nino34", "aportes_pct_media"]].resample("MS").mean()
    assert np.corrcoef(mo["nino34"].shift(2).iloc[2:], mo["aportes_pct_media"].iloc[2:])[0, 1] < 0
