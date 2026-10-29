"""Pruebas de la vista segura por latencia (prevención de data leakage)."""
import numpy as np
import pandas as pd

from src.data.features import (
    build_aportes_features,
    build_aportes_targets,
    build_falla_dataset,
    build_oferta_dataset,
    climatology,
    family_of,
    last_published_month_stats,
)


# Variables conocidas de antemano (el precio mínimo regulatorio se publica antes del mes).
KNOWN_IN_ADVANCE = {"precio_minimo_cop_kwh", "capacidad_util_gwh"}


def _perturb_future(df: pd.DataFrame, after: pd.Timestamp) -> pd.DataFrame:
    """Copia de df donde todo lo posterior a `after` se reemplaza por ruido."""
    d = df.copy()
    m = d["fecha"] > after
    rng = np.random.default_rng(1)
    for c in d.select_dtypes("number").columns.difference(list(KNOWN_IN_ADVANCE)):
        d[c] = d[c].astype(float)
        d.loc[m, c] = d.loc[m, c] * rng.uniform(0.2, 3.0, m.sum())
    return d


def test_aportes_features_use_only_past(df):
    o = pd.Timestamp("2022-06-15")
    a = build_aportes_features(df)
    b = build_aportes_features(_perturb_future(df, o))
    i = df.index[df.fecha == o][0]
    pd.testing.assert_series_equal(a.loc[i], b.loc[i])


def test_aportes_targets_are_future(df):
    Y = build_aportes_targets(df, [1, 7])
    assert Y["y_h1"].iloc[10] == df["aportes_gwh"].iloc[11]
    assert Y["y_h7"].iloc[10] == df["aportes_gwh"].iloc[17]


def test_oferta_features_respect_cutoff(df):
    """Para el día t, cambiar datos desde t-1 en adelante no altera las variables de mercado/hidrología."""
    t = pd.Timestamp("2023-03-20")
    a = build_oferta_dataset(df, None, "E1_observada")
    b = build_oferta_dataset(_perturb_future(df, t - pd.Timedelta(days=2)), None, "E1_observada")
    i = df.index[df.fecha == t][0]
    cols = [c for c in a.columns if c not in ("fecha", "y_norm", "q_floor", "precio_oferta_cop_kwh", "pmin",
                                              "disp_mw", "doy_sin", "doy_cos", "mes", "dow")]
    pd.testing.assert_series_equal(a.loc[i, cols], b.loc[i, cols], check_names=False)


def test_published_month_lag(df):
    stats = last_published_month_stats(df, 0.01)
    d = df.set_index("fecha")
    y = d["precio_oferta_cop_kwh"] / d["precio_minimo_cop_kwh"]
    # El 1 de marzo (corte 28 feb) solo está publicado enero; el 2 de marzo ya está febrero.
    i1 = df.index[df.fecha == "2023-03-01"][0]
    i2 = df.index[df.fecha == "2023-03-02"][0]
    assert np.isclose(stats.loc[i1, "oferta_mes_media"], y["2023-01"].mean())
    assert np.isclose(stats.loc[i2, "oferta_mes_media"], y["2023-02"].mean())


def test_climatology_ignores_test_period(df):
    c1 = climatology(df, "aportes_gwh", "2020-12-31")
    d2 = df.copy()
    d2.loc[d2.fecha > "2020-12-31", "aportes_gwh"] *= 10
    c2 = climatology(d2, "aportes_gwh", "2020-12-31")
    pd.testing.assert_series_equal(c1, c2)


def test_e4_uses_true_future(df):
    ds = build_oferta_dataset(df, None, "E4_cota_superior")
    i = df.index[df.fecha == "2021-05-10"][0]
    assert np.isclose(ds.loc[i, "pred_aporte_t"], df.loc[i, "aportes_gwh"])


def test_falla_target(df):
    ds = build_falla_dataset(df, 7)
    i = int(df.index[df.falla_forzada == 1][0])
    assert ds.loc[i - 1, "y_falla"] == 1
    assert ds.loc[i - 7, "y_falla"] == 1


def test_families():
    assert family_of("pred_aporte_14d") == "Hidrología pronosticada"
    assert family_of("vol_pct") == "Hidrología observada"
    assert family_of("vib_mean7") == "Condición de máquina"
