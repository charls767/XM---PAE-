"""Pruebas del conector de la API de XM (Sinergox) y de NOAA con el simulador de tests/xm_mock.py."""
import numpy as np
import pandas as pd
import pytest

import src.data.noaa as noaa
from src.data.xm_client import XMClient, check_metrics, resolve_plant_codes
from src.data.xm_loader import build_xm_dataset
from tests.xm_mock import XMMock, noaa_weekly_text

START, END = "2023-01-01", "2023-06-30"


@pytest.fixture()
def mock(df):
    return XMMock(df)


@pytest.fixture()
def client(mock, tmp_path):
    return XMClient(transport=mock, cache_dir=tmp_path / "cache")


@pytest.fixture()
def patched_noaa(df, tmp_path, monkeypatch):
    text = noaa_weekly_text(df)
    monkeypatch.setattr(noaa, "_download", lambda url, timeout=60.0: text)
    monkeypatch.setattr(noaa, "xm_cache_dir", lambda: tmp_path)
    return text


def test_parse_daily_and_hourly():
    daily = {"Items": [{"Date": "2024-01-02", "DailyEntities": [{"Id": "x", "Values": {"code": "guavio", "Value": "1500.5"}}]}]}
    d = XMClient.parse(daily, "daily")
    assert list(d.columns) == ["fecha", "codigo", "valor"]
    assert d.iloc[0]["codigo"] == "GUAVIO" and d.iloc[0]["valor"] == 1500.5
    hourly = {"Items": [{"Date": "2024-01-02", "HourlyEntities": [
        {"Id": "y", "Values": {"code": "GVIO", **{f"Hour{h:02d}": str(h) for h in range(1, 25)}}}]}]}
    h = XMClient.parse(hourly, "hourly")
    assert len(h) == 24 and h["valor"].sum() == sum(range(1, 25))
    assert XMClient.parse({"Items": []}, "daily").empty


def test_month_chunks():
    chunks = XMClient.month_chunks(pd.Timestamp("2024-01-15"), pd.Timestamp("2024-03-10"))
    assert chunks[0] == (pd.Timestamp("2024-01-15"), pd.Timestamp("2024-01-31"))
    assert chunks[-1] == (pd.Timestamp("2024-03-01"), pd.Timestamp("2024-03-10"))
    assert all((b - a).days <= 30 for a, b in chunks)


def test_fetch_units_and_cache(client, mock, df):
    s = client.fetch_daily_series("AporEner", "Rio", START, END, ["GUAVIO"], code="GUAVIO")
    ref = df.set_index("fecha").loc[START:END, "aportes_gwh"] * 1e6
    np.testing.assert_allclose(s.values, ref.values, rtol=1e-6)
    n = len(mock.calls)
    client.fetch_daily_series("AporEner", "Rio", START, END, ["GUAVIO"], code="GUAVIO")
    assert len(mock.calls) == n, "la segunda consulta debe salir de la caché"
    g = client.fetch_daily_series("Gene", "Recurso", START, END, ["GVIO"], period="hourly", how="sum", code="GVIO")
    np.testing.assert_allclose(g.values, df.set_index("fecha").loc[START:END, "generacion_gwh"].values * 1e6, rtol=1e-6)


def test_catalog_and_codes(client):
    chk = check_metrics(client)
    assert chk["existe"].all()
    end = pd.Timestamp("2026-09-28")
    assert resolve_plant_codes(client, end=end) == {"recurso": "GVIO", "embalse": "GUAVIO", "rio": "GUAVIO"}


def test_list_code_differs_from_series_code(client, monkeypatch):
    """Caso real: ListadoRios dice GUAVGUAV, pero la serie AporEner no responde a ese filtro."""
    from src.config import load_config

    filtered = client.fetch("AporEner", "Rio", "2026-09-01", "2026-09-28", filters=["GUAVGUAV"])
    assert filtered.empty
    plant = load_config()["xm"]["plant"]
    monkeypatch.setitem(plant, "rio", "GUAVGUAV")
    monkeypatch.setitem(plant, "embalse", "auto")
    codes = resolve_plant_codes(client, end=pd.Timestamp("2026-09-28"))
    assert codes["rio"] == "GUAVIO" and codes["embalse"] == "GUAVIO"


def test_unknown_plant_raises(client, monkeypatch):
    from src.config import load_config

    plant = load_config()["xm"]["plant"]
    monkeypatch.setitem(plant, "search_name", "INEXISTENTE")
    monkeypatch.setitem(plant, "rio", "NADA")
    with pytest.raises(RuntimeError, match="Códigos disponibles"):
        resolve_plant_codes(client, end=pd.Timestamp("2026-09-28"))


def test_noaa_parsers(df, patched_noaa):
    w = noaa.parse_weekly(patched_noaa)
    ref = df.set_index("fecha")["nino34"].reindex(w.index)
    np.testing.assert_allclose(w.values, ref.values, atol=0.051)
    oni = noaa.parse_oni("SEAS  YR   TOTAL   ANOM\n DJF 2024  28.6   1.8\n JFM 2024  28.3   1.5\n")
    assert oni.loc["2024-01-15"] == 1.8 and oni.loc["2024-02-15"] == 1.5
    daily = noaa.load_nino34(START, END)
    assert daily.index.min() == pd.Timestamp(START) and daily.notna().all()


def test_build_dataset_matches_source(client, df, patched_noaa):
    out = build_xm_dataset(client=client, start=START, end=END)
    ref = df.set_index("fecha").loc[START:END]
    o = out.set_index("fecha")
    for col in ("aportes_gwh", "volumen_util_gwh", "generacion_gwh", "precio_oferta_cop_kwh", "precio_bolsa_cop_kwh"):
        np.testing.assert_allclose(o[col].values, ref[col].values, rtol=1e-5, err_msg=col)
    np.testing.assert_allclose(o["volumen_util_pct"].values, ref["volumen_util_pct"].values, rtol=1e-5)
    # Proxy del precio mínimo = piso real (un recurso del simulador siempre oferta al mínimo).
    np.testing.assert_allclose(o["precio_minimo_cop_kwh"].values, ref["precio_minimo_cop_kwh"].values, rtol=1e-6)
    assert (o["oferta_al_minimo"] == ref["oferta_al_minimo"]).all()
    assert o["fraccion_plantas_minimo"].between(0, 1).all()
    assert not o["es_sintetico"].any()
    assert "vibracion_mm_s" not in out and "precipitacion_mm" not in out


def test_optional_and_required_metrics(df, tmp_path, patched_noaa):
    m = XMMock(df, drop_metrics=("VertEner", "AporCaudal"))
    out = build_xm_dataset(client=XMClient(transport=m, cache_dir=tmp_path / "c1"), start=START, end=END)
    assert (out["vertimientos_gwh"] == 0).all()
    m2 = XMMock(df, drop_metrics=("AporEner",))
    with pytest.raises(RuntimeError):
        build_xm_dataset(client=XMClient(transport=m2, cache_dir=tmp_path / "c2"), start=START, end=END)


def test_retry_then_fail(tmp_path, monkeypatch):
    import src.data.xm_client as xc

    monkeypatch.setattr(xc.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def flaky(url, body):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("red caída")
        return {"Items": []}

    c = XMClient(transport=flaky, cache_dir=tmp_path)
    assert c._post("daily", {"MetricId": "AporEner"}) == {"Items": []}

    def dead(url, body):
        raise ConnectionError("sin red")

    with pytest.raises(xc.XMAPIError):
        XMClient(transport=dead, cache_dir=tmp_path)._post("daily", {"MetricId": "AporEner"})


def test_unpublished_bids_are_predictable(df, tmp_path, patched_noaa):
    """Días sin oferta publicada: variables completas y objetivo NaN (se predicen, no se entrenan)."""
    from src.data.features import build_oferta_dataset, oferta_feature_columns

    m = XMMock(df, bids_until="2023-05-31")
    out = build_xm_dataset(client=XMClient(transport=m, cache_dir=tmp_path / "c3"), start="2022-10-01", end=END)
    ds = build_oferta_dataset(out, None, "E1_observada")
    june = ds[ds.fecha >= "2023-06-01"]
    assert june["y_norm"].isna().all() and june["q_floor"].isna().all()
    assert june[oferta_feature_columns(ds)].notna().all().all()
    # El piso de junio se arrastra del último mes con ofertas (sin mirar al futuro).
    assert (out.set_index("fecha").loc["2023-06", "precio_minimo_cop_kwh"]
            == out.set_index("fecha").loc["2023-05-31", "precio_minimo_cop_kwh"]).all()
