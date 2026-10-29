"""Simulador de la API de XM y del archivo semanal de NOAA, para pruebas sin internet.

Responde con el mismo formato JSON que https://servapibi.xm.com.co a partir de los datos
sintéticos: kWh para energía, kW para potencia, COP/kWh para precios, 24 horas para métricas
horarias. Incluye otros recursos del SIN para que el proxy del precio mínimo funcione.

Uso fuera de pytest (ensayo completo del flujo XM):
    python -m tests.xm_mock    # corre el pipeline con --source xm contra el simulador
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import load_config
from src.data.loader import generate_synthetic_data

HOURS = [f"Hour{h:02d}" for h in range(1, 25)]
OTHER_RESOURCES = ["CHVR", "SCAR", "SOGA", "BETA", "URRA", "PRDO", "MIEL", "TSJR"]
# Otros ríos/embalses que XM devuelve cuando se consulta sin filtro (con su factor de escala).
OTHER_SERIES = {"BATATAS": 0.6, "CHUZA": 0.3, "PENOL": 1.4}
# Como en la API real, los listados usan códigos distintos a los de las series.
LIST_CODES = {"ListadoRios": ("GUAVGUAV", "GUAVIO"), "ListadoEmbalses": ("EGUAVIO1", "GUAVIO")}


def _catalog() -> list[dict]:
    m = load_config()["xm"]["metrics"]
    out = [{"MetricId": v["id"], "MetricName": k, "Entity": v["entity"], "Type": "DailyEntities"} for k, v in m.items()]
    return out


class XMMock:
    """Transporte compatible con XMClient(transport=...)."""

    def __init__(self, df: pd.DataFrame | None = None, drop_metrics: tuple[str, ...] = (),
                 bids_until: str | None = None):
        self.df = (df if df is not None else generate_synthetic_data()).set_index("fecha")
        self.bids_until = pd.Timestamp(bids_until) if bids_until else None
        self.drop = set(drop_metrics)
        self.calls: list[tuple[str, dict]] = []
        rng = np.random.default_rng(7)
        d = self.df
        # Otros recursos del SIN: uno siempre al mínimo; los demás al mínimo con prob. = fracción del sistema.
        self.others = {}
        for i, code in enumerate(OTHER_RESOURCES):
            if i == 0:
                p = d["precio_minimo_cop_kwh"].to_numpy()
            else:
                at_min = rng.random(len(d)) < d["fraccion_plantas_minimo"].to_numpy()
                above = d["precio_minimo_cop_kwh"].to_numpy() * rng.uniform(1.3, 6.0, len(d))
                p = np.where(at_min, d["precio_minimo_cop_kwh"].to_numpy(), above)
            self.others[code] = pd.Series(p, index=d.index)

    # ------------------------------------------------------------------
    def _daily_values(self, metric: str, code: str) -> pd.Series | None:
        d = self.df
        clim = d["aportes_gwh"] / (d["aportes_pct_media"] / 100)
        table = {
            ("AporEner", "GUAVIO"): d["aportes_gwh"] * 1e6,
            ("AporEnerMediHist", "GUAVIO"): clim * 1e6,
            ("AporCaudal", "GUAVIO"): d["caudal_m3s"],
            ("VoluUtilDiarEner", "GUAVIO"): d["volumen_util_gwh"] * 1e6,
            ("CapaUtilDiarEner", "GUAVIO"): d["capacidad_util_gwh"] * 1e6,
            ("VertEner", "GUAVIO"): d["vertimientos_gwh"] * 1e6,
        }
        if code in OTHER_SERIES:
            base = table.get((metric, "GUAVIO"))
            return None if base is None else base * OTHER_SERIES[code]
        return table.get((metric, code))

    def _hourly_values(self, metric: str, code: str) -> pd.Series | None:
        d = self.df
        if metric == "Gene" and code == "GVIO":
            return d["generacion_gwh"] * 1e6 / 24
        if metric == "DispoReal" and code == "GVIO":
            return d["disponibilidad_mw"] * 1e3
        if metric == "PrecOferDesp":
            if self.bids_until is not None:  # ofertas aún no publicadas después de esta fecha
                src = d["precio_oferta_cop_kwh"] if code == "GVIO" else self.others.get(code)
                return None if src is None else src[src.index <= self.bids_until]
            if code == "GVIO":
                return d["precio_oferta_cop_kwh"]
            return self.others.get(code)
        if metric == "PrecBolsNaci" and code == "SISTEMA":
            return d["precio_bolsa_cop_kwh"]
        if metric == "DemaCome" and code == "SISTEMA":
            return d["demanda_sin_gwh"] * 1e6 / 24
        return None

    def __call__(self, url: str, body: dict) -> dict:
        self.calls.append((url, body))
        period = url.rstrip("/").split("/")[-1].lower()
        metric = body.get("MetricId")
        if period == "lists":
            return self._lists(metric)
        if metric in self.drop:
            return {"Items": []}
        start, end = pd.Timestamp(body["StartDate"]), pd.Timestamp(body["EndDate"])
        if (end - start).days > 31:
            raise RuntimeError("La API de XM limita cada petición a ~30 días")
        entity = body.get("Entity")
        filters = [f.upper() for f in body.get("Filter") or []]
        if entity == "Sistema":
            codes = ["SISTEMA"]
        elif filters:
            codes = filters
        else:
            codes = ["GVIO"] + OTHER_RESOURCES if entity == "Recurso" else ["GUAVIO"] + list(OTHER_SERIES)
        days = pd.date_range(start, end, freq="D")
        items = []
        key = "HourlyEntities" if period == "hourly" else "DailyEntities"
        for day in days:
            ents = []
            for code in codes:
                s = self._hourly_values(metric, code) if period == "hourly" else self._daily_values(metric, code)
                if s is None or day not in s.index or pd.isna(s.loc[day]):
                    continue
                v = float(s.loc[day])
                values = {"code": code}
                if period == "hourly":
                    values.update({h: f"{v:.6f}" for h in HOURS})
                else:
                    values["Value"] = f"{v:.6f}"
                ents.append({"Id": f"{metric}_{code}", "Values": values})
            if ents:
                items.append({"Date": f"{day:%Y-%m-%d}", key: ents})
        return {"Items": items}

    def _lists(self, metric: str) -> dict:
        if metric == "ListadoMetricas":
            ents = [{"Id": "m", "Values": v} for v in _catalog()]
        elif metric == "ListadoRecursos":
            ents = [{"Id": "r", "Values": {"Code": "GVIO", "Name": "GUAVIO", "Type": "HIDRAULICA"}}] + \
                   [{"Id": "r", "Values": {"Code": c, "Name": c, "Type": "HIDRAULICA"}} for c in OTHER_RESOURCES]
        elif metric in LIST_CODES:
            code, name = LIST_CODES[metric]
            ents = [{"Id": "l", "Values": {"Code": code, "Name": name}}] + \
                   [{"Id": "l", "Values": {"Code": c[:4] + c[:4], "Name": c}} for c in OTHER_SERIES]
        else:
            ents = []
        return {"Items": [{"Date": "2026-10-01", "ListEntities": ents}]}


def noaa_weekly_text(df: pd.DataFrame) -> str:
    """Archivo semanal estilo CPC (wksst9120.for) con el Niño 3.4 del dataset."""
    s = df.set_index("fecha")["nino34"]
    weeks = pd.date_range(s.index.min(), s.index.max(), freq="W-WED")
    lines = [" Weekly SST data starts week centered on 3Jan1990", "",
             "                Nino1+2      Nino3        Nino34        Nino4",
             " Week          SST SSTA     SST SSTA     SST SSTA     SST SSTA"]
    for w in weeks:
        a = float(s.loc[w])
        lines.append(f" {w.strftime('%d%b%Y').upper()}     24.0{-0.3:4.1f}     26.0{0.1:4.1f}"
                     f"     27.0{a:4.1f}     28.6{0.2:4.1f}")
    return "\n".join(lines)


def install(monkeypatch=None, df: pd.DataFrame | None = None) -> XMMock:
    """Hace que XMClient y NOAA usen el simulador (con monkeypatch de pytest o de forma global)."""
    import src.data.noaa as noaa
    import src.data.xm_client as xc

    df = df if df is not None else generate_synthetic_data()
    mock = XMMock(df)
    text = noaa_weekly_text(df)

    def fake_transport(timeout: float = 60.0):
        return mock

    def fake_download(url: str, timeout: float = 60.0) -> str:
        return text

    if monkeypatch is not None:
        monkeypatch.setattr(xc, "requests_transport", fake_transport)
        monkeypatch.setattr(noaa, "_download", fake_download)
    else:
        xc.requests_transport = fake_transport
        noaa._download = fake_download
    return mock


if __name__ == "__main__":
    import sys

    from src.config import set_data_source

    # Fuente "xm_mock": mismo código que la API real, con carpetas propias (no toca la caché ni los modelos reales).
    install()
    set_data_source("xm_mock")
    from scripts.run_pipeline import main

    sys.argv = ["run_pipeline", "--force"] + sys.argv[1:]
    main()
