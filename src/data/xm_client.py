"""Cliente de la API pública de XM (servapibi.xm.com.co), la misma que alimenta Sinergox.

Formato de la API (POST, JSON):
    https://servapibi.xm.com.co/{hourly|daily|monthly|annual|lists}
    {"MetricId": "...", "StartDate": "YYYY-MM-DD", "EndDate": "YYYY-MM-DD", "Entity": "...", "Filter": [...]}

Respuesta:
    {"Items": [{"Date": "YYYY-MM-DD",
                "DailyEntities"|"HourlyEntities"|...: [{"Id": "...", "Values": {"code": "...", "Value": "..."
                                                                             | "Hour01": ..., "Hour24": ...}}]}]}

El cliente:
- divide las consultas en tramos mensuales (la API limita el rango por petición);
- guarda cada tramo en caché (data/external/xm_cache) y solo vuelve a pedir los meses recientes;
- reintenta con espera exponencial ante errores transitorios;
- devuelve DataFrames normalizados: columnas fecha, codigo, valor (o hora para datos horarios).

Uso rápido:
    python -m src.data.xm_client --check     # valida métricas y códigos de Guavio contra el catálogo de XM
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.config import load_config, xm_cache_dir
from src.utils.logging_utils import get_logger

log = get_logger(__name__)

BASE_URL = "https://servapibi.xm.com.co"
Transport = Callable[[str, dict], dict]
ENTITY_KEYS = {"daily": "DailyEntities", "hourly": "HourlyEntities", "monthly": "MonthlyEntities",
               "annual": "AnnualEntities", "lists": "ListEntities"}
HOURS = [f"Hour{h:02d}" for h in range(1, 25)]


class XMAPIError(RuntimeError):
    """Error al consultar la API de XM."""


def requests_transport(timeout: float = 60.0) -> Transport:
    """Transporte HTTP real basado en requests."""
    import requests

    session = requests.Session()
    session.headers.update({"Content-Type": "application/json", "Accept": "application/json",
                            "User-Agent": "guavio-xai/1.0 (prototipo academico)"})

    def _post(url: str, body: dict) -> dict:
        resp = session.post(url, data=json.dumps(body), timeout=timeout)
        if resp.status_code == 400:
            # XM responde 400 cuando la métrica/entidad no existe o no hay datos en el rango.
            raise XMAPIError(f"400 {resp.text[:200]}")
        resp.raise_for_status()
        return resp.json()

    return _post


@dataclass
class XMClient:
    transport: Transport | None = None
    cache_dir: Path | None = None
    use_cache: bool = True
    refresh_days: int = 45          # tramos que terminan a menos de N días de hoy se vuelven a pedir
    max_retries: int = 4
    workers: int = 6
    _catalog: pd.DataFrame | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        cfg = load_config().get("xm", {})
        if self.transport is None:
            self.transport = requests_transport(cfg.get("timeout_s", 60))
        if self.cache_dir is None:
            self.cache_dir = xm_cache_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.workers = cfg.get("workers", self.workers)
        self.refresh_days = cfg.get("refresh_days", self.refresh_days)

    # ------------------------------------------------------------------
    # Bajo nivel
    # ------------------------------------------------------------------
    def _post(self, period: str, body: dict) -> dict:
        url = f"{BASE_URL}/{period}"
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return self.transport(url, body)
            except XMAPIError as exc:
                msg = str(exc).lower()
                if "no encontr" in msg or "not found" in msg or "no existe" in msg:
                    return {"Items": []}
                last_err = exc
            except Exception as exc:  # errores de red o 5xx
                last_err = exc
            wait = 2 ** attempt
            log.warning("XM %s %s (intento %d/%d): %s. Reintento en %ds", period, body.get("MetricId"),
                        attempt + 1, self.max_retries, last_err, wait)
            time.sleep(wait)
        raise XMAPIError(f"No fue posible consultar {body.get('MetricId')} ({period}): {last_err}")

    @staticmethod
    def parse(payload: dict, period: str) -> pd.DataFrame:
        """Convierte la respuesta JSON en filas largas.

        Diario/mensual: fecha, codigo, valor. Horario: fecha, codigo, hora (1-24), valor.
        """
        key = ENTITY_KEYS[period]
        rows: list[dict[str, Any]] = []
        for item in payload.get("Items", []) or []:
            date = item.get("Date")
            for ent in item.get(key, []) or []:
                vals = ent.get("Values", ent)
                code = vals.get("code", vals.get("Code", vals.get("Name", "Sistema")))
                if period == "hourly":
                    for h, hk in enumerate(HOURS, start=1):
                        if hk in vals:
                            rows.append({"fecha": date, "codigo": code, "hora": h, "valor": vals[hk]})
                else:
                    rows.append({"fecha": date, "codigo": code, "valor": vals.get("Value", vals.get("value"))})
        cols = ["fecha", "codigo", "hora", "valor"] if period == "hourly" else ["fecha", "codigo", "valor"]
        df = pd.DataFrame(rows, columns=cols)
        if df.empty:
            return df
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
        df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
        df["codigo"] = df["codigo"].astype(str).str.strip().str.upper()
        return df.dropna(subset=["fecha"])

    @staticmethod
    def month_chunks(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        chunks = []
        cur = pd.Timestamp(start).normalize()
        end = pd.Timestamp(end).normalize()
        while cur <= end:
            month_end = (cur + pd.offsets.MonthEnd(0)).normalize()
            chunks.append((cur, min(month_end, end)))
            cur = month_end + pd.Timedelta(days=1)
        return chunks

    def _cache_path(self, period: str, metric: str, entity: str, filters: list[str], a: pd.Timestamp,
                    b: pd.Timestamp) -> Path:
        tag = hashlib.md5(",".join(sorted(filters)).encode()).hexdigest()[:8] if filters else "all"
        return self.cache_dir / f"{metric}__{entity}__{tag}__{period}__{a:%Y%m%d}_{b:%Y%m%d}.parquet"

    def _fetch_chunk(self, period: str, metric: str, entity: str, filters: list[str], a: pd.Timestamp,
                     b: pd.Timestamp) -> pd.DataFrame:
        path = self._cache_path(period, metric, entity, filters, a, b)
        recent = (pd.Timestamp.today().normalize() - b).days < self.refresh_days
        if self.use_cache and path.exists() and not recent:
            return pd.read_parquet(path)
        body = {"MetricId": metric, "StartDate": f"{a:%Y-%m-%d}", "EndDate": f"{b:%Y-%m-%d}",
                "Entity": entity, "Filter": filters}
        df = self.parse(self._post(period, body), period)
        if self.use_cache:
            df.to_parquet(path, index=False)
        return df

    # ------------------------------------------------------------------
    # Alto nivel
    # ------------------------------------------------------------------
    def fetch(self, metric: str, entity: str, start, end, filters: list[str] | None = None,
              period: str = "daily") -> pd.DataFrame:
        """Descarga una métrica en el rango [start, end] (en tramos mensuales, en paralelo)."""
        filters = [str(f) for f in (filters or [])]
        chunks = self.month_chunks(pd.Timestamp(start), pd.Timestamp(end))
        log.info("XM %s/%s %s: %d tramos mensuales", metric, entity, filters or "todos", len(chunks))
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            parts = list(pool.map(lambda ab: self._fetch_chunk(period, metric, entity, filters, *ab), chunks))
        parts = [p for p in parts if not p.empty]
        if not parts:
            return pd.DataFrame(columns=["fecha", "codigo", "valor"])
        return pd.concat(parts, ignore_index=True)

    def fetch_daily_series(self, metric: str, entity: str, start, end, filters: list[str] | None = None,
                           period: str = "daily", how: str = "sum", code: str | None = None) -> pd.Series:
        """Serie diaria (índice fecha). Para datos horarios agrega con `how` (sum, mean, median, first)."""
        df = self.fetch(metric, entity, start, end, filters, period)
        if df.empty:
            return pd.Series(dtype=float, name=metric)
        if code is not None:
            match = df["codigo"] == str(code).upper()
            if match.any():
                df = df[match]
            elif filters:
                # La API ya filtró del lado del servidor; el código devuelto puede tener otra forma.
                log.info("%s: XM devolvió códigos %s para el filtro %s; se usan tal cual.",
                         metric, sorted(df["codigo"].unique())[:5], filters)
            else:
                df = df[match]
        if period == "hourly":
            df = df.groupby(["fecha", "codigo"])["valor"].agg(how).reset_index()
        # Suma entre códigos (p. ej. varias unidades de un mismo recurso); con un solo código es el valor.
        s = df.groupby("fecha")["valor"].sum(min_count=1)
        return s.sort_index().rename(metric)

    def list_entities(self, metric: str, entity: str = "Sistema") -> pd.DataFrame:
        """Listados (ListadoMetricas, ListadoRecursos, ListadoEmbalses, ListadoRios)."""
        payload = self._post("lists", {"MetricId": metric, "Entity": entity})
        rows = []
        for item in payload.get("Items", []) or []:
            for ent in item.get("ListEntities", []) or []:
                vals = dict(ent.get("Values", ent))
                vals["Date"] = item.get("Date")
                rows.append(vals)
        return pd.DataFrame(rows)

    def catalog(self) -> pd.DataFrame:
        if self._catalog is None:
            self._catalog = self.list_entities("ListadoMetricas")
        return self._catalog


# ----------------------------------------------------------------------
# Validación del catálogo y de los códigos de la planta
# ----------------------------------------------------------------------
def _find_col(df: pd.DataFrame, *names: str) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    return None


def check_metrics(client: XMClient) -> pd.DataFrame:
    """Verifica que cada métrica configurada exista con su entidad; sugiere alternativas si no."""
    cfg = load_config()["xm"]
    cat = client.catalog()
    mid, ent, name = _find_col(cat, "MetricId"), _find_col(cat, "Entity"), _find_col(cat, "MetricName", "Name")
    rows = []
    for key, spec in cfg["metrics"].items():
        ok = bool(((cat[mid] == spec["id"]) & (cat[ent] == spec["entity"])).any()) if mid and ent else False
        suggestion = ""
        if not ok and name:
            words = [w for w in spec.get("keywords", []) if w]
            m = cat[name].astype(str).str.lower()
            hits = cat[m.apply(lambda s: all(w.lower() in s for w in words))] if words else cat.iloc[0:0]
            suggestion = "; ".join(f"{r[mid]}/{r[ent]}" for _, r in hits.head(5).iterrows())
        rows.append({"variable": key, "MetricId": spec["id"], "Entity": spec["entity"], "existe": ok,
                     "sugerencias": suggestion})
    return pd.DataFrame(rows)


def resolve_plant_codes(client: XMClient) -> dict[str, str]:
    """Busca los códigos de Guavio en los listados de recursos, embalses y ríos de XM."""
    plant = load_config()["xm"]["plant"]
    out = {"recurso": plant["recurso"], "embalse": plant["embalse"], "rio": plant["rio"]}
    needle = plant.get("search_name", "GUAVIO").upper()
    for key, metric in (("recurso", "ListadoRecursos"), ("embalse", "ListadoEmbalses"), ("rio", "ListadoRios")):
        try:
            df = client.list_entities(metric)
        except XMAPIError as exc:
            log.warning("No se pudo leer %s: %s", metric, exc)
            continue
        if df.empty:
            continue
        code_col = _find_col(df, "Code", "code")
        name_col = _find_col(df, "Name", "name", "Values_Name")
        if code_col is None:
            continue
        codes = df[code_col].astype(str).str.upper()
        if out[key].upper() in set(codes):
            continue
        if name_col is not None:
            hit = df[df[name_col].astype(str).str.upper().str.contains(needle)]
            if not hit.empty:
                log.info("Código de %s para %s resuelto como %s", key, needle, hit.iloc[0][code_col])
                out[key] = str(hit.iloc[0][code_col])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Utilidades del cliente de la API de XM.")
    parser.add_argument("--check", action="store_true", help="Valida métricas y códigos contra el catálogo de XM.")
    parser.add_argument("--list", metavar="TEXTO", help="Busca métricas cuyo nombre contenga TEXTO.")
    args = parser.parse_args()
    client = XMClient()
    if args.list:
        cat = client.catalog()
        name = _find_col(cat, "MetricName", "Name")
        hits = cat[cat[name].astype(str).str.lower().str.contains(args.list.lower())] if name else cat
        print(hits.drop(columns=[c for c in ("Date",) if c in hits]).to_string(index=False))
    if args.check or not args.list:
        print(check_metrics(client).to_string(index=False))
        print("Códigos de la planta:", resolve_plant_codes(client))


if __name__ == "__main__":
    main()
