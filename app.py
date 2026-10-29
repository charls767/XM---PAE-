"""Dashboard de alerta temprana hidroenergética · Central Guavio (prototipo).

Ejecutar:  streamlit run app.py

Prototipo de apoyo analítico: muestra pronósticos, estimaciones y explicaciones.
No envía ofertas ni recomienda un precio a ofertar.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import streamlit as st

from dashboards import charts as ch
from dashboards.data_access import (
    TERCILE_LABELS,
    artifacts_ready,
    build_alerts,
    deficit_probability,
    drift_table,
    forecast_path,
    load_bundle,
    offer_estimate,
    top_alert,
)
from src.explainability.explainer import (
    build_explainers,
    grouped_importance,
    hydrological_regime,
    importance_by_regime,
    local_contributions,
    local_grouped,
)
from src.config import data_source, set_data_source
from src.models import evaluate as ev

st.set_page_config(page_title="Alerta temprana · Guavio", page_icon="💧", layout="wide")

CSS = """
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
.kpi {border: 1px solid #e6e5e1; border-radius: 10px; padding: 10px 12px; background: #fcfcfb; min-height: 118px;}
.kpi .lbl {font-size: 0.80rem; color: #52514e; display: flex; justify-content: space-between; align-items: center;}
.kpi .val {font-size: 1.55rem; font-weight: 600; color: #0b0b0b; line-height: 1.25;}
.kpi .unit {font-size: 0.85rem; color: #52514e; font-weight: 400;}
.kpi .sub {font-size: 0.78rem; color: #52514e;}
.dot {display:inline-block; width:10px; height:10px; border-radius:50%; margin-left:6px;}
.arrow {text-align:center; font-size: 1.4rem; color:#9a9893; padding-top: 38px;}
.seal {display:inline-block; font-size:0.72rem; border:1px solid #52514e; border-radius:4px; padding:1px 6px;
       color:#52514e; margin-left:8px; vertical-align: middle;}
.why {border: 1px solid #e6e5e1; border-radius: 10px; padding: 12px 14px; background: #fcfcfb;}
.why .row {display:flex; justify-content:space-between; font-size:0.85rem; margin-top:6px;}
.bar {height:6px; border-radius:3px; margin-top:3px;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)
STATUS = {"verde": ch.GOOD, "ambar": ch.WARNING, "rojo": ch.CRITICAL, "gris": "#9a9893"}
STATUS_TXT = {"verde": "normal", "ambar": "atención", "rojo": "crítico", "gris": "informativo"}


# ---------------------------------------------------------------------------
# Carga en caché
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando datos y modelos...")
def get_bundle(source: str):
    set_data_source(source)
    return load_bundle()


@st.cache_resource(show_spinner="Preparando explicadores SHAP...")
def _explainers(source: str):
    set_data_source(source)
    return build_explainers()


def get_explainers():
    return _explainers(SOURCE)


@st.cache_resource(show_spinner="Calculando valores SHAP de la ventana...")
def _window_explanation(source: str, model: str, end: str, days: int):
    item = _explainers(source)[model]
    dates = item["dates"].reset_index(drop=True)
    X = item["X"].reset_index(drop=True)
    m = (dates <= pd.Timestamp(end)) & (dates > pd.Timestamp(end) - pd.Timedelta(days=days))
    return item["explainer"].explain(X[m.values]), dates[m.values]


def window_explanation(model: str, end: str, days: int):
    return _window_explanation(SOURCE, model, end, days)


def local_explanation(model: str, day: pd.Timestamp, X_override: pd.DataFrame | None = None):
    item = get_explainers()[model]
    if X_override is not None:
        return item["explainer"].explain(X_override[item["X"].columns])
    m = (item["dates"] == day).values
    if not m.any():
        return None
    return item["explainer"].explain(item["X"][m])


def _ready(source: str) -> bool:
    set_data_source(source)
    return artifacts_ready()


# Fuente de datos: la de la configuración por defecto; se puede cambiar si hay modelos de ambas.
# Se fija una sola vez por proceso: set_data_source() modifica el entorno en cada recarga.
DEFAULT_SOURCE = os.environ.setdefault("GUAVIO_DEFAULT_SOURCE", data_source())
available = [s_ for s_ in ("xm", "synthetic", "xm_mock") if _ready(s_)]
if not available:
    st.error("No se encontraron modelos entrenados. Ejecute primero:\n\n"
             "`python scripts/run_pipeline.py --source xm --force`  (datos de la API de XM)\n\n"
             "o `python scripts/run_pipeline.py --source synthetic --force`  (simulador, sin internet)")
    st.stop()
SOURCE_LABELS = {"xm": "API XM (Sinergox)", "synthetic": "Simulador (sintético)",
                 "xm_mock": "Simulador de la API XM (pruebas)"}
with st.sidebar:
    SOURCE = st.radio("Fuente de datos", available,
                      index=available.index(DEFAULT_SOURCE) if DEFAULT_SOURCE in available else 0,
                      format_func=SOURCE_LABELS.get, horizontal=True, key="source_widget",
                      help="Solo aparecen las fuentes con modelos entrenados.")
if DEFAULT_SOURCE not in available:
    st.info(f"No hay modelos entrenados para la fuente '{DEFAULT_SOURCE}'. Se muestra '{SOURCE}'. "
            f"Para entrenarla: `python scripts/run_pipeline.py --source {DEFAULT_SOURCE} --force`.")
set_data_source(SOURCE)

B = get_bundle(SOURCE)
cfg = B["cfg"]
IS_XM = B["source"].startswith("xm")
HAS_MACHINE = B["has_machine"]
SEAL = {"xm": "DATOS XM · SINERGOX", "synthetic": "DATOS SINTÉTICOS", "xm_mock": "SIMULADOR API XM"}[SOURCE]
raw = B["raw"]
ofp_all = B["oferta_pred"]

# ---------------------------------------------------------------------------
# Barra lateral: filtros globales
# ---------------------------------------------------------------------------
min_day = B["oferta_ds"].fecha.min().date()
max_day = min(B["oferta_ds"].fecha.max(), B["aportes_pred"].fecha.max()).date()
if "pending_day" in st.session_state:
    st.session_state["date_widget"] = st.session_state.pop("pending_day")
if ("date_widget" not in st.session_state or st.session_state.get("_source") != SOURCE
        or not min_day <= st.session_state["date_widget"] <= max_day):
    st.session_state["date_widget"] = max_day
st.session_state["_source"] = SOURCE

with st.sidebar:
    st.markdown("### Filtros")
    day = pd.Timestamp(st.date_input("Fecha de la oferta (día t)", min_value=min_day, max_value=max_day,
                                     key="date_widget", format="YYYY-MM-DD"))
    scenario = st.selectbox("Escenario hidrológico", list(cfg["scenarios"].keys()), index=0,
                            help="Análisis de sensibilidad: multiplica los aportes pronosticados. No es un pronóstico.")
    factor = float(cfg["scenarios"][scenario])
    window = st.select_slider("Ventana histórica", options=[90, 180, 365, 730], value=180,
                              format_func=lambda d: f"{d} días")
    st.markdown("---")
    st.markdown("### Simulación en tiempo real")
    play = st.toggle("Reproducir día a día", value=False,
                     help="Avanza la fecha de análisis para simular la llegada diaria de datos.")
    speed = st.slider("Segundos por día", 1, 10, 3, disabled=not play)
    st.markdown("---")
    st.caption(cfg["project"]["disclaimer"])
    if IS_XM:
        st.caption(f"Datos: API pública de XM (Sinergox) y NOAA (Niño 3.4), hasta el {raw.fecha.max():%d-%b-%Y}. "
                   "El precio mínimo de oferta es un proxy calculado con las ofertas del SIN.")
    else:
        st.caption("Datos SINTÉTICOS generados por `src/data/loader.py`. Para datos reales: "
                   "`python scripts/run_pipeline.py --source xm --force`.")

origin = day - pd.Timedelta(days=2)       # último día con hidrología publicada al corte de t-1
start = day - pd.Timedelta(days=window)
rawi = raw.set_index("fecha")

# ---------------------------------------------------------------------------
# Cálculos del día
# ---------------------------------------------------------------------------
path = forecast_path(B, origin, factor)
offer = offer_estimate(B, day, factor)
offer_prev = offer_estimate(B, day - pd.Timedelta(days=1), factor)
fp = B["falla_pred"]
p_falla = None
if HAS_MACHINE:
    p_falla_row = fp[fp.fecha == day - pd.Timedelta(days=1)]
    p_falla = float(p_falla_row.p_falla.iloc[0]) if len(p_falla_row) else None
drift = drift_table(B, day)
alerts = build_alerts(B, day, path, offer, offer_prev, p_falla, drift)
main_alert = top_alert(alerts)

# ---------------------------------------------------------------------------
# Encabezado
# ---------------------------------------------------------------------------
h1, h2 = st.columns([3, 2])
with h1:
    st.markdown(f"## Alerta temprana · Central Guavio <span class='seal'>PROTOTIPO</span>"
                f"<span class='seal'>{SEAL}</span>",
                unsafe_allow_html=True)
    st.caption(f"Oferta del {day:%d-%b-%Y} · información al corte de las 18:00 del {day - pd.Timedelta(days=1):%d-%b} · "
               f"hidrología hasta el {origin:%d-%b-%Y} · escenario: **{scenario}**")
with h2:
    if main_alert is None:
        st.success("Sin alertas activas para la fecha seleccionada.")
    else:
        fn = {"rojo": st.error, "ambar": st.warning, "info": st.info, "gris": st.info}[main_alert.level]
        fn(f"**{main_alert.title}.** {main_alert.message}")


def kpi_card(col, label: str, value: str, unit: str, sub: str, status: str, spark: pd.Series | None, color: str):
    with col:
        st.markdown(f"<div class='kpi'><div class='lbl'><span>{label}</span><span title='{STATUS_TXT[status]}'>"
                    f"<span class='dot' style='background:{STATUS[status]}'></span></span></div>"
                    f"<div class='val'>{value} <span class='unit'>{unit}</span></div>"
                    f"<div class='sub'>{sub} · {STATUS_TXT[status]}</div></div>", unsafe_allow_html=True)
        if spark is not None:
            st.plotly_chart(ch.sparkline(spark, color), config={"displayModeBar": False},
                            key=f"spark_{label}")


nino = float(rawi.loc[day - pd.Timedelta(days=1), "nino34"])
ap_pct7 = float(rawi.loc[:origin, "aportes_pct_media"].tail(7).mean())
vol = float(rawi.loc[origin, "volumen_util_pct"])
clim_vol = float(rawi.loc[origin, "clim_vol_pct"])
gen7 = float(rawi.loc[:origin, "generacion_gwh"].tail(7).mean())
spark_win = rawi.loc[origin - pd.Timedelta(days=60):origin]

st_nino = "rojo" if nino >= cfg["alerts"]["nino_red"] else "ambar" if nino >= cfg["alerts"]["nino_amber"] else "verde"
st_ap = "rojo" if ap_pct7 < 70 else "ambar" if ap_pct7 < 90 else "verde"
st_vol = "rojo" if vol < clim_vol - 20 else "ambar" if vol < clim_vol - cfg["alerts"]["low_volume_points"] else "verde"
st_off = "ambar" if offer and offer["tercil"] == "Alto" else "verde"
st_fal = "gris" if p_falla is None else ("rojo" if p_falla >= cfg["alerts"]["failure_red"] else
                                          "ambar" if p_falla >= cfg["alerts"]["failure_amber"] else "verde")

cols = st.columns([5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5] if HAS_MACHINE else [5, 1, 5, 1, 5, 1, 5, 1, 5])
kpi_card(cols[0], "Clima", f"{nino:+.1f}", "°C Niño 3.4",
         "El Niño" if nino >= 0.5 else "La Niña" if nino <= -0.5 else "Neutral", st_nino,
         spark_win["nino34"], ch.YELLOW)
kpi_card(cols[2], "Aportes hídricos", f"{ap_pct7:.0f}", "% de la media", "media 7 días", st_ap,
         spark_win["aportes_gwh"], ch.AQUA)
kpi_card(cols[4], "Embalse", f"{vol:.0f}", "% vol. útil", f"típico {clim_vol:.0f} %", st_vol,
         spark_win["volumen_util_pct"], ch.BLUE)
kpi_card(cols[6], "Generación", f"{gen7 * 1000 / 24:.0f}", "MW medios", "media 7 días", "gris",
         spark_win["generacion_gwh"], ch.VIOLET)
if offer:
    kpi_card(cols[8], "Oferta estimada", f"{offer['precio_cop']:.0f}", "COP/kWh",
             f"P(mínimo) {100 * offer['p_floor']:.0f} % · tercil {offer['tercil'].lower()}", st_off, None, ch.ORANGE)
if HAS_MACHINE:
    kpi_card(cols[10], "Máquina", f"{100 * p_falla:.0f}" if p_falla is not None else "–", "% riesgo 7 d",
             f"vib. {rawi.loc[day - pd.Timedelta(days=1), 'vibracion_mm_s']:.1f} mm/s", st_fal,
             spark_win["vibracion_mm_s"], ch.RED)
for i in ((1, 3, 5, 7, 9) if HAS_MACHINE else (1, 3, 5, 7)):
    cols[i].markdown("<div class='arrow'>→</div>", unsafe_allow_html=True)

tab_names = ["Panorama", "Aportes: histórico", "Aportes: pronóstico", "Embalse y generación",
             "Comportamiento de oferta", "Observado vs. predicho", "Explicabilidad y desempeño"]
if HAS_MACHINE:
    tab_names.append("Mantenimiento predictivo")
tabs = st.tabs(tab_names)

# ---------------------------------------------------------------------------
# 1. Panorama
# ---------------------------------------------------------------------------
with tabs[0]:
    c1, c2 = st.columns([3, 1.25])
    with c1:
        hist = raw[(raw.fecha > origin - pd.Timedelta(days=90)) & (raw.fecha <= origin)]
        st.plotly_chart(ch.aportes_fan(hist, path, scenario))
        recent = ofp_all[(ofp_all.experimento == B["default_experiment"]) & (ofp_all.fecha <= day)
                         & (ofp_all.fecha > day - pd.Timedelta(days=window))]
        st.plotly_chart(ch.bid_chart(recent, B["oferta_art"]["tercile_edges"]))
    with c2:
        st.markdown("#### ¿Por qué esta estimación?")
        if offer:
            stage = "oferta_clasificador" if offer["p_floor"] > offer["tau"] else "oferta_regresor"
            loc = local_explanation(stage, day, offer["x"] if factor != 1.0 else None)
            if loc is not None:
                grp = local_grouped(loc, 0).head(5)
                mx = grp.contribucion.abs().max() or 1
                rows = ""
                for _, r in grp.iterrows():
                    color = ch.POS if r.contribucion > 0 else ch.NEG
                    verb = "sube" if r.contribucion > 0 else "baja"
                    rows += (f"<div class='row'><span>{r.familia}</span><span>{verb} {abs(r.contribucion):.2f}</span></div>"
                             f"<div class='bar' style='width:{100 * abs(r.contribucion) / mx:.0f}%;background:{color}'></div>")
                target = ("la probabilidad de ofertar al mínimo (log-odds)" if stage == "oferta_clasificador"
                          else "el precio de oferta (log de la prima sobre el mínimo)")
                top = local_contributions(loc, 0).iloc[0]
                st.markdown(f"<div class='why'><div style='font-size:0.85rem;color:#52514e'>Contribución SHAP "
                            f"agrupada a {target}</div>{rows}<div style='font-size:0.82rem;margin-top:10px'>"
                            f"Variable principal: <b>{top.variable}</b> = {top.valor:.2f}</div></div>",
                            unsafe_allow_html=True)
            st.markdown("")
            st.metric("Probabilidad de oferta al mínimo", f"{100 * offer['p_floor']:.0f} %",
                      help=f"Umbral τ = {offer['tau']:.2f}, elegido en validación para maximizar F1.")
            st.metric("Precio estimado (si no oferta al mínimo)", f"{offer['precio_reg_cop']:.0f} COP/kWh",
                      f"{offer['y_reg']:.2f} × precio mínimo ({offer['pmin']:.0f})", delta_color="off")
            if np.isnan(offer["observado_cop"]):
                st.caption("La oferta observada de este día aún no está publicada (lote mensual, M+1).")
            else:
                st.caption(f"Oferta observada (publicada después): {offer['observado_cop']:.0f} COP/kWh.")
        st.markdown("#### Alertas activas")
        if not alerts:
            st.caption("Ninguna.")
        for a in sorted(alerts, key=lambda x: {"rojo": 0, "ambar": 1, "info": 2, "gris": 3}[x.level]):
            icon = {"rojo": "🔴", "ambar": "🟠", "info": "🔵", "gris": "⚪"}[a.level]
            st.markdown(f"{icon} **{a.title}** — {a.message}")

# ---------------------------------------------------------------------------
# 2. Aportes: histórico
# ---------------------------------------------------------------------------
with tabs[1]:
    rh = raw[raw.fecha <= origin]
    yr = rh[rh.fecha.dt.year == day.year]
    clim_ytd = rh.loc[yr.index, "clim_aportes"].sum()
    k1, k2, k3 = st.columns(3)
    k1.metric("Aporte acumulado del año", f"{yr.aportes_gwh.sum():,.0f} GWh",
              f"{100 * (yr.aportes_gwh.sum() / clim_ytd - 1):+.0f} % vs. media", delta_color="normal")
    pct_now = float(np.mean(rh.loc[rh.fecha.dt.dayofyear == origin.dayofyear, "aportes_gwh"]
                            <= rawi.loc[origin, "aportes_gwh"]) * 100)
    k2.metric("Percentil del aporte actual", f"{pct_now:.0f}", help="Frente al mismo día del año en la historia.")
    below = (rh.aportes_gwh < rh.clim_aportes).iloc[::-1]
    streak = int((~below).cumsum().eq(0).sum())
    k3.metric("Días consecutivos bajo la media", f"{streak}")
    st.plotly_chart(ch.aportes_history(rh))
    c1, c2 = st.columns(2)
    c1.plotly_chart(ch.climatology_band(rh, day.year))
    c2.plotly_chart(ch.nino_lag_correlation(rh))
    st.plotly_chart(ch.anomaly_heatmap(rh))

# ---------------------------------------------------------------------------
# 3. Aportes: pronóstico
# ---------------------------------------------------------------------------
with tabs[2]:
    k = st.columns(5)
    k[0].metric("P50 día t (h = 2)", f"{path.p50.iloc[1]:.1f} GWh/d")
    k[1].metric("P50 media a 7 días", f"{path.p50.iloc[1:8].mean():.1f} GWh/d")
    k[2].metric("P50 media a 14 días", f"{path.p50.iloc[1:15].mean():.1f} GWh/d")
    k[3].metric("Anomalía pronosticada 14 d", f"{100 * (path.p50.iloc[1:15].mean() / path.clim.iloc[1:15].mean() - 1):+.0f} %")
    k[4].metric("Prob. de déficit (< P30)", f"{100 * deficit_probability(path):.0f} %",
                help="Probabilidad de que el aporte medio a 14 días quede bajo el percentil 30 histórico "
                     "(aproximación normal a partir del intervalo P10–P90).")
    c1, c2 = st.columns([3, 2])
    hist = raw[(raw.fecha > origin - pd.Timedelta(days=45)) & (raw.fecha <= origin)]
    c1.plotly_chart(ch.aportes_fan(hist, path, scenario), key="fan2")
    split_sel = c2.radio("Periodo de evaluación", ["test_a", "test_b"], horizontal=True,
                         format_func=lambda s: {"test_a": "Prueba A (jul-24 a mar-25)",
                                                "test_b": "Prueba B (abr-25 en adelante)"}[s])
    c2.plotly_chart(ch.horizon_error(B["aportes_metrics_h"], split_sel))
    with st.expander("Comparar escenarios"):
        comp = pd.DataFrame({name: forecast_path(B, origin, f)["p50"].values
                             for name, f in cfg["scenarios"].items()}, index=path.fecha.dt.date)
        st.dataframe(comp.round(2))
    if B["aportes_pred"].set_index("fecha").loc[origin, "split"] in ("train", "val"):
        st.caption("Nota: para orígenes dentro del periodo de desarrollo (≤ may-2024) el pronóstico es en muestra.")

# ---------------------------------------------------------------------------
# 4. Embalse y generación
# ---------------------------------------------------------------------------
with tabs[3]:
    proj = float(offer["x"]["pred_vol_proy_14d"].iloc[0]) if offer and "pred_vol_proy_14d" in offer["x"] else None
    k = st.columns(5)
    k[0].metric("Volumen útil", f"{rawi.loc[origin, 'volumen_util_gwh']:,.0f} GWh", f"{vol:.0f} %", delta_color="off")
    k[1].metric("Volumen proyectado 14 d", f"{proj:.0f} %" if proj is not None else "–")
    if "nivel_embalse_msnm" in rawi:
        k[2].metric("Nivel del embalse", f"{rawi.loc[origin, 'nivel_embalse_msnm']:,.1f} msnm")
    else:
        k[2].metric("Capacidad útil", f"{rawi.loc[origin, 'capacidad_util_gwh']:,.0f} GWh")
    k[3].metric("Vertimientos 7 días", f"{rawi.loc[:origin, 'vertimientos_gwh'].tail(7).sum():.1f} GWh")
    cap = rawi.loc[origin, "disponibilidad_mw"]
    k[4].metric("Factor de planta 7 d", f"{100 * gen7 / (cfg['synthetic']['installed_capacity_mw'] * 24 / 1000):.0f} %",
                f"disponibles {cap:.0f} MW", delta_color="off")
    c1, c2 = st.columns(2)
    c1.plotly_chart(ch.volume_chart(raw, start, origin, proj))
    c2.plotly_chart(ch.balance_chart(raw, start, origin))
    st.plotly_chart(ch.generation_chart(raw, start, origin))

# ---------------------------------------------------------------------------
# 5. Comportamiento de oferta
# ---------------------------------------------------------------------------
with tabs[4]:
    if offer:
        pub = ofp_all[(ofp_all.experimento == B["default_experiment"])]
        last_month = (day - pd.Timedelta(days=1)).to_period("M") - 1
        lm = pub[pub.fecha.dt.to_period("M") == last_month]
        k = st.columns(4)
        k[0].metric("P(oferta al mínimo)", f"{100 * offer['p_floor']:.0f} %",
                    "sobre el umbral" if offer["p_floor"] > offer["tau"] else "bajo el umbral", delta_color="off")
        k[1].metric("Precio estimado", f"{offer['precio_cop']:.0f} COP/kWh", f"{offer['y_hat']:.2f} × mínimo",
                    delta_color="off")
        k[2].metric("Tercil esperado", offer["tercil"])
        k[3].metric(f"Días al mínimo en {last_month.strftime('%b-%Y')}", f"{int(lm.q_floor.sum())} de {len(lm)}",
                    help="Último mes publicado al corte de información.")
    rec = ofp_all[(ofp_all.experimento == B["default_experiment"]) & (ofp_all.fecha <= day)
                  & (ofp_all.fecha > day - pd.Timedelta(days=window))]
    c1, c2 = st.columns(2)
    c1.plotly_chart(ch.floor_probability_chart(rec, B["oferta_art"]["tau"]))
    c2.plotly_chart(ch.bid_vs_volume(rec, raw))
    st.plotly_chart(ch.bid_chart(rec, B["oferta_art"]["tercile_edges"]), key="bid2")
    st.caption("Etapa 1 estima si la planta ofertará al mínimo regulatorio; si la probabilidad no supera τ, "
               "la etapa 2 estima el precio. Basado en Herrera-Mejía et al. (2026), Smart Energy 21, 100221.")

# ---------------------------------------------------------------------------
# 6. Observado vs. predicho
# ---------------------------------------------------------------------------
with tabs[5]:
    c0, c00 = st.columns(2)
    exp = c0.selectbox("Experimento", cfg["oferta_model"]["experiments"],
                       index=cfg["oferta_model"]["experiments"].index(B["default_experiment"]),
                       format_func=lambda e: {"E1_observada": "E1 · solo hidrología observada (réplica)",
                                              "E2_observada_mas_pronostico": "E2 · observada + pronosticada",
                                              "E4_cota_superior": "E4 · aportes futuros reales (cota)"}[e])
    split = c00.selectbox("Periodo", ["test_a", "test_b", "val"],
                          format_func=lambda s: {"test_a": "Prueba A · jul-24 a mar-25 (estrés)",
                                                 "test_b": "Prueba B · abr-25 en adelante (El Niño 2026)",
                                                 "val": "Validación · nov-22 a may-24"}[s])
    sel = ofp_all[(ofp_all.experimento == exp) & (ofp_all.split == split) & ofp_all.y_norm.notna()]
    edges = tuple(B["oferta_metrics"]["experimentos"][exp]["terciles"])
    res = B["oferta_metrics"]["experimentos"][exp][split]
    k = st.columns(5)
    k[0].metric("MAE (× mínimo)", f"{res['combinado']['MAE']:.2f}")
    k[1].metric("MAE (COP/kWh)", f"{res['combinado']['MAE_COP_kWh']:.0f}")
    k[2].metric("Acierto por terciles", f"{100 * res['combinado']['acierto_terciles']:.0f} %")
    k[3].metric("F1 oferta al mínimo", f"{res['clasificador']['F1']:.2f}",
                f"prevalencia {100 * res['clasificador']['prevalencia']:.1f} %", delta_color="off")
    k[4].metric("Sesgo medio", f"{(sel.y_hat - sel.y_norm).mean():+.2f}")
    st.plotly_chart(ch.observed_vs_pred(sel, "Observado vs. predicho"))
    c1, c2, c3 = st.columns(3)
    c1.plotly_chart(ch.confusion_heatmap(ev.tercile_confusion(sel.y_norm, sel.y_hat, edges),
                                         "Matriz de terciles"))
    pred_cls = (sel.p_floor > B["oferta_metrics"]["experimentos"][exp]["umbral_tau"]).astype(int)
    cm = pd.crosstab(pd.Categorical(sel.q_floor.map({0: "No mínimo", 1: "Mínimo"}), ["No mínimo", "Mínimo"]),
                     pd.Categorical(pred_cls.map({0: "No mínimo", 1: "Mínimo"}), ["No mínimo", "Mínimo"]),
                     dropna=False)
    c2.plotly_chart(ch.confusion_heatmap(cm, "Matriz del clasificador"))
    c3.plotly_chart(ch.residuals_by_month(sel))
    dm = B["oferta_metrics"]["diebold_mariano_E2_vs_E1"]
    if split in dm:
        st.caption(f"Diebold-Mariano E2 vs. E1 en {split}: DM = {dm[split]['DM']:.2f}, p = {dm[split]['p_value']:.2f} "
                   "(DM < 0 indica que E2 tiene menor error).")

# ---------------------------------------------------------------------------
# 7. Explicabilidad y desempeño
# ---------------------------------------------------------------------------
with tabs[6]:
    names = {"oferta_regresor": "Oferta · etapa 2 (precio)", "oferta_clasificador": "Oferta · etapa 1 (mínimo)",
             "aportes": "Aportes · P50 día t", "falla": "Riesgo de falla a 7 días"}
    names = {k_: v_ for k_, v_ in names.items() if k_ in get_explainers()}
    c0, c00 = st.columns([2, 1])
    model = c0.selectbox("Modelo a explicar", list(names), format_func=names.get)
    shap_days = c00.select_slider("Ventana SHAP", [90, 180, 365], value=180, format_func=lambda d: f"{d} días")
    item = get_explainers()[model]
    ref_day = day if model != "aportes" else origin
    expl, edates = window_explanation(model, str(ref_day.date()), shap_days)
    st.caption(f"Escala de SHAP: **{item['unidad']}**. Las explicaciones describen lo que aprendió el modelo, "
               "no relaciones causales.")
    c1, c2 = st.columns(2)
    loc = local_explanation(model, ref_day)
    if loc is not None:
        contrib = local_contributions(loc, 0, top=10)
        c1.plotly_chart(ch.shap_waterfall(contrib, float(loc.base_values[0]), item["unidad"],
                                          f"Cascada SHAP del {ref_day:%d-%b-%Y}"))
    c2.plotly_chart(ch.family_bar(grouped_importance(expl), "Importancia por familia de variables"))
    st.plotly_chart(ch.shap_beeswarm(expl))
    regimes = hydrological_regime(edates.reset_index(drop=True),
                                  edates.reset_index(drop=True).map(rawi["nino34"]))
    c3, c4 = st.columns(2)
    feat_opts = [f for f in ("pred_aporte_14d", "vol_pct", "aportes_mean30", "vib_mean7", "aportes_lag0",
                             "bolsa_rel_pmin") if f in expl.feature_names] + \
                [f for f in expl.feature_names if f not in ("pred_aporte_14d", "vol_pct")]
    feat = c3.selectbox("Variable para dependencia", feat_opts)
    c3.plotly_chart(ch.shap_dependence(expl, feat, regimes))
    with c4:
        st.markdown("**Importancia por régimen hidrológico** (|SHAP| medio por familia)")
        st.dataframe(importance_by_regime(expl, regimes).round(3))
        if model in ("oferta_clasificador", "falla"):
            if model == "oferta_clasificador":
                v = ofp_all[(ofp_all.experimento == B["default_experiment"]) & ofp_all.split.isin(["test_a", "test_b"])]
                yv, pv = v.q_floor.values, v.p_floor.values
            else:
                v = fp[fp.split.isin(["test_a", "test_b"])]
                yv, pv = v.y_falla.values, v.p_falla.values
            st.plotly_chart(ch.calibration_curve(yv, pv, "Calibración en pruebas A+B"))

    st.markdown("### Desempeño de los modelos")
    t1, t2, t3, t4 = st.tabs(["Aportes", "Oferta", "Baselines y DM", "Deriva"])
    with t1:
        am = B["aportes_metrics_h"]
        st.dataframe(am[["split", "horizonte", "MAE", "RMSE", "WAPE_%", "NSE", "KGE", "skill_vs_clim",
                         "skill_vs_persist", "cobertura_P10_P90"]].round(3), hide_index=True)
        st.caption(f"Conjunto de variables elegido en validación: {B['aportes_metrics']['conjunto_variables']} "
                   f"(skill vs. climatología: {B['aportes_metrics']['skill_validacion']}).")
    with t2:
        rows = []
        for e, r in B["oferta_metrics"]["experimentos"].items():
            for s_ in ("val", "test_a", "test_b"):
                rows.append({"experimento": e, "periodo": s_, "F1": r[s_]["clasificador"]["F1"],
                             "PR_AUC": r[s_]["clasificador"]["PR_AUC"],
                             "prevalencia": r[s_]["clasificador"]["prevalencia"],
                             "MAE_regresor": r[s_]["regresor_dias_sin_minimo"].get("MAE", np.nan),
                             "MAE_combinado": r[s_]["combinado"]["MAE"],
                             "MAE_COP_kWh": r[s_]["combinado"]["MAE_COP_kWh"],
                             "acierto_terciles": r[s_]["combinado"]["acierto_terciles"],
                             "regresor": r["regresor"]})
        st.dataframe(pd.DataFrame(rows).round(3), hide_index=True)
    with t3:
        bl = []
        for s_, d in B["oferta_metrics"]["baselines"].items():
            for name, m in d.items():
                bl.append({"periodo": s_, "baseline": name, **{k_: v_ for k_, v_ in m.items()
                                                               if k_ in ("F1", "PR_AUC", "MAE", "acierto_terciles")}})
        st.dataframe(pd.DataFrame(bl).round(3), hide_index=True)
        st.json(B["oferta_metrics"]["diebold_mariano_E2_vs_E1"])
    with t4:
        st.dataframe(drift.round(3), hide_index=True)
        st.caption(f"PSI de los últimos 30 días frente a entrenamiento; > {cfg['alerts']['psi_threshold']} indica deriva.")

# ---------------------------------------------------------------------------
# 8. Mantenimiento predictivo
# ---------------------------------------------------------------------------
if HAS_MACHINE:
    with tabs[7]:
        dd = day - pd.Timedelta(days=1)
        k = st.columns(5)
        k[0].metric("Riesgo de falla 7 días", f"{100 * p_falla:.0f} %" if p_falla is not None else "–",
                    f"umbral {100 * B['falla_art']['tau']:.0f} %", delta_color="off")
        k[1].metric("Vibración (media 7 d)", f"{rawi.loc[:dd, 'vibracion_mm_s'].tail(7).mean():.2f} mm/s")
        k[2].metric("Temp. cojinete (media 7 d)", f"{rawi.loc[:dd, 'temp_cojinete_c'].tail(7).mean():.1f} °C")
        k[3].metric("Horas desde mantenimiento", f"{rawi.loc[dd, 'horas_desde_mant']:,.0f} h")
        k[4].metric("Unidades disponibles", f"{int(rawi.loc[dd, 'unidades_disponibles'])} de {cfg['synthetic']['n_units']}")
        fr = fp[(fp.fecha <= dd) & (fp.fecha > dd - pd.Timedelta(days=window))]
        st.plotly_chart(ch.failure_prob_chart(fr, raw, cfg["alerts"]["failure_amber"], cfg["alerts"]["failure_red"]))
        c1, c2 = st.columns(2)
        c1.plotly_chart(ch.sensor_chart(raw, start, dd, "vibracion_mm_s", "Vibración de turbinas (RMS)", "mm/s", ch.BLUE))
        c2.plotly_chart(ch.sensor_chart(raw, start, dd, "temp_cojinete_c", "Temperatura de cojinetes", "°C", ch.ORANGE))
        st.caption("Líneas punteadas: rojo = falla forzada, verde = inicio de mantenimiento programado.")
        loc = local_explanation("falla", dd)
        if loc is not None:
            contrib = local_contributions(loc, 0, top=8)
            st.plotly_chart(ch.shap_waterfall(contrib, float(loc.base_values[0]), "log-odds",
                                              f"¿Por qué este riesgo? Cascada SHAP del {dd:%d-%b-%Y}"))

st.markdown("---")
st.caption(f"{'Fuentes: API XM (Sinergox) · NOAA CPC' if IS_XM else 'Fuentes simuladas: XM · IDEAM · NOAA'} | Motor ML: LightGBM, XGBoost, scikit-learn · "
           f"SHAP | Datos hasta {raw.fecha.max():%d-%b-%Y} | {cfg['project']['disclaimer']}")

# Reproducción día a día (simulación de llegada de datos)
if play and day.date() < max_day:
    time.sleep(speed)
    st.session_state["pending_day"] = (day + pd.Timedelta(days=1)).date()
    st.rerun()
