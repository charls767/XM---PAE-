"""Figuras Plotly del dashboard. Paleta categórica fija y colores de estado reservados."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap

from src.data.features import family_of

# Paleta categórica (orden fijo) y colores de estado.
BLUE, ORANGE, AQUA, YELLOW, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6", "#e34948"
GOOD, WARNING, SERIOUS, CRITICAL = "#0ca30c", "#fab219", "#ec835a", "#d03b3b"
INK, INK2, GRID, BAND = "#0b0b0b", "#52514e", "#e6e5e1", "rgba(42,120,214,0.14)"
NEG, POS = "#2a78d6", "#e34948"  # SHAP: reduce / aumenta la salida
FONT = "Inter, Segoe UI, Roboto, sans-serif"


def _layout(fig: go.Figure, title: str = "", height: int = 340, y_title: str = "", x_title: str = "") -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=15, color=INK)),
        height=height, margin=dict(l=8, r=8, t=48 if title else 16, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=12, color=INK2),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, title_text=x_title)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, title_text=y_title)
    return fig


def sparkline(series: pd.Series, color: str = BLUE) -> go.Figure:
    fig = go.Figure(go.Scatter(x=series.index, y=series.values, mode="lines",
                               line=dict(color=color, width=2), hovertemplate="%{y:.1f}<extra></extra>"))
    fig.update_layout(height=60, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def aportes_fan(hist: pd.DataFrame, path: pd.DataFrame, scenario: str) -> go.Figure:
    """Observado reciente + pronóstico P10-P90 + media histórica."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.fecha, y=hist.clim_aportes, name="Media histórica", mode="lines",
                             line=dict(color=INK2, width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=path.fecha, y=path.clim, showlegend=False, mode="lines",
                             line=dict(color=INK2, width=1.5, dash="dot"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=list(path.fecha) + list(path.fecha[::-1]),
                             y=list(path.p90) + list(path.p10[::-1]), fill="toself", fillcolor=BAND,
                             line=dict(width=0), name="Intervalo P10–P90", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=hist.fecha, y=hist.aportes_gwh, name="Observado", mode="lines",
                             line=dict(color=INK, width=2)))
    fig.add_trace(go.Scatter(x=path.fecha, y=path.p50, name=f"Pronóstico P50 · {scenario}", mode="lines",
                             line=dict(color=BLUE, width=2, dash="dash")))
    if path["observado"].notna().any():
        fig.add_trace(go.Scatter(x=path.fecha, y=path.observado, name="Observado posterior (validación)",
                                 mode="markers", marker=dict(color=INK, size=6, symbol="circle-open")))
    fig.add_vline(x=path.fecha.iloc[0] - pd.Timedelta(days=1), line=dict(color=INK2, width=1))
    return _layout(fig, "Aportes hídricos: observado y pronóstico a 15 días", 360, "GWh/día")


def aportes_history(raw: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    nino_on = raw["nino34"] >= 1.0
    blocks = (nino_on != nino_on.shift()).cumsum()
    for _, g in raw[nino_on].groupby(blocks[nino_on]):
        fig.add_vrect(x0=g.fecha.iloc[0], x1=g.fecha.iloc[-1], fillcolor=WARNING, opacity=0.12, line_width=0)
    m = raw.set_index("fecha")["aportes_gwh"].rolling(30).mean()
    fig.add_trace(go.Scatter(x=raw.fecha, y=raw.aportes_gwh, name="Diario", mode="lines",
                             line=dict(color="rgba(42,120,214,0.35)", width=1)))
    fig.add_trace(go.Scatter(x=m.index, y=m.values, name="Media móvil 30 días", mode="lines",
                             line=dict(color=BLUE, width=2)))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="Niño 3.4 ≥ +1,0 °C",
                             marker=dict(color=WARNING, size=10, symbol="square", opacity=0.5)))
    return _layout(fig, "Aportes 2016 a la fecha (franjas: El Niño)", 340, "GWh/día")


def climatology_band(raw: pd.DataFrame, year: int) -> go.Figure:
    hist = raw[raw.fecha.dt.year < year]
    q = hist.groupby(hist.fecha.dt.dayofyear)["aportes_gwh"].quantile([0.1, 0.5, 0.9]).unstack()
    q = q.rolling(15, center=True, min_periods=1).mean()
    cur = raw[raw.fecha.dt.year == year]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(q.index) + list(q.index[::-1]), y=list(q[0.9]) + list(q[0.1][::-1]),
                             fill="toself", fillcolor=BAND, line=dict(width=0), name="P10–P90 histórico",
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=q.index, y=q[0.5], name="Mediana histórica", line=dict(color=INK2, dash="dot")))
    fig.add_trace(go.Scatter(x=cur.fecha.dt.dayofyear, y=cur.aportes_gwh, name=str(year),
                             line=dict(color=ORANGE, width=2)))
    return _layout(fig, f"Año {year} frente a la distribución histórica", 320, "GWh/día", "Día del año")


def anomaly_heatmap(raw: pd.DataFrame) -> go.Figure:
    m = raw.assign(anio=raw.fecha.dt.year, mes=raw.fecha.dt.month)
    piv = m.groupby(["anio", "mes"])["aportes_pct_media"].mean().unstack() - 100
    meses = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
    fig = go.Figure(go.Heatmap(z=piv.values, x=[meses[c - 1] for c in piv.columns], y=piv.index.astype(str),
                               colorscale=[[0, "#b5442c"], [0.5, "#f0efec"], [1, "#1c5cab"]], zmid=0,
                               zmin=-60, zmax=60, colorbar=dict(title="% vs media"),
                               hovertemplate="%{y} %{x}: %{z:+.0f} %<extra></extra>", xgap=2, ygap=2))
    fig = _layout(fig, "Anomalía mensual de aportes (%)", 360)
    fig.update_layout(hovermode="closest")
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    return fig


def nino_lag_correlation(raw: pd.DataFrame) -> go.Figure:
    mo = raw.set_index("fecha")[["nino34", "aportes_pct_media"]].resample("MS").mean()
    lags = list(range(0, 7))
    corr = [mo["aportes_pct_media"].corr(mo["nino34"].shift(lag)) for lag in lags]
    fig = go.Figure(go.Bar(x=[f"{lag} m" for lag in lags], y=corr, marker_color=BLUE,
                           hovertemplate="Rezago %{x}: r = %{y:.2f}<extra></extra>"))
    fig = _layout(fig, "Correlación Niño 3.4 rezagado vs. anomalía de aportes", 300, "r de Pearson",
                  "Rezago del Niño 3.4")
    fig.update_layout(hovermode="closest")
    return fig


def horizon_error(metrics_h: pd.DataFrame, split: str) -> go.Figure:
    d = metrics_h[metrics_h.split == split]
    fig = go.Figure()
    for col, name, color in (("MAE", "Modelo (P50)", BLUE), ("MAE_persistencia", "Persistencia", ORANGE),
                             ("MAE_climatologia", "Climatología", AQUA)):
        fig.add_trace(go.Scatter(x=d.horizonte, y=d[col], name=name, mode="lines+markers",
                                 line=dict(color=color, width=2), marker=dict(size=8)))
    return _layout(fig, "Error absoluto medio por horizonte", 300, "MAE (GWh/día)", "Horizonte (días)")


def volume_chart(raw: pd.DataFrame, start: pd.Timestamp, day: pd.Timestamp, proj: float | None) -> go.Figure:
    d = raw[(raw.fecha >= start) & (raw.fecha <= day)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.fecha, y=d.clim_vol_pct, name="Media histórica de la fecha",
                             line=dict(color=INK2, dash="dot", width=1.5)))
    fig.add_trace(go.Scatter(x=d.fecha, y=d.volumen_util_pct, name="Volumen útil", line=dict(color=BLUE, width=2)))
    if proj is not None and not np.isnan(proj):
        fig.add_trace(go.Scatter(x=[d.fecha.iloc[-1] - pd.Timedelta(days=2), day + pd.Timedelta(days=14)],
                                 y=[d.volumen_util_pct.iloc[-3], proj], name="Proyección a 14 días",
                                 line=dict(color=BLUE, dash="dash", width=2)))
    return _layout(fig, "Volumen útil del embalse", 320, "% de la capacidad útil")


def balance_chart(raw: pd.DataFrame, start: pd.Timestamp, day: pd.Timestamp) -> go.Figure:
    d = raw[(raw.fecha >= start) & (raw.fecha <= day)].set_index("fecha")
    w = d[["aportes_gwh", "generacion_gwh", "vertimientos_gwh"]].resample("W").sum()
    net = w.aportes_gwh - w.generacion_gwh - w.vertimientos_gwh
    fig = go.Figure(go.Bar(x=net.index, y=net.values, marker_color=np.where(net >= 0, BLUE, ORANGE),
                           hovertemplate="%{x|%d %b %Y}: %{y:+.0f} GWh<extra></extra>"))
    fig = _layout(fig, "Balance semanal: aportes − generación − vertimientos", 300, "GWh/semana")
    fig.update_layout(hovermode="closest")
    return fig


def generation_chart(raw: pd.DataFrame, start: pd.Timestamp, day: pd.Timestamp) -> go.Figure:
    d = raw[(raw.fecha >= start) & (raw.fecha <= day)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.fecha, y=d.aportes_gwh.rolling(7).mean(), name="Aportes (media 7 d)",
                             line=dict(color=AQUA, width=2)))
    fig.add_trace(go.Scatter(x=d.fecha, y=d.generacion_gwh.rolling(7).mean(), name="Generación (media 7 d)",
                             line=dict(color=BLUE, width=2)))
    fig.add_trace(go.Bar(x=d.fecha, y=d.vertimientos_gwh, name="Vertimientos", marker_color=ORANGE))
    return _layout(fig, "Generación, aportes y vertimientos", 320, "GWh/día")


def floor_probability_chart(pred: pd.DataFrame, tau: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pred.fecha, y=pred.p_floor, name="Probabilidad estimada",
                             line=dict(color=BLUE, width=2)))
    fl = pred[pred.q_floor == 1]
    fig.add_trace(go.Scatter(x=fl.fecha, y=np.ones(len(fl)) * 1.02, mode="markers", name="Ofertó al mínimo",
                             marker=dict(color=INK, size=6, symbol="line-ns-open")))
    fig.add_hline(y=tau, line=dict(color=INK2, dash="dash", width=1),
                  annotation_text=f"umbral τ = {tau:.2f}", annotation_position="top left")
    fig = _layout(fig, "Etapa 1 · probabilidad de oferta al mínimo regulatorio", 300, "Probabilidad")
    fig.update_yaxes(range=[0, 1.06], tickformat=".0%")
    return fig


def bid_chart(pred: pd.DataFrame, edges: tuple[float, float]) -> go.Figure:
    fig = go.Figure()
    ymax = max(pred.y_norm.max(), pred.y_hat.max()) * 1.05
    for (lo, hi, name, alpha) in ((0, edges[0], "Tercil bajo", 0.04), (edges[0], edges[1], "Tercil medio", 0.08),
                                  (edges[1], ymax, "Tercil alto", 0.13)):
        fig.add_hrect(y0=lo, y1=hi, fillcolor=f"rgba(235,104,52,{alpha})", line_width=0,
                      annotation_text=name, annotation_position="top left",
                      annotation_font=dict(size=11, color=INK2))
    fig.add_trace(go.Scatter(x=pred.fecha, y=pred.y_norm, name="Observado (publicado M+1)",
                             line=dict(color=INK, width=1.5)))
    fig.add_trace(go.Scatter(x=pred.fecha, y=pred.y_hat, name="Estimado (dos etapas)",
                             line=dict(color=BLUE, width=2)))
    return _layout(fig, "Precio de oferta normalizado (oferta / precio mínimo)", 340, "× precio mínimo")


def bid_vs_volume(pred: pd.DataFrame, raw: pd.DataFrame) -> go.Figure:
    d = pred.merge(raw[["fecha", "volumen_util_pct"]], on="fecha")
    fig = go.Figure(go.Scatter(x=d.volumen_util_pct, y=d.y_norm, mode="markers",
                               marker=dict(size=8, color=d.fecha.dt.year, colorscale="Blues", showscale=True,
                                           colorbar=dict(title="Año"), line=dict(width=1, color="#fcfcfb")),
                               hovertemplate="Volumen %{x:.0f} % · oferta %{y:.2f}×<extra></extra>"))
    fig = _layout(fig, "Oferta observada vs. volumen útil", 320, "× precio mínimo", "Volumen útil (%)")
    fig.update_layout(hovermode="closest")
    return fig


def observed_vs_pred(pred: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pred.fecha, y=pred.y_norm, name="Observado", line=dict(color=INK, width=1.5)))
    fig.add_trace(go.Scatter(x=pred.fecha, y=pred.y_hat, name="Predicho", line=dict(color=BLUE, width=2)))
    return _layout(fig, title, 330, "× precio mínimo")


def confusion_heatmap(cm: pd.DataFrame, title: str) -> go.Figure:
    z = cm.values
    pct = z / np.maximum(z.sum(axis=1, keepdims=True), 1)
    fig = go.Figure(go.Heatmap(z=pct, x=[str(c) for c in cm.columns], y=[str(i) for i in cm.index],
                               colorscale=[[0, "#f0efec"], [1, "#1c5cab"]], zmin=0, zmax=1, showscale=False,
                               text=[[f"{v}<br>{p:.0%}" for v, p in zip(r, pr)] for r, pr in zip(z, pct)],
                               texttemplate="%{text}", xgap=2, ygap=2,
                               hovertemplate="Obs %{y} · Pred %{x}<extra></extra>"))
    fig = _layout(fig, title, 300, "Observado", "Predicho")
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    fig.update_layout(hovermode="closest")
    return fig


def residuals_by_month(pred: pd.DataFrame) -> go.Figure:
    r = pred.assign(res=pred.y_hat - pred.y_norm).set_index("fecha")["res"].resample("MS").mean()
    fig = go.Figure(go.Bar(x=r.index, y=r.values, marker_color=np.where(r >= 0, ORANGE, BLUE),
                           hovertemplate="%{x|%b %Y}: %{y:+.2f}<extra></extra>"))
    oct24 = pd.Timestamp("2024-10-01")
    if r.index.min() <= oct24 <= r.index.max():
        fig.add_vline(x=oct24, line=dict(color=CRITICAL, dash="dot"))
        fig.add_annotation(x=oct24, y=1, yref="paper", text="Estrés oct. 2024", showarrow=False,
                           font=dict(color=CRITICAL, size=11), xanchor="left")
    fig = _layout(fig, "Sesgo mensual (predicho − observado)", 280, "× precio mínimo")
    fig.update_layout(hovermode="closest")
    return fig


def shap_waterfall(contrib: pd.DataFrame, base: float, unit: str, title: str) -> go.Figure:
    """Cascada SHAP: valor base -> contribuciones -> predicción (escala del modelo)."""
    c = contrib.iloc[::-1]
    labels = [f"{v} = {x:.2f}" if pd.notna(x) else v for v, x in zip(c.variable, c.valor)]
    fig = go.Figure(go.Waterfall(
        orientation="h", y=["Valor base"] + labels + ["Predicción"],
        x=[base] + list(c.contribucion) + [0], measure=["absolute"] + ["relative"] * len(c) + ["total"],
        increasing=dict(marker=dict(color=POS)), decreasing=dict(marker=dict(color=NEG)),
        totals=dict(marker=dict(color=INK2)), connector=dict(line=dict(color=GRID)),
        hovertemplate="%{y}: %{x:+.3f}<extra></extra>"))
    fig = _layout(fig, title, 40 + 30 * (len(c) + 2), unit)
    fig.update_layout(hovermode="closest", showlegend=False)
    return fig


def shap_beeswarm(expl: shap.Explanation, top: int = 12) -> go.Figure:
    imp = np.abs(expl.values).mean(axis=0)
    order = np.argsort(imp)[::-1][:top][::-1]
    fig = go.Figure()
    rng = np.random.default_rng(0)
    for rank, j in enumerate(order):
        v = expl.data[:, j]
        norm = (v - np.nanmin(v)) / (np.nanmax(v) - np.nanmin(v) + 1e-12)
        fig.add_trace(go.Scatter(
            x=expl.values[:, j], y=rank + rng.uniform(-0.28, 0.28, len(v)), mode="markers",
            marker=dict(size=6, color=norm, colorscale=[[0, NEG], [1, POS]], opacity=0.7,
                        showscale=rank == len(order) - 1,
                        colorbar=dict(title="Valor de la variable", tickvals=[0, 1], ticktext=["bajo", "alto"])),
            name=expl.feature_names[j], showlegend=False,
            hovertemplate=expl.feature_names[j] + ": SHAP %{x:+.3f}<extra></extra>"))
    fig = _layout(fig, "Resumen SHAP (beeswarm): cada punto es un día", 40 + 32 * len(order) + 60, "",
                  "Valor SHAP")
    fig.update_yaxes(tickvals=list(range(len(order))), ticktext=[expl.feature_names[j] for j in order],
                     gridcolor="rgba(0,0,0,0)")
    fig.add_vline(x=0, line=dict(color=GRID))
    fig.update_layout(hovermode="closest")
    return fig


def family_bar(grouped: pd.DataFrame, title: str) -> go.Figure:
    g = grouped.sort_values("shap_medio_abs")
    fig = go.Figure(go.Bar(x=g.shap_medio_abs, y=g.familia, orientation="h", marker_color=BLUE,
                           text=[f"{p:.0f} %" for p in g["participacion_%"]], textposition="outside",
                           hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    fig = _layout(fig, title, 60 + 34 * len(g), "", "|SHAP| medio agrupado")
    fig.update_layout(hovermode="closest")
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
    return fig


def shap_dependence(expl: shap.Explanation, feature: str, regimes: pd.Series) -> go.Figure:
    j = expl.feature_names.index(feature)
    fig = go.Figure()
    colors = {"Húmedo": BLUE, "Seco": ORANGE, "El Niño": CRITICAL}
    for reg, color in colors.items():
        m = np.asarray(regimes) == reg
        if m.any():
            fig.add_trace(go.Scatter(x=expl.data[m, j], y=expl.values[m, j], mode="markers", name=reg,
                                     marker=dict(size=7, color=color, opacity=0.7, line=dict(width=1, color="#fcfcfb"))))
    fig = _layout(fig, f"Dependencia SHAP de {feature}", 320, "Valor SHAP", feature)
    fig.update_layout(hovermode="closest")
    return fig


def calibration_curve(y: np.ndarray, p: np.ndarray, title: str) -> go.Figure:
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
    df = pd.DataFrame({"b": idx, "y": y, "p": p}).groupby("b").agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Calibración perfecta",
                             line=dict(color=INK2, dash="dot", width=1)))
    fig.add_trace(go.Scatter(x=df.p, y=df.y, mode="lines+markers", name="Modelo", line=dict(color=BLUE, width=2),
                             marker=dict(size=8), customdata=df.n,
                             hovertemplate="Predicho %{x:.2f} · observado %{y:.2f} · n=%{customdata}<extra></extra>"))
    fig = _layout(fig, title, 300, "Frecuencia observada", "Probabilidad predicha")
    fig.update_layout(hovermode="closest")
    return fig


def sensor_chart(raw: pd.DataFrame, start: pd.Timestamp, day: pd.Timestamp, col: str, title: str,
                 unit: str, color: str) -> go.Figure:
    d = raw[(raw.fecha >= start) & (raw.fecha <= day)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.fecha, y=d[col], name="Diario", line=dict(color=color, width=1.2)))
    fig.add_trace(go.Scatter(x=d.fecha, y=d[col].rolling(7).mean(), name="Media 7 días",
                             line=dict(color=INK, width=2)))
    for _, r in d[d.falla_forzada == 1].iterrows():
        fig.add_vline(x=r.fecha, line=dict(color=CRITICAL, width=1, dash="dot"))
    for _, r in d[(d.mantenimiento_programado == 1) & (d.mantenimiento_programado.shift() != 1)].iterrows():
        fig.add_vline(x=r.fecha, line=dict(color=GOOD, width=1, dash="dot"))
    return _layout(fig, title, 280, unit)


def failure_prob_chart(pred: pd.DataFrame, raw: pd.DataFrame, amber: float, red: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pred.fecha, y=pred.p_falla, name="Probabilidad de falla a 7 días",
                             line=dict(color=BLUE, width=2)))
    ev = raw[(raw.falla_forzada == 1) & raw.fecha.between(pred.fecha.min(), pred.fecha.max())]
    fig.add_trace(go.Scatter(x=ev.fecha, y=np.ones(len(ev)) * 1.02, mode="markers", name="Falla forzada",
                             marker=dict(color=CRITICAL, size=10, symbol="x")))
    fig.add_hline(y=amber, line=dict(color=WARNING, dash="dash", width=1), annotation_text="ámbar",
                  annotation_position="top left")
    fig.add_hline(y=red, line=dict(color=CRITICAL, dash="dash", width=1), annotation_text="rojo",
                  annotation_position="top left")
    fig = _layout(fig, "Riesgo de falla forzada en los próximos 7 días", 300, "Probabilidad")
    fig.update_yaxes(range=[0, 1.06], tickformat=".0%")
    return fig


def families_of(features: list[str]) -> list[str]:
    return [family_of(f) for f in features]
