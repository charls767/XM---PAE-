# guavio-xai · Analítica predictiva e IA explicable para la Central Guavio

Prototipo de apoyo analítico que implementa la propuesta del proyecto:

1. **Modelo de aportes hídricos** (GWh/día) a 1–15 días con cuantiles P10/P50/P90 (LightGBM), con intervalo calibrado de forma conformal.
2. **Modelo de oferta en dos etapas** (Herrera-Mejía et al., 2026): clasificador de oferta al mínimo regulatorio (XGBoost + calibración isotónica + umbral τ por F1) y regresor del precio normalizado (Ridge / Random Forest / XGBoost, elegido en validación).
3. **Acople hidrología → oferta**: los aportes pronosticados *fuera de muestra* (walk-forward anual) entran como variables del modelo de oferta; ablación E1 (observada), E2 (observada + pronosticada) y E4 (cota superior con aportes reales).
4. **Mantenimiento predictivo** (complementario): riesgo de falla forzada a 7 días a partir de vibración, temperatura de cojinetes, carga y horas desde mantenimiento.
5. **Explicabilidad**: SHAP global, local (cascada), agrupado por familia de variables y por régimen hidrológico; LIME como comparación local.
6. **Dashboard Streamlit** de 8 vistas con KPIs encadenados, alertas, escenarios y reproducción día a día.

> **Importante.** El prototipo **no** recomienda precios de oferta ni toma decisiones operativas.

## Fuentes de datos

| Fuente | Qué trae | Cómo se activa |
| --- | --- | --- |
| **`xm`** (por defecto) | API pública de XM (`servapibi.xm.com.co`, la que alimenta Sinergox) + Niño 3.4 semanal de NOAA | `--source xm` o `data.source: xm` en `config/config.yaml` |
| `synthetic` | Simulador: hidrología, mercado y condición de máquina (vibración, temperatura de cojinetes, fallas) | `--source synthetic` |
| `xm_mock` | Simulador de la API de XM con el mismo formato JSON, para ensayar el flujo real sin internet | `python -m tests.xm_mock` |

Cada fuente guarda sus datos, modelos y reportes en carpetas propias (`data/processed/<fuente>`, `models/<fuente>`, `reports/<fuente>`), y el dashboard permite alternar entre las que estén entrenadas.

**Métricas de XM usadas** (editables en `config/config.yaml → xm.metrics`):

| Variable | MetricId / Entidad | Filtro | Unidad XM → proyecto |
| --- | --- | --- | --- |
| Aportes | `AporEner` / Rio | `GUAVIO` | kWh → GWh/día |
| Media histórica de aportes | `AporEnerMediHist` / Rio | `GUAVIO` | kWh → GWh/día |
| Caudal | `AporCaudal` / Rio | `GUAVIO` | m³/s |
| Volumen y capacidad útil | `VoluUtilDiarEner`, `CapaUtilDiarEner` / Embalse | `GUAVIO` | kWh → GWh |
| Vertimientos | `VertEner` / Embalse | `GUAVIO` | kWh → GWh/día |
| Generación | `Gene` / Recurso (horaria, suma) | `GVIO` | kWh → GWh/día |
| Disponibilidad | `DispoReal` / Recurso (horaria, media) | `GVIO` | kW → MW |
| Precio de oferta | `PrecOferDesp` / Recurso (horaria, mediana) | `GVIO` y todo el SIN | COP/kWh |
| Precio de bolsa | `PrecBolsNaci` / Sistema (horaria, media) | — | COP/kWh |
| Demanda | `DemaCome` / Sistema (horaria, suma) | — | kWh → GWh/día |

Decisiones de diseño con datos reales:

- **Precio mínimo regulatorio:** se aproxima con la mediana mensual del menor precio ofertado (positivo) en el SIN cada día. Para usar los valores oficiales CREG, cambie `xm.price_floor.source` a `csv` y llene `config/precio_minimo.csv` (columnas `mes`, `precio_cop_kwh`).
- **Ofertas publicadas mes vencido:** por defecto (`latency.bids_published: monthly`) las ofertas propias y la fracción de plantas al mínimo solo entran como estadísticos del último mes publicado. Los días cuya oferta aún no se publica se predicen, pero no se usan para entrenar ni evaluar.
- **Mantenimiento predictivo:** XM no publica vibración ni temperatura de cojinetes, así que el modelo de fallas y su pestaña se omiten con la fuente `xm`.
- **Caché:** cada mes descargado se guarda en `data/external/xm_cache`; solo se vuelven a pedir los últimos 45 días (XM revisa datos recientes).

## Estructura

```
guavio-xai/
├── app.py                      # Dashboard Streamlit (8 vistas)
├── config/config.yaml          # Parámetros, particiones temporales, umbrales de alerta, escenarios
├── data/
│   ├── raw/                    # guavio_synthetic_daily.parquet
│   ├── processed/              # features, pronósticos walk-forward, predicciones
│   └── external/               # (para XM/IDEAM/NOAA reales)
├── dashboards/
│   ├── charts.py               # Figuras Plotly
│   └── data_access.py          # Datos, escenarios, deriva (PSI) y reglas de alerta
├── models/                     # Modelos entrenados (.joblib)
├── notebooks/01_exploracion.ipynb
├── reports/
│   ├── figures/                # Summary, bar, waterfall, familias, dependencia (SHAP)
│   └── metrics/                # JSON/CSV de métricas, SHAP y LIME
├── scripts/
│   ├── create_structure.py     # Crea la estructura de carpetas
│   ├── run_pipeline.py         # Datos -> entrenamiento -> explicabilidad
│   ├── setup.sh / setup.ps1    # Instalación completa (Linux/macOS / Windows)
├── src/
│   ├── config.py
│   ├── data/loader.py          # Carga por fuente (xm | synthetic) y generador sintético
│   ├── data/xm_client.py       # Cliente de la API de XM: tramos mensuales, reintentos, caché
│   ├── data/xm_loader.py       # Dataset real de Guavio (XM + NOAA), proxy del precio mínimo
│   ├── data/noaa.py            # Niño 3.4 semanal de NOAA/CPC
│   ├── data/features.py        # Vista segura por latencia (D-1, corte 18:00 de t-1, ofertas M+1)
│   ├── models/train.py         # Aportes, oferta en dos etapas, falla
│   ├── models/evaluate.py      # NSE, KGE, skill, pinball, F1, PR-AUC, terciles, Diebold-Mariano, PSI
│   ├── models/transforms.py
│   ├── explainability/explainer.py
│   └── utils/logging_utils.py
└── tests/                      # 34 pruebas: conector XM, datos, anti-leakage, métricas, SHAP, dashboard
                                # xm_mock.py: simulador de la API de XM
```

## Guía de ejecución

Requiere Python 3.11–3.13.

### Linux / macOS

```bash
cd guavio-xai
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env

python -m src.data.xm_client --check     # 0. valida métricas y códigos de Guavio contra el catálogo de XM
python -m src.data.loader --source xm --force   # 1. descarga 2016 → ayer (la primera vez tarda varios minutos)
python -m src.models.train               # 2. entrena aportes y oferta (≈2 min)
python -m src.explainability.explainer   # 3. figuras y tablas SHAP/LIME (≈30 s)
python -m pytest -q                      # 4. pruebas (34)
streamlit run app.py                     # 5. dashboard en http://localhost:8501
```

Los pasos 1 a 3 equivalen a `python scripts/run_pipeline.py --source xm --force`. Para actualizar con los datos más recientes, repita ese comando: solo descarga los meses nuevos. Sin internet: `--source synthetic`.

### Windows (PowerShell)

```powershell
cd guavio-xai
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python -m src.data.xm_client --check
python scripts/run_pipeline.py --source xm --force
python -m pytest -q
streamlit run app.py
```

Todo junto: `powershell -ExecutionPolicy Bypass -File scripts/setup.ps1`.

### Con conda

```bash
conda env create -f environment.yml
conda activate guavio-xai
python scripts/run_pipeline.py --source xm --force && streamlit run app.py
```

## Particiones temporales (config/config.yaml)

| Bloque | Periodo |
| --- | --- |
| Entrenamiento | 2016-01-01 a 2022-10-31 |
| Validación (selección, calibración, τ) | 2022-11-01 a 2024-05-31 |
| Prueba A (estrés oct-2024) | 2024-07-01 a 2025-03-31 |
| Prueba B (El Niño 2026) | 2025-04-01 a la fecha más reciente |

## Si la API de XM falla

- `python -m src.data.xm_client --check` muestra qué métrica o código no existe y sugiere alternativas del catálogo. Corrija el `id`, la `entity` o los códigos en `config/config.yaml`.
- `python -m src.data.xm_client --list aportes` busca métricas por nombre.
- Errores de red: el cliente reintenta 4 veces con espera creciente; lo ya descargado queda en caché.

## Resultados con los datos sintéticos (semilla 42)

| Modelo | Validación | Prueba A | Prueba B |
| --- | --- | --- | --- |
| Aportes (media h=1..15): NSE / cobertura P10–P90 | — | 0,71 / 87 % | 0,75 / 78 % |
| Oferta E2: F1 al mínimo | 0,67 | n/a (0,4 % de días al mínimo) | 0,55 |
| Oferta E2: MAE combinado (× mínimo) / terciles | 0,38 / 95 % | 1,41 / 100 % | 1,39 / 73 % |
| Oferta E1 (réplica): MAE / terciles | 0,44 / 94 % | 1,43 / 99 % | 1,40 / 73 % |
| Falla 7 días: F1 / PR-AUC (prevalencia ≈ 10 %) | 0,37 / 0,31 | 0,30 / 0,14 | 0,26 / 0,23 |

Como en el artículo, el error crece fuera de muestra por cambio de régimen; la ventaja de E2 sobre E1 no es estadísticamente significativa en estas pruebas (Diebold-Mariano, p > 0,3). Con datos reales es la pregunta que el proyecto debe responder.

## Ampliaciones posibles

- Precipitación de estaciones IDEAM en la cuenca del Guavio (datos.gov.co): el modelo de aportes la usa automáticamente si el dataset trae la columna `precipitacion_mm`.
- Vibración y temperatura de cojinetes desde el SCADA/CMS de la planta: con esas columnas se reactiva el modelo de fallas.
