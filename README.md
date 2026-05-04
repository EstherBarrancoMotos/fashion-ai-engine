# Fashion AI Engine

> **A Machine Learning decision engine that helps fashion brands reduce overproduction and returns through demand forecasting, return prediction and economic simulation.**

🚀 **[Live demo · Streamlit Cloud →](https://fashion-ai-engine-app.streamlit.app/)**

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-2C8EBB)](https://lightgbm.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.36+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-Educational-blue)](#license)

---

## 🌍 Languages

[**🇬🇧 English**](#-english) · [**🇪🇸 Español**](#-español)

---

## 🇬🇧 English

### The problem

The fashion industry destroys or heavily discounts an estimated **30% of its production every year**. The drivers are well known:

- Inaccurate demand forecasts
- Online return rates of 15–40% depending on category (Narvar, Optoro)
- Production decisions made without quantified risk per product
- No tooling to translate ML predictions into € impact

The aggregate cost is in the billions and the sustainability impact is significant.

### The solution

**Fashion AI Engine** is a modular ML system that turns historical sales and product metadata into actionable production decisions:

1. **Demand model** — predicts weekly units sold per article
2. **Returns model** — predicts probability of return per article
3. **Economic simulator** — translates the two predictions into € of expected margin under different production policies
4. **KMeans clustering** — segments 75K products into style archetypes for portfolio analysis
5. **Streamlit decision app** — lets a non-technical user explore the data and run policy scenarios in real time

### Key results

| Metric | Value | Notes |
|---|---|---|
| Demand WAPE | **0.515** | LightGBM on 1.76M article × week rows. +7.5% over `lag_1` baseline |
| Returns ROC-AUC | **0.64** | Semi-synthetic labels (industry rates) |
| Margin uplift | **+175 K€** | Optimized policy vs. naive human baseline at constant production budget |
| Returns avoided | **5.6 K units** | At the same total production volume |
| Style clusters | **4** | Selected via silhouette analysis |
| Models compared | **5 supervised + 1 unsupervised** | Pipeline + GridSearchCV with TimeSeriesSplit |

**The engine doesn't produce less — it produces *better*.**

### The Streamlit app

Five pages explore every angle of the system:

| Page | What it shows |
|---|---|
| **Executive Dashboard** | Headline KPIs, 4-scenario comparison, alpha sensitivity |
| **Decision Simulator** | Interactive sliders, live P&L recalculation |
| **Style Intelligence** | Garment × colour heatmap, KMeans cluster profile, drilldown |
| **Returns Risk** | Model metrics, calibration plot, top 50 risky products (CSV download) |
| **Demand Explorer** | Forecast vs actual per product, error distribution, worst predictions |

Try it live: **[fashion-ai-engine-app.streamlit.app](https://fashion-ai-engine-app.streamlit.app/)**

### Project structure

```
fashion-ai-engine/
├── data/
│   ├── raw/                     # H&M CSVs (gitignored)
│   ├── processed/               # parquets (small ones tracked for deploy)
│   ├── train/                   # temporal split (gitignored)
│   └── test/
├── notebooks/
│   ├── 01_Fuentes.ipynb                  # data sources & industry benchmarks
│   ├── 02_LimpiezaEDA.ipynb              # EDA + feature store
│   ├── 03_Entrenamiento_Evaluacion.ipynb # 5 supervised + KMeans + GridSearch
│   ├── 03b_Demand_Full_Dataset.ipynb     # LightGBM on the full 2.1M dataset
│   ├── 03c_Returns_Model.ipynb           # return probability classifier
│   └── 03d_Economic_Simulator.ipynb      # policy P&L simulation
├── src/
│   ├── data_processing.py
│   ├── training.py
│   ├── evaluation.py
│   ├── config.py
│   └── simulator/
├── models/                      # 7 .pkl files
├── app_streamlit/
│   ├── streamlit_app.py
│   ├── pages/                   # 4 pages
│   ├── utils/
│   └── requirements.txt
├── tests/
├── reports/
├── docs/
│   ├── memoria.md
│   ├── negocio.pptx
│   └── ds.pptx
├── configs/default.yaml
├── requirements.txt
└── README.md
```

### Quickstart

```bash
git clone https://github.com/EstherBarrancoMotos/fashion-ai-engine.git
cd fashion-ai-engine

python -m venv .venv
.venv\Scripts\activate          # Windows

pip install -r requirements.txt

# Download dataset (Kaggle API + competition agreement required)
mkdir data\raw
kaggle competitions download -c h-and-m-personalized-fashion-recommendations -p data/raw
cd data\raw && Expand-Archive *.zip . && cd ..\..

# Run pipeline
python -m src.data_processing
python -m src.training
python src\evaluation.py

# Launch app locally
streamlit run app_streamlit/streamlit_app.py
```

### Methodology & honest limitations

**Modelling**

- **Demand**: LightGBM with log-transformed target, native categorical handling, strict temporal validation (train ≤ 2020-06-22, val ≤ 2020-08-22, test > 2020-08-22). Cold-start filter.
- **Returns**: LightGBM classifier with class imbalance handling. Labels semi-synthetic.
- **Comparative study**: 5 supervised algorithms on a 100K stratified sample, each wrapped in `sklearn.Pipeline` and tuned via `GridSearchCV` with `TimeSeriesSplit(3)`.
- **Clustering**: KMeans on one-hot encoded product attributes, k chosen via silhouette analysis.

**Economic assumptions** (`src/config.py`)

| Parameter | Value | Source |
|---|---|---|
| Gross margin | 53% | H&M Annual Report 2023 |
| Return handling cost | 18 €/unit | Optoro 2023 |
| Markdown depth | 40% | McKinsey State of Fashion 2024 |
| Destruction rate | 15% | Industry consensus |

**Honest limitations**

- Returns labels are semi-synthetic. H&M does not publish return data.
- The economic simulator is static (no inter-temporal dynamics, no substitution, no price elasticity).
- The interactive simulator interpolates over the precomputed sensitivity curve rather than performing per-article live recomputation.

### Tech stack

scikit-learn, LightGBM, XGBoost, pandas, Plotly, Streamlit, MLflow, pytest.

### Author

Built by **Esther Barranco Motos** as a portfolio project for the Data Science bootcamp final module.

LinkedIn: *(coming soon)*

### License

Educational use only. The H&M dataset is subject to its own competition terms (Kaggle).

---
---

## 🇪🇸 Español

### El problema

La industria de la moda destruye o rebaja masivamente alrededor del **30% de su producción cada año**. Las causas son conocidas:

- Forecasts de demanda imprecisos
- Tasas de devolución online del 15–40% según categoría
- Decisiones de producción sin cuantificación del riesgo por producto
- Falta de herramientas que traduzcan predicciones de ML en impacto en €

### La solución

**Fashion AI Engine** es un sistema modular de ML que convierte ventas históricas y metadatos de producto en decisiones accionables de producción:

1. **Modelo de demanda** — predice unidades vendidas semanales por artículo
2. **Modelo de devoluciones** — predice probabilidad de devolución por artículo
3. **Simulador económico** — traduce ambas predicciones en € de margen bajo distintas políticas
4. **Clustering KMeans** — segmenta los 75K productos en arquetipos de estilo
5. **App Streamlit** — permite a un usuario no técnico explorar los datos y ejecutar escenarios en tiempo real

### Resultados clave

| Métrica | Valor | Nota |
|---|---|---|
| WAPE demanda | **0.515** | LightGBM sobre 1.76M filas. +7.5% sobre baseline `lag_1` |
| ROC-AUC devoluciones | **0.64** | Labels semi-sintéticas |
| Margen incremental | **+175 K€** | Política optimizada vs. baseline humano |
| Devoluciones evitadas | **5.6 K unidades** | Al mismo volumen de producción |
| Clusters de estilo | **4** | Seleccionados por silhouette |
| Modelos comparados | **5 supervisados + 1 no supervisado** | Pipeline + GridSearchCV |

**El motor no produce menos: produce mejor.**

### La app Streamlit

Pruébala en vivo: **[fashion-ai-engine-app.streamlit.app](https://fashion-ai-engine-app.streamlit.app/)**

| Página | Qué muestra |
|---|---|
| Executive Dashboard | KPIs, comparativa de 4 escenarios, sensibilidad a α |
| Decision Simulator | Sliders interactivos, recálculo del P&L en vivo |
| Style Intelligence | Heatmap garment × color, perfil de clusters, drilldown |
| Returns Risk | Métricas del modelo, calibración, top 50 productos riesgo |
| Demand Explorer | Forecast vs real por producto, distribución del error |

### Documentación

Ver `docs/memoria.md` para la memoria completa del proyecto y `docs/negocio.pptx` + `docs/ds.pptx` para las presentaciones.

### Autor

Proyecto desarrollado por **Esther Barranco Motos** como entrega final del módulo de Machine Learning del bootcamp de Data Science.

LinkedIn: *(próximamente)*

### Licencia

Uso educativo. El dataset de H&M está sujeto a sus propios términos (Kaggle).
