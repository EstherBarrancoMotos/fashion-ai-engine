# Fashion AI Engine

> **A Machine Learning decision engine that helps fashion brands reduce overproduction and returns through demand forecasting, return prediction and economic simulation.**

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

The output is not just a model — it is a quantified business case.

### Key results

| Metric | Value | Notes |
|---|---|---|
| Demand WAPE | **0.515** | LightGBM on 1.76M article × week rows. +7.5% over `lag_1` baseline |
| Returns ROC-AUC | **0.64** | Semi-synthetic labels (industry rates) — see methodology |
| Margin uplift | **+175 K€** | Optimized policy vs. naive human baseline at constant production budget |
| Returns avoided | **5.6 K units** | At the same total production volume |
| Style clusters | **4** | Selected via silhouette analysis |
| Models compared | **5 supervised + 1 unsupervised** | Pipeline + GridSearchCV with TimeSeriesSplit |

**The engine doesn't produce less — it produces *better*.** The optimized policy reallocates the same production budget across articles using expected margin per unit, not raw demand volume.

### Architecture

```
┌─────────────────────┐
│   H&M dataset       │  31.8M transactions · 105K articles · 2018-2020
│   (Kaggle)          │
└──────────┬──────────┘
           │
   ┌───────┴────────┐
   │ Feature Store  │  weekly_sales (article × week)
   └───────┬────────┘
           │
   ┌───────┼─────────────────────────────┐
   ▼       ▼              ▼              ▼
┌──────┐ ┌──────┐  ┌──────────────┐ ┌──────────┐
│Demand│ │Return│  │   KMeans     │ │   EDA    │
│LGBM  │ │LGBM  │  │  k=4 styles  │ │  Style   │
└───┬──┘ └───┬──┘  └──────┬───────┘ └─────┬────┘
    │       │             │               │
    └───────┴──────┬──────┴───────────────┘
                   ▼
         ┌──────────────────────┐
         │ Economic Simulator   │   margin = f(demand, returns, costs)
         │ + lost-demand penalty│
         └────────┬─────────────┘
                  ▼
         ┌──────────────────┐
         │  Streamlit App   │   5 pages · live policy simulation
         └──────────────────┘
```

### Project structure

```
fashion-ai-engine/
├── data/
│   ├── raw/                     # H&M CSVs (gitignored)
│   ├── processed/               # parquets + clusters (gitignored)
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
│   ├── data_processing.py       # raw → processed pipeline
│   ├── training.py              # full training pipeline (cold-start, lags, log target)
│   ├── evaluation.py            # WAPE on test set
│   ├── config.py                # economic assumptions
│   └── simulator/               # economic model with lost-demand penalty
├── models/
│   ├── final_model.pkl                   # production model (LightGBM)
│   ├── trained_model_1_dummy.pkl         # baseline
│   ├── trained_model_2_ridge.pkl
│   ├── trained_model_3_randomforest.pkl
│   ├── trained_model_4_xgboost.pkl
│   ├── trained_model_5_lightgbm.pkl
│   └── kmeans_clusters.pkl
├── app_streamlit/
│   ├── streamlit_app.py
│   ├── pages/
│   ├── utils/
│   └── requirements.txt
├── tests/
│   └── test_simulator.py
├── reports/
│   ├── figures/
│   └── results/                 # JSON/CSV with metrics & comparisons
├── docs/
│   ├── memoria.md
│   ├── negocio.pptx
│   └── ds.pptx
├── configs/default.yaml
├── requirements.txt
├── .gitignore
└── README.md
```

### Quickstart

```bash
# 1. Clone
git clone https://github.com/EstherBarrancoMotos/fashion-ai-engine.git
cd fashion-ai-engine

# 2. Virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 3. Install
pip install -r requirements.txt

# 4. Get the H&M dataset (Kaggle API + competition agreement required)
mkdir data\raw
kaggle competitions download -c h-and-m-personalized-fashion-recommendations -p data/raw
cd data\raw && Expand-Archive *.zip . && cd ..\..

# 5. Run the pipeline
python -m src.data_processing
python -m src.training
python src\evaluation.py

# 6. Launch the Streamlit app
streamlit run app_streamlit/streamlit_app.py
```

### Methodology & honest limitations

**Modelling**

- **Demand**: LightGBM with log-transformed target, native categorical handling, strict temporal validation (train ≤ 2020-06-22, val ≤ 2020-08-22, test > 2020-08-22). Cold-start filter excludes products with <4 weeks of training history.
- **Returns**: LightGBM classifier with class imbalance handling. Labels are semi-synthetic, derived from sectoral return rates (Narvar 2023, Optoro 2023).
- **Comparative study (notebook 03)**: 5 supervised algorithms (Dummy, Ridge, RandomForest, XGBoost, LightGBM) on a 100K stratified sample, each wrapped in `sklearn.Pipeline` and tuned via `GridSearchCV` with `TimeSeriesSplit(3)`.
- **Clustering**: KMeans on one-hot encoded product attributes, k chosen via silhouette analysis.

**Economic assumptions** (`src/config.py`)

| Parameter | Value | Source |
|---|---|---|
| Gross margin | 53% | H&M Annual Report 2023 |
| Return handling cost | 18 €/unit | Optoro 2023 |
| Markdown depth | 40% | McKinsey State of Fashion 2024 |
| Destruction rate | 15% | Industry consensus |

**Honest limitations**

- Returns labels are semi-synthetic. H&M does not publish return data; the model learns realistic patterns but is biased to the assumed sectoral rates.
- The economic simulator is static — no inter-temporal inventory dynamics, no cross-product substitution, no price elasticity.
- The demand cap (2.5× prediction) is a hyperparameter, not derived from data.
- Comparative model study uses a 100K sample for tractable GridSearch; the production model is retrained on the full 2.1M dataset.

### Tech stack

| Layer | Tools |
|---|---|
| Data | pandas, polars, pyarrow |
| Modelling | scikit-learn, LightGBM, XGBoost |
| Visualization | Plotly, matplotlib, seaborn |
| App | Streamlit, Streamlit Community Cloud |
| Experiment tracking | MLflow |
| Testing | pytest |

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
- Tasas de devolución online del 15–40% según categoría (Narvar, Optoro)
- Decisiones de producción sin cuantificación del riesgo por producto
- Falta de herramientas que traduzcan predicciones de ML en impacto en €

El coste agregado se cuenta en miles de millones y el impacto medioambiental es significativo.

### La solución

**Fashion AI Engine** es un sistema modular de ML que convierte ventas históricas y metadatos de producto en decisiones accionables de producción:

1. **Modelo de demanda** — predice unidades vendidas semanales por artículo
2. **Modelo de devoluciones** — predice probabilidad de devolución por artículo
3. **Simulador económico** — traduce ambas predicciones en € de margen esperado bajo distintas políticas
4. **Clustering KMeans** — segmenta los 75K productos en arquetipos de estilo
5. **App Streamlit** — permite a un usuario no técnico explorar los datos y ejecutar escenarios en tiempo real

El output no es solo un modelo: es un caso de negocio cuantificado.

### Resultados clave

| Métrica | Valor | Nota |
|---|---|---|
| WAPE demanda | **0.515** | LightGBM sobre 1.76M filas. +7.5% sobre baseline `lag_1` |
| ROC-AUC devoluciones | **0.64** | Labels semi-sintéticas (tasas sectoriales) |
| Margen incremental | **+175 K€** | Política optimizada vs. baseline humano a presupuesto constante |
| Devoluciones evitadas | **5.6 K unidades** | Al mismo volumen total de producción |
| Clusters de estilo | **4** | Seleccionados por silhouette |
| Modelos comparados | **5 supervisados + 1 no supervisado** | Pipeline + GridSearchCV |

**El motor no produce menos: produce mejor.** La política optimizada reasigna el mismo presupuesto de producción entre artículos usando margen neto esperado por unidad, no volumen bruto de demanda.

### Estructura del proyecto

Ver la sección [Project structure](#project-structure) más arriba — el árbol es idéntico.

### Quickstart

Ver la sección [Quickstart](#quickstart) más arriba — los comandos son idénticos.

### Metodología y limitaciones honestas

**Modelado**

- **Demanda**: LightGBM con target log-transformado, manejo nativo de categóricas, validación temporal estricta (train ≤ 2020-06-22, val ≤ 2020-08-22, test > 2020-08-22). Filtro de cold-start excluye productos con <4 semanas de historial.
- **Devoluciones**: LightGBM classifier con manejo de desbalance. Labels semi-sintéticas derivadas de tasas sectoriales (Narvar 2023, Optoro 2023).
- **Estudio comparativo (notebook 03)**: 5 algoritmos supervisados (Dummy, Ridge, RandomForest, XGBoost, LightGBM) sobre sample estratificado de 100K, cada uno envuelto en `sklearn.Pipeline` y tuneado con `GridSearchCV` y `TimeSeriesSplit(3)`.
- **Clustering**: KMeans sobre atributos categóricos one-hot, k elegido por análisis de silhouette.

**Supuestos económicos** (`src/config.py`)

| Parámetro | Valor | Fuente |
|---|---|---|
| Margen bruto | 53% | H&M Annual Report 2023 |
| Coste devolución | 18 €/ud | Optoro 2023 |
| Markdown medio | 40% | McKinsey State of Fashion 2024 |
| Tasa destrucción | 15% | Consenso de industria |

**Limitaciones honestas**

- Las labels de devolución son semi-sintéticas. H&M no publica datos de devolución; el modelo aprende patrones realistas pero está sesgado por las tasas sectoriales asumidas.
- El simulador económico es estático: no modela dinámicas de inventario inter-temporales, sustitución entre productos, ni elasticidad de precio.
- El cap de demanda (2.5× predicción) es un hiperparámetro, no derivado de datos.
- El estudio comparativo usa sample de 100K para que el GridSearch sea tratable; el modelo de producción se reentrena sobre el dataset completo de 2.1M.

### Autor

Proyecto desarrollado por **Esther Barranco Motos** como entrega final del módulo de Machine Learning del bootcamp de Data Science.

LinkedIn: *(próximamente)*

### Licencia

Uso educativo. El dataset de H&M está sujeto a sus propios términos (Kaggle).
