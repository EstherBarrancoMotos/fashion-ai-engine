# Memoria · Fashion AI Engine

**Autora:** Esther Barranco Motos
**Bootcamp:** Data Science · Módulo final de Machine Learning
**Entrega:** 4 de mayo de 2026
**Repositorio:** https://github.com/EstherBarrancoMotos/fashion-ai-engine

---

## 1. Resumen ejecutivo

**Fashion AI Engine** es un motor de decisión basado en Machine Learning que ayuda a marcas de moda a reducir la sobreproducción y las devoluciones a través de tres componentes integrados: un modelo de demanda, un modelo de devoluciones y un simulador económico que traduce las predicciones en impacto cuantificado en euros.

El proyecto demuestra una pipeline ML completa de extremo a extremo —desde la adquisición de los datos crudos hasta una aplicación web desplegada— sobre un dataset real y de tamaño industrial: **31,8 millones de transacciones** de H&M correspondientes al periodo septiembre 2018 – septiembre 2020.

**Resultado clave de negocio:** la política optimizada genera **+175.535 €** de margen adicional al mismo presupuesto de producción que el baseline humano (+6,8%), evitando 5.612 unidades de stock muerto.

---

## 2. Planteamiento del problema

### 2.1. Contexto de negocio

La industria de la moda destruye o rebaja masivamente cerca del 30% de su producción anual. Las causas son conocidas: forecasts de demanda imprecisos, tasas de devolución del 15-40% en categorías online, y decisiones de producción tomadas sin cuantificación rigurosa del riesgo por producto.

Los responsables de compras (buyers) actúan a menudo con intuición y datos agregados, sin una herramienta que cuantifique en euros el impacto de mover una unidad de un artículo a otro.

### 2.2. Pregunta de investigación

¿Es posible construir un sistema que, manteniendo constante el presupuesto total de unidades a producir, lo reasigne entre artículos de forma que maximice el margen económico esperado teniendo en cuenta tanto la demanda predicha como la probabilidad de devolución?

### 2.3. Definición técnica

El proyecto combina tres problemas de ML:

- **Regresión** (predicción de demanda semanal por artículo)
- **Clasificación** (probabilidad de devolución por artículo)
- **Clustering no supervisado** (segmentación de productos por estilo)

Y los integra en un simulador económico que toma decisiones de producción a partir de las salidas de los modelos.

---

## 3. Datos

### 3.1. Fuente principal

**H&M Personalized Fashion Recommendations** (Kaggle competition, 2022).

| Tabla | Filas | Periodo |
|---|---|---|
| `transactions_train.csv` | 31.788.324 | sept 2018 – sept 2020 |
| `articles.csv` | 105.542 | catálogo H&M |
| `customers.csv` | 1.371.980 | clientes activos |

### 3.2. Fuentes secundarias (industria)

Los supuestos económicos se derivan de fuentes públicas:

- **H&M Annual Report 2023** → margen bruto 53%
- **Optoro Returns Report 2023** → coste medio de devolución 18 €/ud
- **Narvar Consumer Returns Report 2023** → tasas sectoriales de devolución
- **McKinsey State of Fashion 2024** → markdown depth medio 40%

### 3.3. Volumen final del dataset de modelado

Tras feature engineering, el dataset principal `weekly_sales.parquet` tiene **2.210.899 filas × 13 columnas** a granularidad `article × week`. Tras aplicar el filtro cold-start y las features derivadas (lags, rolling means, calendario), el dataset usado para modelar tiene **1.758.271 filas × 26 columnas**, muy por encima del mínimo del enunciado (1.000 × 10).

---

## 4. Metodología

### 4.1. Adquisición y preparación

Los CSVs originales se descargan vía Kaggle API y se persisten en `data/raw/`. El script `src/data_processing.py` los procesa: agrega transacciones a granularidad article × week, hace join con metadatos de producto y persiste el resultado en `data/processed/weekly_sales.parquet`.

El feature engineering se documenta en `notebooks/02_LimpiezaEDA.ipynb` y se reproduce en `src/training.py`. Las features creadas:

- **Lags temporales**: `lag_1`, `lag_2`, `lag_4` (autocorrelación corta y mensual)
- **Rolling means**: `rolling_mean_4`, `rolling_mean_8` (tendencia suavizada)
- **Calendario**: `month`, `week_of_year` (estacionalidad)
- **Categóricas**: `product_group_name`, `colour_group_name`, `garment_group_name`, `department_name`, `index_group_name` (atributos del producto)
- **Log-transform**: target y lags transformados con `log1p` (distribución muy asimétrica)
- **Filtro cold-start**: se eliminan productos con menos de 4 semanas de histórico en train para evitar predicciones espurias

### 4.2. Validación temporal estricta

Tres splits temporales en lugar de aleatorios para evitar fuga de información del futuro:

- **Train**: hasta 2020-06-22 → 1.590.749 filas
- **Validation**: 2020-06-22 a 2020-08-22 → 119.594 filas
- **Test**: posterior a 2020-08-22 → 47.928 filas

### 4.3. Modelado

#### Estudio comparativo (notebook 03)

Cinco algoritmos supervisados entrenados sobre un sample estratificado de 100.000 filas, cada uno envuelto en `sklearn.Pipeline` y tuneado con `GridSearchCV` y `TimeSeriesSplit(n_splits=3)`.

| # | Modelo | Test WAPE |
|---|---|---|
| 1 | DummyRegressor (baseline) | 0,875 |
| 2 | Ridge | 0,898 |
| 3 | RandomForest | **0,484** |
| 4 | XGBoost | 0,538 |
| 5 | LightGBM | 0,538 |

Conclusión metodológica: los modelos basados en árboles superan ampliamente al baseline naive y a la regresión lineal. RandomForest y los gradient boosters quedan empatados en sample reducido.

#### Modelo de producción (notebook 03b)

LightGBM reentrenado sobre el dataset completo (1,76M filas tras filtro cold-start) usando categoricals nativos. Selección por velocidad y robustez con categóricas de alta cardinalidad.

- **WAPE final: 0,515** (mejora 7,5% sobre baseline `lag_1`)
- Convergencia en iteración 273 con early stopping
- Validación L1 en escala log: 0,347

#### Modelo de devoluciones (notebook 03c)

LightGBM classifier con manejo de desbalance (`is_unbalance=True`).

- **ROC-AUC: 0,637**
- **PR-AUC: 0,314**
- **Brier score: 0,170**

Las labels de devolución son **semi-sintéticas**, derivadas de tasas sectoriales reales por categoría (Narvar, Optoro). H&M no publica datos de devolución, por lo que esto representa una limitación honesta del proyecto.

#### Modelo no supervisado (notebook 03)

KMeans con k=4, seleccionado vía silhouette analysis (silhouette = 0,122 en k=4, máximo dentro del rango 3-8). Los 4 clusters representan arquetipos de producto basados en `garment_group_name`, `colour_group_name`, `department_name`, `index_group_name` y `product_group_name`.

### 4.4. Simulador económico (notebook 03d)

El simulador combina las salidas de los modelos en un P&L cuantificado. Para cada artículo:

```
margin = revenue_full + revenue_markdown - COGS - return_cost - lost_demand_penalty
```

Donde:
- `revenue_full = vendido_a_precio_completo × precio × (1 - return_prob)`
- `revenue_markdown = stock_no_vendido × precio × (1 - markdown_pct)`
- `COGS = producción × precio × (1 - margen_bruto)`
- `return_cost = devoluciones × 18 €`
- `lost_demand_penalty = demanda_perdida × margen_unitario`

El **lost demand penalty** se incorporó en V2 del simulador para evitar que la política optimizada "haga trampa" produciendo menos. Este es un fix conceptual no trivial detectado durante la iteración.

### 4.5. Comparativa de políticas

Se evalúan cuatro políticas de producción:

| Política | Lógica | Uso |
|---|---|---|
| **Oracle** | `actual_demand × 1.2` | Techo teórico (perfect info) |
| **Naive baseline** | `actual_demand + ruido gaussiano σ=25%` | Forecast humano sin ML |
| **Forecast-only (ML)** | Predicción del modelo de demanda | Sin lógica de devoluciones |
| **Optimized (smart, α=0,1)** | Reasigna el presupuesto del Oracle según `demand × (1 - α × return_prob)` | Sistema completo |

---

## 5. Resultados

### 5.1. Resultados técnicos

| Componente | Métrica | Valor | Mejora vs baseline |
|---|---|---|---|
| Modelo de demanda | WAPE | 0,515 | +7,5% |
| Modelo de devoluciones | ROC-AUC | 0,637 | (sin baseline directo) |
| Clustering | Silhouette | 0,122 | (k=4 elegido) |

### 5.2. Resultados de negocio

| Escenario | Producción (uds) | Stock no vendido | Margen total |
|---|---|---|---|
| Oracle | 505.538 | 84.256 | **2.955.347 €** |
| Naive baseline | 504.594 | 103.017 | 2.565.575 € |
| Forecast-only (ML) | 503.043 | 94.610 | 2.667.790 € |
| **Optimized (smart, α=0,1)** | **505.538** | **97.404** | **2.741.111 €** |

**Impacto de la política optimizada vs naive baseline:**

- **Margen incremental: +175.536 € (+6,84%)**
- Producción reasignada al mismo presupuesto total
- 5.612 unidades de stock muerto evitadas
- Coste de devolución reducido en 25.206 €

### 5.3. Sensibilidad al parámetro α

El barrido de α ∈ [0,0; 1,0] muestra que el óptimo está en **α = 0,1**. Valores mayores destruyen valor al desviar producción de productos populares, aunque éstos tengan alta probabilidad de devolución.

---

## 6. Implementación: aplicación web

Se ha desarrollado una aplicación Streamlit con cinco páginas que permite a un usuario no técnico explorar los datos y simular escenarios:

1. **Executive Dashboard**: KPIs, comparativa de escenarios, sensibilidad a α
2. **Decision Simulator**: sliders interactivos sobre los hiperparámetros, recálculo del P&L en vivo
3. **Style Intelligence**: heatmap garment × color, análisis por cluster, drilldown por categoría
4. **Returns Risk**: métricas del modelo, distribución del riesgo, calibración, top 50 productos riesgo
5. **Demand Explorer**: forecast vs real por producto, distribución del error, productos peor predichos

El diseño visual sigue una estética SaaS premium (tipografía Inter, paleta restringida, KPI cards con sombra sutil) consistente con productos comerciales como Stripe o PostHog.

---

## 7. Limitaciones honestas

El proyecto se ha desarrollado con rigor y transparencia. Las limitaciones identificadas son:

1. **Labels de devolución semi-sintéticas.** H&M no publica datos de devolución. El modelo aprende patrones realistas a partir de tasas sectoriales, pero está sesgado hacia los supuestos asumidos. Una v2 con datos reales (por ejemplo ASOS GraphReturns) eliminaría este sesgo.

2. **Simulador económico estático.** No modela dinámicas inter-temporales de inventario, sustitución entre productos, ni elasticidad de precio. Una versión más sofisticada incorporaría estos efectos.

3. **El cap de demanda (2,5× predicción) es un hiperparámetro.** No se deriva de los datos. Una calibración rigurosa requeriría análisis adicional de saturación.

4. **Estudio comparativo en sample reducido.** La comparativa de los 5 modelos del notebook 03 usa 100K filas para que el GridSearch sea tratable. El modelo de producción se reentrena sobre el dataset completo (1,76M).

5. **Simulador interactivo de la app.** Por restricciones de tiempo de respuesta, los sliders interpolan sobre la curva de sensibilidad precomputada en lugar de recalcular física por artículo. Una v2 desacoplaría el cálculo en un servicio asíncrono.

---

## 8. Próximos pasos

- Integrar datos reales de devoluciones (ASOS GraphReturns u otros)
- Modelar elasticidad de precio y dinámicas de stock
- Recomputación física en el simulador interactivo (pipeline asíncrono)
- Bayesian optimization de α por categoría (no global)
- Integrar el sistema con un sistema de información de almacén real

---

## 9. Conclusión

El proyecto demuestra que un sistema de Machine Learning bien diseñado puede aportar valor cuantificado en una industria tradicionalmente regida por intuición. La narrativa clave —**no producir menos, producir mejor**— es defendible con números: a presupuesto constante, la política optimizada genera 175.536 € de margen adicional anuales.

Más allá de las cifras, el proyecto muestra un proceso completo de Data Science: planteamiento de problema, adquisición de datos a escala, feature engineering riguroso, comparativa metodológica de modelos, evaluación honesta, integración en un producto usable y comunicación de resultados orientada a negocio.
