# Día 1 — Richter's Predictor

> **Reto:** [Richter's Predictor — Modeling Earthquake Damage](https://www.drivendata.org/competitions/57/nepal-earthquake/) (DrivenData)
> **Leaderboard:** https://www.drivendata.org/competitions/57/nepal-earthquake/leaderboard/?page=1
> **Objetivo:** Predecir el grado de daño (1: bajo / 2: medio / 3: destrucción casi total) de 260k+ edificios de Nepal tras el terremoto de Gorkha (2015) a partir de cómo estaban construidos.
> **Métrica:** F1 micro-averaged.

## Resultado final del Día 1

| | F1 micro |
|---|---|
| **Score público** | **0.7534** |
| **Rank** | **55 / ~8.800** (top 0.6%) |
| CV 5-fold (StratifiedKFold seed=42) | 0.7531 ± 0.0014 |

5 submissions a lo largo del día:

| # | Modelo | CV | Público | Rank |
|---|---|---|---|---|
| 1 | Baseline trivial (clase mayoritaria) | 0.5689 | 0.5670 | — |
| 2 | LightGBM out-of-the-box | 0.7182 | 0.7179 | — |
| 3 | LightGBM + `geos como category` + tuning | 0.7497 | 0.7488 | 491 |
| 4 | Ensemble LightGBM + CatBoost (depth=6) | 0.7518 | 0.7510 | 280 |
| 5 | **Ensemble + CatBoost beefier (depth=7) + feature engineering** | **0.7531** | **0.7534** | **55** |

## La historia en 5 pasos

### 1. Baseline honesto (F1 0.7179)

LightGBM out-of-the-box sobre las 38 features originales. Categóricas de texto como `dtype="category"`. 5-fold StratifiedKFold con seed=42 fijo (todos los experimentos posteriores comparan contra estos mismos folds).

### 2. El bug obvio (F1 0.7488, +0.0309)

`geo_level_1_id`, `geo_level_2_id`, `geo_level_3_id` venían como `int64` — LightGBM las trataba como **enteros ordinales**. Pero `geo_level_3_id` tiene 11.595 aldeas únicas; la aldea 12.345 no es "mayor" que la 12.344, es otra. Casteando a `category`, LightGBM busca splits agrupando aldeas en lugar de splits numéricos sin sentido.

Una línea de código → **+0.0309 F1 público**. La mayor ganancia individual del día.

### 3. Ensemble con CatBoost (F1 0.7510, +0.0022)

Añadimos CatBoost (depth=6, 300 iteraciones) y promediamos las probabilidades de ambos modelos por fila. CatBoost solo daba F1 CV 0.7443 — peor que LightGBM — pero **sus errores no estaban correlacionados al 100%** con los de LightGBM, así que el promedio sube +0.0020 CV / +0.0022 público sobre el mejor individual.

### 4. Target encoding OOF (NEGATIVO, -0.0028)

Intento de codificar `geo_level_*_id` con la proporción de cada clase de daño por aldea (OOF para evitar leakage). Esperaba +0.005-0.010 según lo que hace la solución del top 28.

Resultado: **-0.0028 F1 CV** vs el ensemble previo. Probé con dos niveles de smoothing bayesiano (m=20 y m=5) — sin diferencia. Conclusión: LightGBM con `astype("category")` ya hace algo equivalente internamente para alta cardinalidad. Añadir el TE explícito solo introduce ruido.

Documentado porque saber qué no funciona también es parte del trabajo.

### 5. Feature engineering + CatBoost beefier (F1 0.7534, +0.0024)

Dos cambios:

**(a) 18 features derivadas** que el árbol no puede calcular solo:

- **Agregaciones por aldea:** `geo3_count`, `geo3_mean_age`, `geo3_mean_floors`, `geo3_mean_area`, `geo3_mean_height`, `geo3_mean_families`, `geo3_pct_adobe_mud`, `geo3_pct_mud_stone`, `geo3_pct_rc_eng`, `geo3_pct_timber`.
- **Agregaciones por distrito:** `geo2_count`, `geo2_mean_age`, `geo2_mean_floors`.
- **Domain features:** `n_superstructure_types`, `volume_proxy = area × height`, `age_per_floor`, `age_minus_geo3_mean`, `floors_minus_geo3_mean` (lo "extraño" que es un edificio respecto a su aldea).

LightGBM solo con feateng → CV 0.7510 (+0.0012 sobre sin feateng).

**(b) CatBoost beefier:** depth 6 → 7, iteraciones 300 → 600 con early stopping. Pasa de F1 CV 0.7443 a 0.7480 (+0.0037).

Ensemble final: media simple de probabilidades de los dos modelos por fila. CV 0.7531, público **0.7534**. Rank **55**.

## Lo que NO movió la aguja (también probado)

- **Hyperparameter tuning fino** de LightGBM (num_leaves=127, lr=0.03, feature_fraction=0.85). CV +0.0008, público marginal — el tuning solo gana algo cuando ya no quedan fugas de señal por tapar.
- **Multi-seed LightGBM** (3 seeds promediadas). +0.0002 CV. Los 3 modelos son funcionalmente idénticos a seed única.
- **Stacking con logistic regression** sobre LightGBM + CatBoost. 0.7530 vs 0.7531 → indistinguible. Con solo 2 modelos base no hay suficiente diversidad para que el meta-modelo mejore sobre el promedio simple.

## El pipeline final

[pipeline.py](pipeline.py) implementa el modelo de la fila 5 entero (CV + retraining + submission):

```
DATOS:    260k train + 87k test (DrivenData)
             ↓
FEATURES: 38 originales + 18 agregaciones + 5 domain  =  56 features
             ↓
FOLD CV (5-fold StratifiedKFold seed=42):
   ┌────────────────────────────────────┐    ┌──────────────────────────────────┐
   │ LightGBM (feateng)                 │    │ CatBoost beefier                 │
   │ n_est=250, lr=0.03                 │    │ iter=600, lr=0.04, depth=7       │
   │ num_leaves=127, min_data=40        │    │ early_stopping=80                │
   │ feature_frac=0.85, bagging_frac=.85│    │ MultiClass loss                  │
   │ category dtype para 11 cats        │    │ cat_features (incluye geos)      │
   │ → 3 probas por fila                │    │ → 3 probas por fila              │
   └────────────────────────────────────┘    └──────────────────────────────────┘
                       ↓                                       ↓
                       └────────────  AVG  ────────────────────┘
                                      ↓
                                argmax(probas) → predicción 1/2/3
```

## Cómo reproducir

```bash
# 1. Crear cuenta en DrivenData y descargar los 4 CSVs:
#    https://www.drivendata.org/competitions/57/nepal-earthquake/data/
#    Guardar en data/ a nivel de la raíz del repo.

# 2. Instalar dependencias:
pip install pandas numpy scikit-learn lightgbm catboost matplotlib

# 3. Ejecutar el pipeline:
python day_01/day01.py      # baseline + EDA + class_distribution.png (~2 min)
python day_01/pipeline.py   # ensemble final (~15-20 min CPU)

# Output: day_01/submissions/submission.csv  →  subir manualmente al leaderboard.
```

Sin GPU. Todo CPU.

## Lo que viene

- Próximo paso obvio: añadir XGBoost al ensemble (más diversidad → stacking sí puede aportar).
- Después: posible NN para embeddings densos de `geo_level_3_id` (el truco que separa top 30 del resto del leaderboard).
- Objetivo razonable con CPU: público 0.755-0.758.
