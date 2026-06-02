# Proyecto 4 — DengAI: Predicting Disease Spread

> **Competición:** [DengAI: Predicting Disease Spread](https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/) (DrivenData)
> **Tarea:** Predecir el número de casos semanales de dengue en San Juan (Puerto Rico) e Iquitos (Perú) a partir de variables climáticas y ambientales (temperatura, precipitación, vegetación NDVI, humedad…).
> **Métrica:** MAE (↓ mejor).
> **Submission:** CSV (`city, year, weekofyear, total_cases`).
> **Estado:** practice abierta — leaderboard vivo, rank real.

## Por qué este proyecto

Primer problema de **regresión + serie temporal** del portfolio. Los slots 1–3 son clasificación (tabular, imagen) y generación (texto); aquí entra la familia de forecasting, la más común en industria y la que conecta con economía.

## Plan

- [x] EDA ([eda.py](eda.py) → [eda.png](eda.png)): dos ciudades con escala/estacionalidad/histórico distintos → **modelamos por separado**. Test = bloque de futuro puro. Target muy asimétrico (skew ~4). Autocorrelación alta pero no usable (holdout futuro largo). Missing en clima (ndvi_ne 13%).
- [x] **Validación temporal** (holdout: primeras 75% semanas → train, últimas 25% → val). CV aleatoria sería leakage.
- [x] Features de **rolling** sobre drivers climáticos (humedad/temp/precip; el clima precede a los casos) + estacionalidad `weekofyear` (sin/cos). **Sin lags del target** (no disponibles en el test futuro).
- [x] Baseline ([pipeline.py](pipeline.py)): seasonal-naive como referencia honesta → LightGBM L1 (objetivo MAE) por ciudad, clip≥0 + redondeo a entero.
- [ ] Mejoras pendientes: lags explícitos del clima (no solo rolling), modelo de conteo (Poisson/NegBinomial o Tweedie), suavizado de la predicción, blend con seasonal-naive en iq.
- [ ] Vigilar el gap validación→leaderboard (distribución no estacionaria entre train y test).

## Resultado

Validación temporal (MAE, menor = mejor):

| Ciudad | seasonal-naive | LightGBM | mejora |
|---|---|---|---|
| sj | 24.86 | **16.89** | −7.97 |
| iq | 7.71 | **7.62** | −0.08 |

- **sj** mejora mucho sobre la referencia estacional; **iq** apenas (serie más ruidosa, ~18% de semanas a 0, clima menos predictivo).
- _LB público: pendiente de submitear `submissions/submission.csv` en DrivenData._
