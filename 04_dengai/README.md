# Proyecto 4 — DengAI: Predicting Disease Spread

> **Competición:** [DengAI: Predicting Disease Spread](https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/) (DrivenData)
> **Tarea:** Predecir el número de casos semanales de dengue en San Juan (Puerto Rico) e Iquitos (Perú) a partir de variables climáticas y ambientales (temperatura, precipitación, vegetación NDVI, humedad…).
> **Métrica:** MAE (↓ mejor).
> **Submission:** CSV (`city, year, weekofyear, total_cases`).
> **Estado:** practice abierta — leaderboard vivo, rank real.

## Por qué este proyecto

Primer problema de **regresión + serie temporal** del portfolio. Los slots 1–3 son clasificación (tabular, imagen) y generación (texto); aquí entra la familia de forecasting, la más común en industria y la que conecta con economía.

## Plan

- [ ] EDA: dos ciudades con dinámicas y escalas muy distintas → casi seguro modelar por separado (o con feature de ciudad).
- [ ] **Validación temporal** (TimeSeriesSplit / corte por fecha). CV aleatoria sería leakage: no puedes "ver el futuro" para predecir el pasado.
- [ ] Features de **lag y rolling** sobre las climáticas (la transmisión del dengue va retrasada respecto al clima — mosquitos).
- [ ] Baseline: media estacional / persistencia → luego gradient boosting (LightGBM) con features temporales.
- [ ] Vigilar el gap validación→leaderboard (distribución no estacionaria entre train y test).

## Resultado

_Pendiente._
