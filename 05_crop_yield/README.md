# Proyecto 5 — CGIAR Crop Yield Prediction

> **Competición:** [CGIAR Crop Yield Prediction Challenge](https://zindi.africa/competitions/cgiar-crop-yield-prediction-challenge) (Zindi)
> **Tarea:** Predecir el **rendimiento** (yield) de cada parcela agrícola a partir de imágenes satélite **Sentinel-2** y datos de suelo/clima. **Regresión.**
> **Métrica:** RMSE (↓ mejor) — *verificar en la web de la competición*.
> **Submission:** CSV (`Field_ID, Yield`).
> **Estado:** competición cerrada; el rank se obtendría vía Zindi (late submission). Mientras, evaluamos con **validación cruzada espacial** robusta.

## Por qué este proyecto

Cierra el portfolio con un problema **geoespacial + satélite**, pero **ejecutable en CPU** (no hay GPU disponible): en vez de meter las imágenes en una CNN, se **agregan los parches satélite a features tabulares** y se modela con boosting — que además es un enfoque competitivo real en mapeo/rendimiento de cultivos.

## Datos

- `image_arrays_train` / `image_arrays_test`: un array numpy **`(360, 41, 41)`** por campo = bandas Sentinel-2 × pasos temporales (360) sobre un parche espacial de 41×41 px alrededor de la parcela.
- `fields_w_additional_info.csv`: metadatos + suelo (ISRIC SoilGrids) y clima (TERRACLIM) por campo.
- Etiqueta: `Yield` por `Field_ID` (train). ~930 MB en total.
- Mirror en Kaggle (`menziwaafrica/cgiar-crop-yield-prediction-challenge`) → descarga con credenciales de Kaggle, sin aceptar reglas. Gitignored.

## Plan

- [x] **EDA** ([eda.py](eda.py) → [eda.png](eda.png)): ver hallazgos abajo.
- [x] **Feature engineering satélite** ([pipeline.py](pipeline.py)): por campo y mes, media del parche 41×41 **enmascarando nubes con QA60** (bits 10/11); **NDVI/EVI/NDWI**; agregados temporales (mean/std/min/max) por señal + **fenología del NDVI** (mes de pico, integral, rango) + 12 NDVI mensuales + suelo ISRIC. **147 features.**
- [x] **Baseline LightGBM** (objetivo RMSE) con **GroupKFold por año**: ver resultado.
- [ ] Mejoras: KFold aleatorio (estima mejor el LB de años mezclados), añadir clima de `fields_w_additional_info`, selección de features, manejo de la cola alta del yield, percentiles del parche.
- [ ] Submission a Zindi (late) → RMSE public LB.

## Resultado

Validación **GroupKFold por año** (conservadora: predice un año no visto):

| año (fold) | RMSE |
|---|---|
| 2016 | 1.459 |
| 2017 | 1.609 |
| 2018 | 1.763 |
| 2019 | 1.800 |
| **OOF** | **1.608** |
| baseline (media) | 1.742 |

- El modelo bate a predecir la media (1.61 vs 1.74). Mejora modesta — yield desde satélite es intrínsecamente ruidoso.
- La CV por año es un **estrés** (el test real mezcla años) → el RMSE del leaderboard debería ser ≤ 1.61.
- _LB público: pendiente de subir `submissions/submission.csv` a Zindi._

## EDA — hallazgos

- **2.977 campos train / 1.055 test.** `fields_w_additional_info.csv` cubre el 100% de ambos (suelo + clima).
- **Yield**: media 3.17, std 1.74, rango 0–14.45; **sesgado a la derecha** (cola de valores altos).
- **4 años** (2016: 1024, 2017: 1203, 2018: 181, 2019: 569) → validación **GroupKFold por año**.
- `Quality` (1/2/3) = calidad de la *etiqueta*, no se correlaciona fuerte con el yield.
- **Arrays (360,41,41) = 12 meses × 30 canales** (13 bandas Sentinel-2 + 3 QA + 14 de clima).
- **NDVI estacional muy marcado** (pico meses 6–8) → la **fenología** es señal fuerte.
- ⚠️ **Nubes**: los parches salen nubosos → imprescindible **enmascarar con bandas QA** al agregar.
- ⚠️ **Sin coordenadas** → cae la idea de CV espacial; el ángulo nuevo del portfolio es el **feature engineering de series satélite** (fenología + índices + máscara de nubes).

## Hardware

CPU local — todo tabular tras agregar los parches satélite. Sin GPU.

## Resultado

_Pendiente — en desarrollo._
