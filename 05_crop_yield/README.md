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
- [ ] **Feature engineering satélite** (la clave en CPU): por campo, agregar el parche 41×41 (media/percentiles **enmascarando nubes con las bandas QA**) por banda y mes; **índices de vegetación** (NDVI, EVI, NDWI); **fenología temporal** (máximo de NDVI, integral, pendientes, mes de pico). Unir con suelo/clima de `fields_w_additional_info.csv`.
- [ ] **Validación GroupKFold por año** (2016–2019) para una estimación honesta (no hay coordenadas → no se puede CV espacial).
- [ ] Baseline: media → LightGBM regresión (acorde a RMSE) sobre las features tabulares.
- [ ] Mejoras: selección de features, índices adicionales, manejo de la cola alta del yield.
- [ ] Submission a Zindi (late) o reporte de RMSE en CV.

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
