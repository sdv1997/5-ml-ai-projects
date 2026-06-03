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

- [ ] **EDA**: distribución del yield, nº de campos, qué son las 360 capas (bandas×tiempo), valores faltantes/nubes, rango temporal, distribución espacial de los campos.
- [ ] **Feature engineering satélite** (la clave en CPU): por campo, agregar el parche 41×41 (media/percentiles enmascarando nubes) por banda y paso temporal; **índices de vegetación** (NDVI, EVI, NDWI); **fenología temporal** (máximos, integral de NDVI, pendientes, fecha de pico). Unir con suelo/clima.
- [ ] **Validación cruzada espacial** (parcela/región-disjunta) para evitar *spatial leakage* — el primo geográfico del site-disjoint del slot 2.
- [ ] Baseline: media → LightGBM regresión (objetivo acorde a RMSE) sobre las features tabulares.
- [ ] Mejoras: selección de features, índices adicionales, blending; comparar con un baseline espacial (vecinos / kriging) si aporta.
- [ ] Submission a Zindi (late) o reporte de RMSE en CV espacial.

## Hardware

CPU local — todo tabular tras agregar los parches satélite. Sin GPU.

## Resultado

_Pendiente — en desarrollo._
