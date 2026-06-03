# Proyecto 5 — CGIAR Crop Yield Prediction

> **Competición:** [CGIAR Crop Yield Prediction Challenge](https://zindi.africa/competitions/cgiar-crop-yield-prediction-challenge) (Zindi)
> **Tarea:** Predecir el **rendimiento** (yield) de cada parcela agrícola a partir de imágenes satélite **Sentinel-2** y datos de suelo/clima. **Regresión.**
> **Métrica:** RMSE (↓ mejor).
> **Submission:** CSV (`Field_ID, Yield`).
> **Resultado:** **RMSE 1.75** (late submission). **Competición cerrada (feb 2021) — leaderboard cerrado**, sin rank oficial en vivo.

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
- [x] Iteración 2 (ideas del 4º puesto): mediana del parche, SAVI/ratios, clima de temporada, filtro de calidad, KFold aleatorio, log-target. **No batió a v1 en el LB** (ver Resultado).
- [x] Submission a Zindi (late) → **RMSE 1.75** (v1). LB cerrado.

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

### Submission 1 — gap CV→LB

| | CV | LB private |
|---|---|---|
| v1 (media+QA, todas las calidades, GroupKFold) | 1.61 | **1.75** |

El LB (1.75) ≈ baseline de la media → el modelo apenas generaliza al test. Mismo patrón que DengAI: la CV optimista.

### Iteración 2 — ideas de la solución 4ª clasificada (CV ~1.59)

Estudiando un notebook del **4º puesto**, adoptamos sus trucos:
- **Filtro de calidad de etiqueta**: entrenar solo con `Quality ∈ {1,3}` (2977 → 1746 campos) — etiquetas más limpias generalizan mejor al test.
- **MEDIANA del parche** por mes (robusta a nubes) en vez de media + máscara QA.
- **SAVI** + **ratios red-edge** (B7/B5, B7/B6) además de NDVI/EVI.
- **Clima de temporada de maíz** (meses 3–9: pr/tmmn/tmmx, media 4 años) + suelo ISRIC.
- **KFold aleatorio 5-fold** (imita el LB; el test mezcla años) y **sin log-target** (gana a log: 1.583 vs 1.614).

| config | KFold aleatorio (≈LB) | LB real |
|---|---|---|
| v3 (estilo 4º: filtro calidad + mediana) | 1.583 | **1.86 (PEOR)** |

**Negativo documentado:** los trucos del 4º puesto **no transfieren a nuestro LB**. El culpable principal: el **filtro de calidad** — quedarse solo con `Quality ∈ {1,3}` tira el **41% de los datos** (2977 → 1746), y en un dataset pequeño esa pérdida de cobertura hace más daño que el ruido de etiqueta. Quitar el filtro (manteniendo las features de mediana) recupera la CV (1.538) pero ya no lo subimos.

### Conclusión

Nos quedamos con la **v1** (media+QA sobre el parche, todas las calidades, GroupKFold): **RMSE 1.75** en late submission, el mejor de las tres versiones probadas. La competición está **cerrada (feb 2021)**, así que no hay rank oficial en vivo.

Lección (repetida de DengAI): **una CV mejor no garantiza mejor LB**. Aquí incluso copiar a una solución top empeoró — porque sus decisiones (filtro de calidad) dependían de su pipeline concreto, no son universales. Valor del proyecto: un flujo **satélite→tabular en CPU** honesto (NDVI/EVI/fenología, máscara de nubes, GroupKFold) que cubre el hueco geoespacial+satélite del portfolio.

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
