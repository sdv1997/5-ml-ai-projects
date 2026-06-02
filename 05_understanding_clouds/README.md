# Proyecto 5 — Understanding Clouds from Satellite Images

> **Competición:** [Understanding Clouds from Satellite Images](https://www.kaggle.com/competitions/understanding_cloud_organization) (Kaggle)
> **Tarea:** Segmentar 4 patrones de organización de nubes — **Fish, Flower, Gravel, Sugar** — en imágenes satélite (MODIS / NASA Worldview).
> **Métrica:** mean Dice coefficient (↑ mejor).
> **Submission:** CSV con máscaras en **RLE** (`Image_Label, EncodedPixels`). No es code competition.
> **Estado:** cerrada (2019, 1.538 equipos). Admite **late submission** → puntúa contra el private leaderboard, así que reportamos "top X%" en vez de rank oficial en vivo.

## Por qué este proyecto

Primer problema de **segmentación densa** (predicción píxel a píxel) y primera modalidad **geoespacial / satélite** del portfolio. Añade un tipo de tarea que clasificación y generación no cubren.

## Plan

- [ ] EDA: tamaños de imagen, frecuencia de cada clase, solapamiento entre patrones (una imagen puede tener varios).
- [ ] Utilidades de **RLE**: decodificar máscaras de train, codificar predicciones para la submission.
- [ ] Baseline: U-Net con encoder preentrenado (`segmentation_models.pytorch`, p.ej. ResNet/EfficientNet) por clase.
- [ ] Loss: combinación **BCE + Dice**. Resize agresivo (las imágenes son grandes; las máscaras son patrones gruesos, no detalle fino).
- [ ] Post-proceso típico de esta competición: umbral por clase + **min-size** (descartar máscaras pequeñas) — fue clave en el leaderboard.
- [ ] TTA (flips) y, si da tiempo, ensemble de encoders.

Corre en RunPod (RTX A5000).

## Resultado

_Pendiente._
