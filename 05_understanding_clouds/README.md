# Proyecto 5 — Understanding Clouds from Satellite Images

> **Competición:** [Understanding Clouds from Satellite Images](https://www.kaggle.com/competitions/understanding_cloud_organization) (Kaggle)
> **Tarea:** Segmentar 4 patrones de organización de nubes — **Fish, Flower, Gravel, Sugar** — en imágenes satélite (MODIS / NASA Worldview).
> **Métrica:** mean Dice coefficient (↑ mejor).
> **Submission:** CSV con máscaras en **RLE** (`Image_Label, EncodedPixels`). No es code competition.
> **Estado:** cerrada (2019, 1.538 equipos). Admite **late submission** → puntúa contra el private leaderboard, así que reportamos "top X%" en vez de rank oficial en vivo.

## Por qué este proyecto

Primer problema de **segmentación densa** (predicción píxel a píxel) y primera modalidad **geoespacial / satélite** del portfolio. Añade un tipo de tarea que clasificación y generación no cubren.

## Plan

- [x] **EDA** ([eda.py](eda.py) → [eda.png](eda.png)) + **utilidades RLE/Dice** ([rle.py](rle.py), con test de round-trip).
- [ ] Baseline U-Net ([pipeline.py](pipeline.py)): encoder preentrenado (`segmentation_models.pytorch`), 4 canales sigmoid (multi-label), resize agresivo, loss BCE+Dice. **En RunPod (A5000).**
- [ ] Post-proceso típico de esta competición: umbral por clase + **min-size** (descartar máscaras pequeñas) — fue clave en el leaderboard.
- [ ] Two-stage: clasificador "¿está la clase?" para quitar falsos positivos.
- [ ] TTA (flips) + ensemble de encoders. Late submission → Dice private LB.

## EDA — hallazgos

- **5.546 imágenes** de train, 3.698 de test. Multi-label: **media 2.13 clases/imagen**, todas tienen ≥1 (nunca vacía del todo).
- Frecuencia por clase: **Sugar 67.6% · Gravel 53.0% · Fish 50.1% · Flower 42.6%** (equilibrado, sin clase rara).
- Imágenes **2100×1400**. **Las máscaras son bloques rectangulares gruesos** (los anotadores marcaron cajas, no contornos) → se puede **reescalar muy agresivo** (p.ej. 384×576) sin perder señal; el reto no es el detalle fino sino *localizar* el patrón y *decidir si está*.
- Implicación de diseño: importa más el **post-proceso** (umbral + min-size, decidir presencia de clase) que la precisión de contorno.

## Hardware

RunPod community cloud — RTX A5000 (igual que slots 2 y 3). El EDA y las utilidades RLE corren en local; el entrenamiento va en el pod.

## Resultado

_Pendiente — modelo en desarrollo._
