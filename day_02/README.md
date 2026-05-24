# Día 2 — Conser-vision

**Competición:** [DrivenData – Competition Image Classification: Wildlife Conservation](https://www.drivendata.org/competitions/87/)

**Tarea:** Clasificar imágenes de cámaras trampa en 8 categorías de fauna salvaje.

**Métrica:** Log-loss (↓ mejor).

**Hardware:** RunPod community cloud — RTX A5000 (24 GB VRAM).

---

## Clases

`antelope_duiker` · `bird` · `blank` · `civet_genet` · `hog` · `leopard` · `monkey_prosimian` · `rodent`

---

## Pipeline

| Componente | Detalle |
|---|---|
| Detección | MegaDetectorV5 (YOLOv5, 140M params) → bbox del animal → crop |
| Modelo base | ConvNeXt V2 Base (ImageNet-22K pretrained, `timm`) |
| Fine-tune | Fase 1: cabeza congelada 3 épocas (batch=256, lr=1e-3) → Fase 2: diferencial 12 épocas (batch=64, lr_head=1e-3, lr_backbone=1e-5) |
| Imagen | 224×224 px |
| Augmentación | RandomResizedCrop · HorizontalFlip · ColorJitter · RandomGrayscale · RandomErasing |
| Optimizador | AdamW (wd=1e-4) |
| Scheduler | CosineAnnealingLR (eta_min=1e-7) |
| Loss | CrossEntropyLoss + label smoothing 0.1 |
| Precisión | Mixed precision (torch.cuda.amp) |
| CV | 5-fold StratifiedGroupKFold por **site** (site-disjoint) seed=42 |
| Ensemble | Media aritmética de probabilidades OOF por fold |

---

## Resultados

| # | Descripción | CV log-loss | LB log-loss | Δ LB |
|---|---|---|---|---|
| 1 | ConvNeXt V2 baseline (sin CV site-disjoint, sin MegaDetector) | — | 1.2758 | — |
| 2 | ConvNeXt V2 + CV site-disjoint + MegaDetector crops | 0.9479 | **0.8990** | **-0.377** |

**Rank actual: #18**

### CV por fold (experimento 2)

| Fold | Log-loss |
|---|---|
| 1 | 0.9334 |
| 2 | 1.0894 |
| 3 | 0.8210 |
| 4 | 0.7968 |
| 5 | 1.1351 |
| **Media** | **0.9479** |

La varianza entre folds es alta (0.80 – 1.13), lo que indica que el reto principal es la **generalización cross-site**, no la capacidad del modelo. Algunos grupos de sites son mucho más difíciles que otros.

Gap CV → LB: CV 0.9479 → LB 0.8990 (LB mejor que CV → los sites de test son relativamente más fáciles que los de validación).

---

## Lecciones

- **MegaDetector es la mejora más grande posible** en camera trap classification: elimina el fondo site-específico y fuerza al clasificador a fijarse en el animal. Salto de 1.28 → 0.90 en LB con un solo cambio.
- **CV site-disjoint es obligatorio** en problemas con sites disjuntos train/test. CV con sites mezclados da estimaciones optimistas y peores modelos (el modelo aprende atajos del entorno).
- La varianza entre folds refleja dificultad real de generalización, no ruido. Folds con sites "fáciles" bajan de 0.82, folds con sites difíciles superan 1.10.
- MegaDetectorV5 corre en ~48 min sobre 20k imágenes (A5000), se cachea en JSON, solo corre una vez.
- 69% de imágenes tienen animal detectado. El 8% de `blank` con detección son falsos positivos — coherente.

---

## Reproducibilidad

```bash
# Datos en data/ (gitignored, descargar de DrivenData)
# 1. Detección (solo primera vez, ~48 min en A5000)
python day_02/day02_v2.py --detect-only

# 2. Entrenamiento + CV + submission (~2h en A5000)
tmux new-session -d -s training \
  'python day_02/day02_v2.py --train-only 2>&1 | tee day_02/training_v2.log'
tmux attach -t training
```
