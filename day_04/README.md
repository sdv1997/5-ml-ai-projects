# Día 4 — BirdCLEF+ 2026

**Competición:** [Kaggle – BirdCLEF+ 2026](https://www.kaggle.com/competitions/birdclef-2026)

**Tarea:** Identificar 234 especies (aves, ranas, insectos, mamíferos) en ventanas de 5 segundos de soundscapes (grabaciones de campo continuas).

**Métrica:** ROC-AUC macro (↑ mejor).

**Hardware:** RunPod community cloud — RTX A5000 (24 GB VRAM).

> ⚠️ **Estado: EN PROGRESO.** Pipeline base funcionando y submission-ready, pero a mitad de iteración. Ver [Estado actual y plan del próximo día](#estado-actual-y-plan-del-próximo-día) abajo.

---

## El reto de fondo

El dato de entrenamiento son **clips focales** (un animal en primer plano, limpio) descargados de Xeno-canto. Pero el test son **soundscapes**: grabaciones de campo con ruido ambiente, varias especies solapadas, llamadas lejanas. Esa **brecha de dominio focal→soundscape** es el problema central de BirdCLEF y lo que separa un 0.78 de un 0.94.

Además hay muy pocas etiquetas reales de soundscape (solo 66 ficheros gold), así que hay que generar **pseudo-labels** con un teacher entrenado en clips.

---

## Pipeline (`day04.py`)

Two-stage con knowledge distillation, exportado a ONNX para inferencia en CPU (Kaggle limita a 90 min sin GPU en este reto):

| Fase | Qué hace |
|---|---|
| **Phase 1 — teacher** | EfficientNet-B0 sobre 35.549 clips focales individuales. Mel-spectrogram → 3 canales → 224×224. 25 epochs, mixup + SpecAugment. |
| **Pseudo-label** | El teacher etiqueta 10.592 soundscapes sin label (12 ventanas de 5s c/u). Se quedan las ventanas con `max_prob ≥ 0.25`. |
| **Phase 2 — student** | Fine-tune sobre gold (1.478 windows) + pseudo (63.103 windows), 10 epochs, peso pseudo=0.5. |
| **Export** | ONNX (opset 17, batch dinámico) para CPU. |

### Config clave (`Cfg`)
- Audio: 32 kHz, ventanas de 5s (160k samples).
- Mel: 128 bins, n_fft=1024, hop=320, fmin=50, fmax=14000.
- Labels: primary=1.0, secondary=0.5 (multi-label, BCE). Clips con rating<3 pesan 0.5.
- CV: StratifiedKFold(5, seed=42) por `primary_label`. Solo se ha corrido fold 0.
- **Resume robusto:** `resume_fold0.pt` guarda phase+epoch+optimizer+scheduler cada epoch, así que la corrida se puede reanudar tras un crash o un apagado del pod.

---

## Resultados (fold 0)

| Fase | AUC | Sobre |
|---|---|---|
| Phase 1 (teacher) | **0.9804** | clips limpios (28.439 train / 7.110 val) |
| Phase 2 (student, final) | **0.7757** | soundscapes (64.289 windows train) |

Aún sin submission al leaderboard. **El top general del LB ronda 0.94.**

### Diagnóstico
- El teacher clava clips focales limpios (0.98) pero al evaluar en soundscapes ruidosos cae a 0.78 → **la brecha de dominio es el cuello de botella**, no la capacidad de aprender clips.
- Phase 2 hace **plateau en epoch 5** (0.7757) y empieza a sobreajustar (ep08 ya bajaba a 0.7641). Más epochs del mismo setup **no** suben.
- El val set de Phase 2 es pequeño y ruidoso (~300 windows de un puñado de ficheros gold), así que el 0.7757 es un proxy con varianza alta.

### Bug resuelto en esta sesión
Pseudo-labeling crasheaba con `RuntimeError: stft input and window must be on the same device`: el módulo `MelSpec` estaba en CUDA pero las ventanas de audio se le pasaban en CPU. Fix en `generate_pseudo_labels` (línea ~312): mover cada ventana a `cfg.device` **antes** de pasarla al mel (`mel_fn(w.to(cfg.device))`), no después.

---

## Estado actual y plan del próximo día

**Decisión tomada:** para cerrar la brecha 0.78 → ~0.94, combinar las dos palancas de mayor impacto (opción "1+2"). Estimado ~2h de pod.

### 1. Background-noise augmentation (la palanca clásica focal→soundscape)
Durante Phase 1, mezclar ruido/ambiente real de soundscape en los clips focales a SNR aleatorio, para que el teacher aprenda a reconocer especies *con* ruido de fondo. Plan de implementación:
- Construir un pool en memoria de ~400 ventanas de 5s sacadas al azar de `train_soundscapes`.
- En `ClipDataset.__getitem__` (solo cuando `augment=True`): con prob ~0.5, sumar una muestra del pool escalada a SNR ∈ [3, 20] dB. Opcional: ruido gaussiano leve con prob ~0.3.
- Solo en Phase 1 (los soundscapes de Phase 2 ya traen ruido real).

### 2. Backbone más fuerte
Cambiar `cfg.backbone`: `efficientnet_b0` (~5M params) → **`eca_nfnet_l0`** (~24M, normalizer-free + ECA channel attention). Es el backbone canónico de las soluciones top de BirdCLEF. Alternativa más ligera si hay problemas de VRAM/tiempo: `tf_efficientnet_b3_ns`. Vigilar VRAM a bs=64; bajar a 48 si hace OOM.

### Limpieza necesaria antes de relanzar
Los artefactos del fold 0 son de la arquitectura B0 y **no** sirven para el run nuevo:
- `model_fold0.pt`, `model_fold0_sc.pt`, `resume_fold0.pt`, `model_fold0.onnx` → resume incompatible con arquitectura nueva. Respaldar como `.b0` o borrar.
- `pseudo_fold0.npz` → **borrar obligatoriamente**: son pseudo-labels del teacher B0; el teacher nuevo (con noise-aug) debe regenerarlos.

### Ideas adicionales si 1+2 no bastan
- Pseudo-labeling iterativo (re-etiquetar con el student y reentrenar).
- Umbral pseudo más duro (0.5) + sobre-peso al gold en Phase 2.
- TTA en inferencia; ensemble de folds.
- Subir resolución mel / nº de bins.

---

## Reproducibilidad

```bash
# Desde el repo. Datos en data/birdclef-2026/ (gitignored, descargar de Kaggle).

# Corre fold 0 completo (Phase 1 → pseudo-label → Phase 2 → ONNX).
# Reanuda solo si existe resume_fold0.pt.
python day_04/day04.py --fold 0

# Solo evaluar el teacher ya entrenado sobre el val de clips:
python day_04/day04.py --fold 0 --eval-only
```

Pesos (`*.pt`), ONNX, pseudo-cache (`*.npz`) y logs están gitignored — se regeneran corriendo el script. Lanzar entrenamientos largos en `tmux`, no como background de la sesión (muere si se cae el SSH).

---

## Lecciones (preliminares)

- **En audio focal→soundscape, la brecha de dominio domina todo.** Un teacher con 0.98 en clips limpios se desploma a 0.78 en soundscapes ruidosos. La calidad del clasificador de clips no es el límite; la transferencia al dominio ruidoso sí.
- **Mover tensores al device en el sitio correcto importa con módulos `nn.Module` de torchaudio.** El `MelSpec` en CUDA necesita input en CUDA; el crash de STFT delata el mismatch.
- **Checkpoint resumible (phase+epoch+opt+sched) es clave en pods que se pagan por hora**: permite apagar y reanudar sin perder progreso ni rehacer pseudo-labeling (se cachea en `.npz`).
