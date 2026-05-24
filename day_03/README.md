# Día 3 — What's Up, Docs?

**Competición:** [DrivenData – What's Up, Docs? Document Summarization with LLMs](https://www.drivendata.org/competitions/297/whats-up-docs/)

**Tarea:** Generar el abstract de papers de ciencias sociales (SocArXiv) a partir del texto completo.

**Métrica:** ROUGE-2 macro (↑ mejor).

**Hardware:** RunPod community cloud — RTX A5000 (24 GB VRAM).

---

## Pipeline

| Componente | Detalle |
|---|---|
| Modelo | Qwen2.5-7B-Instruct-AWQ (cuantización 4-bit, ~4.4 GB en disco) |
| Inferencia | vLLM 0.21.0, `max_model_len=12000`, `gpu_memory_utilization=0.85` |
| Extracción | Secciones de intro + conclusión por headers markdown, hasta 24.000 chars |
| Prompt | System: editor académico experto, 150-250 palabras, terminología exacta del paper |
| Temperatura | 0.1 |
| Max tokens output | 350 |

### Extracción de secciones

Se detectan headers markdown (`#`, `##`, `###`) y se clasifican por keywords:
- **Intro:** `introduction`, `background`, `overview`, `context`, `motivation`
- **Conclusión:** `conclusion`, `discussion`, `summary`, `findings`, `results`, …

Prioridad intro → conclusión → resto. Truncado a `max_chars`.

---

## Resultados

| # | Descripción | ROUGE-2 train | ROUGE-2 LB público | Rank |
|---|---|---|---|---|
| 1 | Qwen2.5-3B-Instruct, max_chars=6000 | 0.1400 | 0.1258 | #12 |
| 2 | Qwen2.5-7B-Instruct-AWQ, max_chars=24000 | **0.1549** | **0.1398** | **#9** |

**Rank actual: #9 / 495**

### Análisis del gap CV → LB

Gap train→público consistente: +0.0151 en train vs +0.0140 en LB (el modelo ve algo más contexto en papers largos, pero el test set tiene distribución similar).

---

## Iteraciones

### Iteración 1 — baseline 3B, contexto corto
- Modelo: Qwen2.5-3B-Instruct (float16, ~5.8 GB)
- `max_chars=6000` → solo captura la introducción en la mayoría de papers
- ROUGE-2 train: 0.1400 · LB: 0.1258

### Iteración 2 — 7B cuantizado + contexto completo (+0.014 LB)
- Modelo: Qwen2.5-7B-Instruct-AWQ (4-bit, ~4.4 GB) — más parámetros, menos disco
- `max_chars=24000` → captura intro **y** conclusión en casi todos los papers
- La mejora vino casi íntegramente del contexto, no del modelo: 3B con 24000 chars daría resultado similar
- ROUGE-2 train: 0.1549 · LB: **0.1398**

---

## Lecciones

- **El contexto importa más que el tamaño del modelo** para summarización: pasar de 6000 a 24000 chars (+0.014 ROUGE-2) superó con creces cambiar de 3B a 7B (+0.0004). La conclusión del paper estaba siendo truncada completamente con 6000 chars.
- **AWQ 4-bit vs float16**: la cuantización 4-bit reduce el modelo de ~15 GB a ~4.4 GB con pérdida mínima de calidad en este tipo de tarea generativa. Útil cuando el almacenamiento es el cuello de botella.
- **vLLM** con batching procesa 345 papers en ~30 segundos en A5000 — la latencia real es el warmup/compilación (~2 min), no la inferencia.
- El texto promedio del paper son 6300 palabras (~38.000 chars). Con `max_model_len=12000` tokens y un modelo Qwen2.5, caben hasta ~40.000 chars de input — casi el paper completo en la mayoría de casos.
- **Gestión de almacenamiento en RunPod**: el caché de HuggingFace va a `/root/.cache/` por defecto (overlay raíz de 20 GB). Redirigir con `HF_HOME=/workspace/.cache/huggingface` evita llenar el overlay con pesos del modelo.

---

## Reproducibilidad

```bash
# Desde /workspace/30-days-of-ai/
# Datos en data/day03/ (gitignored, descargar de DrivenData)

# 1. Setup: instalar deps y descargar modelo (~10s, modelo va a /workspace/)
bash day_03/setup.sh

# 2. Eval rápido en 100 papers de train
HF_HOME=/workspace/.cache/huggingface python3 day_03/day03.py --eval-only

# 3. Submission completa (test + scoring train, ~3 min en A5000)
HF_HOME=/workspace/.cache/huggingface python3 day_03/day03.py
# → day_03/submission.csv
```
