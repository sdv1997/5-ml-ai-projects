# Proyecto 4 — DengAI

**Competición:** [DrivenData – DengAI: Predicting Disease Spread](https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/)

**Tarea:** Predecir cuántos casos de dengue habrá cada semana en dos ciudades —San Juan (Puerto Rico) e Iquitos (Perú)— usando datos del clima (temperatura, lluvia, humedad, vegetación).

**Métrica:** MAE (↓ mejor).

**Hardware:** CPU local.

---

## Pipeline

| Componente | Detalle |
|---|---|
| Modelo | LightGBM, **uno por ciudad** (San Juan e Iquitos son muy distintos) |
| Objetivo | MAE (el modelo aprende a optimizar directamente la métrica) |
| Features | Pocas variables de clima (humedad, temperatura, lluvia) con su **desfase** correcto |
| Idea clave | El clima de hace **1–2 meses** es el que predice los casos de hoy |
| Validación | Por tiempo: se entrena con el pasado y se valida con el futuro (nunca al azar) |
| Salida | Casos por semana, redondeados a entero |

Se modela cada ciudad por separado porque tienen escalas, estacionalidad e histórico diferentes.

---

## Resultados

| # | Descripción | MAE LB público |
|---|---|---|
| 1 | LightGBM con muchas features (~90) | 24.29 |
| 2 | **LightGBM con pocas features + desfase del clima** | **23.67** ← mejor |

**MAE 23.67 · rank 809 / 16.396 (top 4.9%)** — por debajo del benchmark oficial de la competición (~25).

---

## Lecciones

- **Menos features, mejor resultado.** Pasar de ~90 variables a solo unas pocas bien elegidas bajó el error real (24.29 → 23.67). Con muchas variables el modelo se sobreajustaba.
- **Cada ciudad por separado.** San Juan responde bien al clima; Iquitos es mucho más ruidosa (casi un 20% de semanas con 0 casos). Mezclarlas empeora ambas.
- **El clima va con retraso.** La humedad y la temperatura de hace ~1–2 meses son las que mejor anticipan los casos. Acertar ese desfase fue clave en San Juan.
- **Validar con el futuro, no al azar.** Como es una serie temporal, la validación tiene que respetar el orden del tiempo; barajar las semanas da estimaciones engañosas.
- Probamos también modelos clásicos de series temporales y combinaciones de varios modelos, pero ninguno mejoró al LightGBM simple, así que el modelo final es el más sencillo.

---

## Reproducibilidad

```bash
# Datos en 04_dengai/data/ (gitignored, descargar de DrivenData)
python 04_dengai/eda.py        # exploración → eda.png
python 04_dengai/pipeline.py   # modelo final → submission.csv
```
