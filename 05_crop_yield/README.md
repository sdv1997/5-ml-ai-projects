# Proyecto 5 — CGIAR Crop Yield

**Competición:** [Zindi – CGIAR Crop Yield Prediction Challenge](https://zindi.africa/competitions/cgiar-crop-yield-prediction-challenge)

**Tarea:** Predecir el **rendimiento** de cada parcela agrícola a partir de imágenes de satélite (Sentinel-2) y datos de suelo y clima. Regresión.

**Métrica:** RMSE (↓ mejor).

**Hardware:** CPU local (sin GPU).

---

## Pipeline

| Componente | Detalle |
|---|---|
| Idea clave | En vez de meter las imágenes en una red neuronal, se **resumen en una tabla de números** y se modela con boosting (funciona bien y corre en CPU) |
| Satélite | Por cada parcela y mes: media del parche de imagen, **quitando las nubes** |
| Vegetación | Índices de verdor de la planta (NDVI, EVI) y su evolución a lo largo del año |
| Extra | Datos de suelo y clima por parcela |
| Modelo | LightGBM (objetivo RMSE) |
| Validación | Por año (se valida prediciendo un año que el modelo no ha visto) |

---

## Resultados

| | RMSE |
|---|---|
| Validación (predecir un año nuevo) | 1.61 |
| Predecir siempre la media (referencia) | 1.74 |
| **Submission (tardía)** | **1.75** |

El modelo mejora a la simple media en validación. Predecir rendimiento desde satélite es **intrínsecamente ruidoso**, así que la mejora es modesta.

> La competición **cerró en febrero de 2021**, así que la submission es tardía y no hay ranking en vivo.

---

## Lecciones

- **Una validación mejor no garantiza un resultado mejor.** Probamos copiar varios trucos de una de las mejores soluciones publicadas; mejoraban la validación pero empeoraron el resultado final.
- **El culpable fue tirar datos.** Ese enfoque se quedaba solo con las etiquetas "limpias" y descartaba el **41% de las parcelas**. En un dataset pequeño, perder tanta cobertura hace más daño que el ruido que intentas quitar.
- **Quitar las nubes es imprescindible.** Los parches de satélite salen muy nubosos; sin filtrarlas, las medias quedan contaminadas.
- **La fenología es señal fuerte.** El verdor de la planta tiene un pico estacional claro (meses 6–8); cómo evoluciona a lo largo del año dice mucho del rendimiento.

---

## Reproducibilidad

```bash
# Datos en 05_crop_yield/data/ (gitignored, descargar del mirror de Kaggle)
python 05_crop_yield/eda.py        # exploración → eda.png
python 05_crop_yield/pipeline.py   # modelo final → submission.csv
```
