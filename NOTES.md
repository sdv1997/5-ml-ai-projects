# Notas técnicas del portfolio

Notas de diseño y lecciones que se repiten entre los 5 proyectos. El detalle de cada uno está en su `README.md`.

## Hardware

- Proyecto 1: CPU local (problema tabular).
- Proyectos 2–3: GPU en la nube (RTX A5000, 24 GB).
- Proyectos 4–5: CPU local.

## Stack

Python 3.12, pandas, NumPy, scikit-learn, LightGBM, CatBoost, PyTorch (cuando hay GPU), matplotlib. Sin frameworks pesados.

## Convenciones del repo

- `NN_nombre/` por proyecto: `README.md` + scripts mínimos (`eda.py` / `pipeline.py`) + algún plot.
- `data/` y las submissions van en `.gitignore` (se regeneran corriendo el pipeline).
- Los experimentos que no funcionan también se documentan: forman parte de la historia de cada proyecto.

## Lecciones transferibles

**Proyecto 1 — Richter (tabular)**
- Con identificadores de zona de alta cardinalidad, tratarlos como categoría (no como número) antes de tunear nada: la mayor ganancia individual.
- Un ensemble de dos modelos parecidos no mejora al promedio simple; hace falta más diversidad.
- Fijar los folds de validación y comparar todos los experimentos contra los mismos.

**Proyecto 2 — Conser-vision (visión)**
- En cámaras trampa, recortar al animal con un detector es la mayor mejora: quita el fondo del sitio y el modelo deja de aprender el entorno en vez del animal.
- Si train y test tienen sitios distintos, la validación tiene que separar por sitio; mezclarlos engaña.

**Proyecto 3 — What's Up Docs (NLP / LLM)**
- En resumen automático, el contexto importa más que el tamaño del modelo: ampliar el texto de entrada mejoró mucho más que pasar de 3B a 7B parámetros.
- Cuantización 4-bit: el modelo ocupa ~4 GB en vez de ~15 con pérdida mínima en tareas generativas.

**Proyecto 4 — DengAI (series temporales)**
- Validar respetando el tiempo (entrenar con el pasado, validar con el futuro); barajar las semanas da estimaciones falsas.
- Menos variables generaliza mejor: pasar de ~90 a unas pocas bien elegidas bajó el error real.
- El clima va con retraso: el de hace ~1–2 meses es el que predice los casos.
- Probamos modelos clásicos de series temporales y combinaciones de modelos; ninguno mejoró al LightGBM simple.

**Proyecto 5 — Crop Yield (satélite)**
- Satélite no implica GPU: resumir los parches de imagen en una tabla (índices de vegetación + máscara de nubes) y modelar con boosting corre en CPU y es competitivo.
- Una validación mejor no garantiza un resultado mejor: copiar los trucos de una solución top empeoró, porque su mejor "truco" (descartar el 41% de los datos) dependía de su pipeline concreto.
