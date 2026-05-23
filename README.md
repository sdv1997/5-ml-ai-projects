# 30 días de ML/IA — Richter's Predictor

30 días intentando subir mi score en una competición de ML que importa: predecir el daño que sufrió cada edificio de Nepal tras el terremoto de Gorkha en 2015.

> **Competición:** [Richter's Predictor — Modeling Earthquake Damage](https://www.drivendata.org/competitions/57/nepal-earthquake/) (DrivenData)
> **Métrica:** F1 micro-averaged sobre 3 clases de daño (1: bajo / 2: medio / 3: destrucción casi total).
> **Dataset:** 260k+ edificios encuestados por Kathmandu Living Labs + Nepal Central Bureau of Statistics. Uno de los mayores datasets post-desastre jamás publicados.

## Por qué este reto

- Es **data for good real**: el output del modelo importa — priorizar inspecciones, mejorar códigos de construcción, dirigir ayuda humanitaria.
- Es **tabular** → todo corre en CPU. No necesito alquilar GPUs para hacer ML serio.
- Tiene **leaderboard público** → progreso visible y honesto cada día.
- Tiene **dataset enorme y rico** (38 features mezclando geografía, materiales, edad, uso) → 30 días de feature engineering posibles sin agotar el tema.

## Estructura del repo

```
.
├── data/                      # CSVs descargados de DrivenData (no incluidos en el repo)
├── day_01/                    # EDA + 2 baselines
│   ├── README.md
│   ├── day01.py
│   ├── class_distribution.png
│   └── submissions/
└── README.md                  # este fichero
```

## Tabla de progreso

| Día | Tema | F1 micro (local CV) | F1 micro (público) | Notas |
|---|---|---|---|---|
| [01](day_01/) | EDA + 2 baselines | 0.7182 | **0.7179** | LightGBM out-of-the-box. CV ↔ público alineados (Δ -0.0003). |
| [01 it.2](day_01/) | geos como category + tuning | 0.7497 | **0.7488** | Fix gordo: `geo_level_*_id` como `category` salta +0.0306. Rank 491. |
| [01 it.3](day_01/) | Ensemble LightGBM + CatBoost | 0.7518 | **0.7510** | +0.0022 sobre E4. Rank 280 / ~8.800 (top 3.2%, +211 puestos). |
| [01 it.4](day_01/) | Target encoding OOF (negativo) | 0.7470 | — | -0.0028 vs E4. LightGBM categórico nativo ya hace lo equivalente internamente. |
| [01 it.5](day_01/) | Feature engineering (agregaciones) | 0.7510 | _por subir_ | +0.0012 sobre E4 puro. σ del CV baja de 0.0016 a 0.0010. |

_Se actualiza cada día._

## Reproducir

1. Crear cuenta en [DrivenData](https://www.drivendata.org/accounts/signup/) y descargar los 4 CSVs de la competición a `data/`.
2. Instalar dependencias: `pip install pandas numpy scikit-learn lightgbm matplotlib`.
3. Ejecutar el día que toque: `python day_XX\dayXX.py`.

## Sígueme en Twitter

Postero el progreso cada día. _(link al perfil)_
