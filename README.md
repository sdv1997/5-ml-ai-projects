# 30 días de IA

30 días seguidos construyendo algo distinto cada día relacionado con ML / IA. Sigo el progreso en X.

## Tabla de progreso

| Día | Tema | Resultado | Notas |
|---|---|---|---|
| [01](day_01/) | [Richter's Predictor](https://www.drivendata.org/competitions/57/nepal-earthquake/) (DrivenData) | **F1 0.7534 público, rank 55 / ~8.800 (top 0.6%)** | Ensemble LightGBM + CatBoost con feature engineering. 5 iteraciones documentadas (4 positivas, 1 negativa). [Leaderboard](https://www.drivendata.org/competitions/57/nepal-earthquake/leaderboard/?page=1). |
| 02 | _por decidir_ | — | — |

_Se actualiza cada día._

## Estructura

Cada día tiene su propia carpeta `day_XX/` con scripts, README explicando qué se hizo, y outputs (plots). Datos crudos no se versionan — cada README indica cómo obtenerlos.

```
.
├── README.md                     # este fichero
├── data/                         # CSVs descargados, gitignored
└── day_01/
    ├── README.md                 # historia completa del día 1
    ├── day01.py                  # baseline + EDA + plot
    ├── pipeline.py               # modelo ganador (F1 0.7534)
    └── class_distribution.png    # plot de distribución del target
```

## Stack

Lo que voy usando según el día: Python 3.12, pandas, NumPy, scikit-learn, LightGBM, CatBoost, PyTorch (cuando toque NN pequeños), matplotlib. Sin frameworks pesados.

## Sígueme en X

Postero el progreso cada día. _(link al perfil)_
