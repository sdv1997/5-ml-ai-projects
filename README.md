# 30 días de IA

30 días seguidos construyendo algo distinto cada día relacionado con ML / IA. Sigo el progreso en X.

## Hardware

Día 1 corrió en CPU. A partir del Día 2, GPU en RunPod community cloud (RTX A5000). Lo indico en el README de cada día.

## Tabla de progreso

| Día | Tema | Resultado | Notas |
|---|---|---|---|
| [01](day_01/) | [Richter's Predictor](https://www.drivendata.org/competitions/57/nepal-earthquake/) (DrivenData) | **F1 0.7534 público, rank 55 / 8.254 (top 0.6%)** | ML Tabular. Ensemble LightGBM + CatBoost con feature engineering. CPU. |
| [02](day_02/) | [Conser-vision](https://www.drivendata.org/competitions/87/competition-image-classification-wildlife-conservation/) (DrivenData) | **Log-loss 0.8990 público, rank 18 / 2.064 (top 0.9%)** | DL Imagen. MegaDetectorV5 → crop → ConvNeXt V2 Base. CV site-disjoint. RTX A5000. |

_Se actualiza cada día._

## Estructura

Cada día tiene su propia carpeta `day_XX/` con scripts, README explicando qué se hizo, y outputs (plots). Datos crudos no se versionan — cada README indica cómo obtenerlos.

```
.
├── README.md
├── data/                         # imágenes y CSVs, gitignored
├── day_01/                       # Richter's Predictor
│   ├── README.md
│   ├── day01.py
│   └── pipeline.py
└── day_02/                       # Conser-vision
    ├── README.md
    ├── day02_v2.py               # pipeline ganador
    └── cv_summary_v2.png
```

## Stack

Lo que voy usando según el día: Python 3.12, pandas, NumPy, scikit-learn, LightGBM, CatBoost, PyTorch (cuando toque NN pequeños), matplotlib. Sin frameworks pesados.

## Sígueme en X

Postero el progreso cada día. _(link al perfil)_
