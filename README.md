# 30 días de IA

30 días seguidos construyendo algo distinto cada día relacionado con ML / IA. Sin GPU, todo en CPU. Sigo el progreso en Twitter.

> **Restricciones autoimpuestas:**
> - ~1 hora de trabajo por día.
> - CPU only — sin GPU, sin nube, sin alquilar máquinas.
> - Cada día queda público en este repo + un tweet con el resultado.

## Tabla de progreso

| Día | Tema | Resultado | Notas |
|---|---|---|---|
| [01](day_01/) | Richter's Predictor (DrivenData) | **F1 0.7510, rank 280 / ~8.800 (top 3.2%)** | EDA + baselines + ensemble LightGBM/CatBoost + feature engineering. Iteraciones documentadas, incluido un experimento negativo. |
| 02 | _por decidir_ | — | — |

_Se actualiza cada día._

## Estructura

Cada día tiene su propia carpeta `day_XX/` con scripts, README explicando qué se hizo, y outputs (plots, logs). Datos crudos no se versionan — cada README indica cómo obtenerlos.

## Stack

Lo que voy usando según el día: Python 3.12, pandas, NumPy, scikit-learn, LightGBM, CatBoost, PyTorch (cuando toque NN pequeños), matplotlib. Sin frameworks pesados.

## Sígueme en Twitter

Postero el progreso cada día. _(link al perfil)_
