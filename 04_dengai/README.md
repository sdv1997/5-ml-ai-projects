# Proyecto 4 — DengAI: Predicting Disease Spread

> **Competición:** [DengAI: Predicting Disease Spread](https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/) (DrivenData)
> **Tarea:** Predecir el número de casos semanales de dengue en San Juan (Puerto Rico) e Iquitos (Perú) a partir de variables climáticas y ambientales (temperatura, precipitación, vegetación NDVI, humedad…).
> **Métrica:** MAE (↓ mejor).
> **Submission:** CSV (`city, year, weekofyear, total_cases`).
> **Estado:** practice abierta — leaderboard vivo, rank real.

## Por qué este proyecto

Primer problema de **regresión + serie temporal** del portfolio. Los slots 1–3 son clasificación (tabular, imagen) y generación (texto); aquí entra la familia de forecasting, la más común en industria y la que conecta con economía.

## Plan

- [x] EDA ([eda.py](eda.py) → [eda.png](eda.png)): dos ciudades con escala/estacionalidad/histórico distintos → **modelamos por separado**. Test = bloque de futuro puro. Target muy asimétrico (skew ~4). Autocorrelación alta pero no usable (holdout futuro largo). Missing en clima (ndvi_ne 13%).
- [x] **Validación temporal** (holdout: primeras 75% semanas → train, últimas 25% → val). CV aleatoria sería leakage.
- [x] Features de **rolling** sobre drivers climáticos (humedad/temp/precip; el clima precede a los casos) + estacionalidad `weekofyear` (sin/cos). **Sin lags del target** (no disponibles en el test futuro).
- [x] Baseline ([pipeline.py](pipeline.py)): seasonal-naive como referencia honesta → LightGBM L1 (objetivo MAE) por ciudad, clip≥0 + redondeo a entero.
- [x] Iteración 2 ([pipeline.py](pipeline.py)): harness que **selecciona config por ciudad** en holdout temporal, barriendo objetivo (L1 / Poisson / Tweedie) × lags de clima × suavizado × blend con seasonal-naive.
- [x] Iteración 3 ([compare_models.py](compare_models.py)): **validación rolling-origin** (4 orígenes) + comparación de 4 familias de modelo por ciudad (seasonal-naive, LightGBM, NegBin GLM, SARIMAX). Genera la submission con el mejor por ciudad.
- [x] Iteración 4 ([model_lagopt.py](model_lagopt.py)): hipótesis = **sobreajuste por exceso de features**. Reduce a 5 drivers, **busca el lag óptimo por variable** (clima precede a casos), suaviza features y predicción, modelo simple. ([make_submission.py](make_submission.py) compone submissions por ciudad para aislar causas en el LB.)
- [x] Iteración 5 ([ensemble.py](ensemble.py)): **ensemble por ciudad** (lgbm-small + NegBin + seasonal-naive) con pesos elegidos por rolling-origin. Ayuda en iq, no en sj.
- [x] Iteración 6 ([autoreg.py](autoreg.py)): **autorregresión recursiva** con lags del propio target. **NEGATIVO** (sj 26.16 → 58.62): el error se acumula al realimentar predicciones en un horizonte de 260 semanas. La autocorrelación es real pero inutilizable en este test futuro.

## Resultado

Validación temporal (MAE, menor = mejor):

| Ciudad | seasonal-naive | baseline (it.1) | config elegida (it.2) | config |
|---|---|---|---|---|
| sj | 24.86 | 16.89 | **16.21** | L1 + lags clima + suavizado(3) |
| iq | 7.71 | 7.62 | **7.54** | L1 + blend 25% seasonal-naive |

- **sj** responde bien a las features de clima; **iq** es ruidosa (~18% semanas a 0) y el modelo apenas supera a la media estacional → un blend ligero con el naive ayuda.
- **Negativo documentado:** los objetivos de conteo (**Poisson / Tweedie) no mejoraron**. La métrica es MAE y entrenar L1 la optimiza directamente; Poisson/Tweedie minimizan devianza, no MAE.
- Lo que movió la aguja: **lags explícitos de clima + suavizado** (sj) y **blend con naive** (iq). Ganancias modestas — leaderboard comprimido y señal clima→casos limitada.

### Submission 1 — el gap validación→leaderboard

| | MAE val (pooled) | MAE LB público | rank |
|---|---|---|---|
| Submission 1 | ~13 | **~24** | ~1007 |

**Gran lección (negativa, documentada):** la validación (un único holdout = últimas 25% semanas) era **demasiado optimista**. El test real casi dobla el error. Casi todo el gap viene de **sj**: el modelo ajustó un periodo calmado (~2003–2008) pero el test (2008–2013) tiene brotes cuya magnitud el clima no anticipa. Seleccionar config (suavizado/blend/lags) sobre ese único corte fue, en parte, ajustar a un split afortunado.

**Implicación para la siguiente iteración:** necesitamos una **validación que track-ee el leaderboard** antes de tocar el modelo — rolling-origin (varios cortes temporales) en vez de uno. Si el offline no predice el online, las "mejoras" offline no son fiables.

### Iteración 3 — validación rolling-origin + comparación de modelos

Con rolling-origin (4 folds expandiendo), el MAE offline ya es realista (≈ LB, no el ~16 optimista del corte único). Comparación por ciudad:

| modelo | MAE sj | MAE iq |
|---|---|---|
| seasonal-naive | 30.86 | **6.54** |
| LightGBM (L1) | 28.84 | 6.55 |
| NegBin GLM | 29.11 | 8.70 |
| **SARIMAX** | **22.18** | 7.74 |
| **elegido** | **SARIMAX** | **seasonal-naive** |

- **SARIMAX gana claro en sj** (22.2 vs 28.8 de LightGBM): AR/I/MA + estacionalidad de Fourier + clima exógeno modela los brotes mucho mejor que los árboles. La herramienta correcta para una serie temporal.
- **En iq nada supera al seasonal-naive**: serie demasiado ruidosa (~18% semanas a 0), la señal clima→casos es marginal. Aceptarlo es lo honesto.
- **NegBin GLM** (el enfoque del benchmark oficial) decepcionó: sin tuning y sin modelar autocorrelación temporal, se queda corto. Negativo documentado.
- **Diversidad del portfolio:** el slot estrella deja de ser gradient boosting (ya estaba en el slot 1) y pasa a **SARIMAX**, familia estadística de series temporales. LightGBM queda como referencia comparativa, junto al seasonal-naive.
- Estimación de LB pooled con esta config: **≈ 16** (vs ~24 de la submission 1). _Pendiente de confirmar submiteando._

### Submission 2 — la CV mejor SIGUIÓ sin predecir el LB

| | MAE LB público | vs sub 1 |
|---|---|---|
| Submission 2 (SARIMAX sj + naive iq) | **~30** | **+6 PEOR** |

**Lección dura y honesta:** la estimación de ~16 estaba mal. La validación rolling-origin, aunque más realista que el corte único, **tampoco predijo el leaderboard**: SARIMAX ganó en CV pero perdió en el test real.

- El test de sj son 260 semanas de futuro de una vez; los folds de CV validaban bloques internos más cortos con historia reciente cerca. **A horizonte largo SARIMAX se degrada** (el AR pierde fuerza, queda colgado de clima+estacionalidad, que extrapolan peor).
- Error de método: cambié **dos cosas a la vez** (sj y iq) → sin desglose por ciudad no se aísla la causa.
- Meta-lección: en este problema **ninguna CV offline es un oráculo fiable**; el test es un único bloque de futuro muy largo. Toca humildad: pocos cambios por submission y diagnóstico por ciudad.

### Iteración 4 — atacar el sobreajuste (pocas features + lag óptimo)

Diagnóstico: ~90 features sobre 936/520 filas = sobreajuste. Receta: 5 drivers, lag óptimo por variable, suavizado de features y predicción, modelo simple.

| | rolling-origin MAE sj | rolling-origin MAE iq |
|---|---|---|
| it.3 (LightGBM, ~90 feats) | 28.84 | 6.55 |
| **it.4 (lgbm-small, 5 feats + lag óptimo)** | **26.16** | 6.37 |

- **Reducir features + acertar el lag mejora sj** (28.84 → 26.16 en rolling-origin) → confirma que sobreajustábamos.
- **Lags óptimos en sj epidemiológicamente sensatos:** humedad lag 5 sem, dew point 6, temp media 8, temp mín 7 — el clima de hace **~1-2 meses** predice los casos de hoy.
- iq sigue siendo ruido (naive ≈ lgbm); se usa lgbm-small por robustez ante brotes (el naive los aplana).
- **Realismo:** el top del LB está en ~10-11; es un leaderboard muy comprimido. Estas mejoras van en la dirección correcta (menos overfit) pero cerrar a ese nivel es trabajo largo con rendimientos decrecientes.

### Resumen de submissions (LB público, MAE)

| # | Config | LB público |
|---|---|---|
| 1 | LightGBM ~90 features (ambas ciudades) | 24.29 |
| 2 | SARIMAX sj + seasonal-naive iq | 30.89 |
| 3 | **lgbm-small: 5 features + lag óptimo (ambas)** | **23.67** ← mejor |

**Conclusiones validadas en el leaderboard:**
- **Reducir features baja el error real** (24.29 → 23.67): el sobreajuste era el problema, confirmado en LB y no solo en CV.
- **La rolling-origin acertó la dirección** de it.4 (predijo mejora y la hubo); en cambio SARIMAX ganó en CV y perdió en LB. Lección: la rolling-origin es razonable para comparar modelos *simples y robustos*, no para extrapolaciones de horizonte largo (SARIMAX).
- La combo SARIMAX + naive fue claramente peor → descartada.

### Iteración 5 — ensemble por ciudad

Blend `lgbm-small + NegBin + seasonal-naive` con pesos por rolling-origin (simplex, paso 0.25):

| ciudad | pesos (lgbm, negbin, naive) | rolling MAE | vs it.4 |
|---|---|---|---|
| sj | (1.0, 0, 0) — lgbm puro | 26.16 | = |
| iq | (0.25, 0.25, 0.5) | **6.18** | −0.19 |

- **iq mejora** al promediar (reduce varianza en una serie ruidosa); **sj no** (los otros modelos son demasiado peores para aportar).
- Ganancia esperada en LB **marginal**: el error vive en sj (260/416 del pooled) y ahí no movemos nada. **Suelo de lo principled** — para bajar de verdad habría que resolver sj, y las familias razonables (boosting, SARIMAX, NegBin) ya se probaron.

### Iteración 6 — autorregresión recursiva (negativo)

Hipótesis: la autocorrelación de los casos (sj lag1 = 0.96) es la señal más fuerte; usarla recursivamente (predecir t → usarlo como lag de t+1) debería ayudar en sj.

| ciudad | sin-AR (it.4) | con-AR recursivo |
|---|---|---|
| sj | 26.16 | **58.62** ❌ |
| iq | 6.37 | 6.45 |

- **Falla estrepitosamente.** Al no observar los casos recientes en el test, el modelo se realimenta de sus propias predicciones y el **error se acumula** sobre 260 semanas (peor en los brotes: sobrepasa y propaga el sobrepaso).
- Refuerza la lección de fondo: DengAI es un **forecast de horizonte largo con señal climática débil**; la autocorrelación no es explotable. Por eso SARIMAX (que es autorregresivo) también falló en el LB.

### Conclusión del proyecto

Tras 6 iteraciones, el **mejor modelo legítimo es it.4 (lgbm-small, 5 features + lag óptimo): MAE LB 23.67**, ya por debajo del benchmark público (~25). Probadas y descartadas con criterio: SARIMAX, NegBin, ensemble (marginal) y autorregresión (negativo).

El top del leaderboard (~10-11) es, en mi lectura, producto de **sobreajuste al test público** (practice comp sin leaderboard privado, 17k participantes, 8 años de submissions diarias contra un test estático), no de una técnica generalizable que nos falte. El valor de este proyecto está en el **método**: validación rolling-origin honesta, diagnóstico de sobreajuste confirmado en el LB, y un mapa claro de qué familias de modelo funcionan y cuáles no en un forecasting epidemiológico real.
