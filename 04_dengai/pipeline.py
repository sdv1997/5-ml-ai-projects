"""
Proyecto 4 — DengAI: pipeline final
===================================

Modelo final tras 6 iteraciones (ver experiments.py para las alternativas
probadas y descartadas, y el README para la narrativa). Mejor resultado:
MAE LB público 23.67 (por debajo del benchmark oficial ~25).

Receta ganadora — la clave fue ATACAR EL SOBREAJUSTE:
  - Un modelo por ciudad (sj e iq son casi otro problema).
  - POCAS features (5 drivers climáticos, sin fuentes duplicadas).
  - LAG ÓPTIMO por variable: el clima precede a los casos ~1-2 meses; se elige
    el lag que maximiza |corr| con los casos en train (sj: humedad 5 sem, temp 8...).
  - Suavizado de features y de la predicción (el dengue responde a condiciones
    sostenidas, no al ruido semanal).
  - LightGBM regularizado con objetivo L1 (= métrica MAE). Clip>=0 + entero.
  - Validación rolling-origin (varios orígenes temporales); CV aleatoria miente.

Uso:    python 04_dengai/pipeline.py
Salida: 04_dengai/submissions/submission.csv
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT_DIR = Path(__file__).resolve().parent
DATA_DIR = OUT_DIR / "data"
SUB_DIR = OUT_DIR / "submissions"

# Pocos drivers, sin duplicar fuentes (1 humedad específica, 1 dew point, 2 temp, 1 precip).
DRIVERS = [
    "reanalysis_specific_humidity_g_per_kg",
    "reanalysis_dew_point_temp_k",
    "station_avg_temp_c",
    "station_min_temp_c",
    "precipitation_amt_mm",
]
SMOOTH_FEAT = 5     # ventana de suavizado de features (semanas)
LMAX = 20           # rango de búsqueda de lag (semanas)
SMOOTH_PRED = 3     # suavizado de la predicción

LGB_PARAMS = dict(
    objective="regression_l1", n_estimators=2000, learning_rate=0.02,
    num_leaves=7, min_child_samples=25, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.9, reg_lambda=5.0, random_state=42, verbose=-1,
)


def prep_city(df_city):
    """Serie completa de la ciudad (train+test), interpola NaNs y suaviza drivers."""
    df = df_city.sort_values("week_start_date").reset_index(drop=True).copy()
    df[DRIVERS] = df[DRIVERS].interpolate(method="linear", limit_direction="both")
    woy = df["weekofyear"].clip(upper=52)
    df["woy_sin"] = np.sin(2 * np.pi * woy / 52.0)
    df["woy_cos"] = np.cos(2 * np.pi * woy / 52.0)
    for d in DRIVERS:
        df[d + "_sm"] = df[d].rolling(SMOOTH_FEAT, min_periods=1).mean()
    return df


def select_lags(df_train):
    """Lag óptimo por driver = el que maximiza |corr(driver_sm.shift(L), casos)| en train."""
    chosen = {}
    for d in DRIVERS:
        sm_series = df_train[d + "_sm"]
        best_lag, best_corr = 0, -1.0
        for L in range(0, LMAX + 1):
            c = sm_series.shift(L).corr(df_train["total_cases"])
            if pd.notna(c) and abs(c) > best_corr:
                best_lag, best_corr = L, abs(c)
        chosen[d] = best_lag
    return chosen


def add_lagged(df, lags):
    cols = ["woy_sin", "woy_cos"]
    for d, L in lags.items():
        name = f"{d}_lag{L}"
        df[name] = df[d + "_sm"].shift(L)
        cols.append(name)
    return cols


def smooth(arr, w):
    if w <= 1:
        return np.asarray(arr, float)
    return pd.Series(arr).rolling(w, center=True, min_periods=1).mean().values


def mae_int(y, p):
    return mean_absolute_error(y, np.clip(np.round(p), 0, None))


def fit_lgbm(tr, feats):
    m = lgb.LGBMRegressor(**LGB_PARAMS)
    m.fit(tr[feats], tr["total_cases"])
    return m


def rolling_mae(tr_f, feats, n_splits=4):
    tr_f = tr_f.sort_values("week_start_date").reset_index(drop=True)
    maes = []
    for tr_idx, val_idx in TimeSeriesSplit(n_splits=n_splits).split(tr_f):
        tr, val = tr_f.iloc[tr_idx], tr_f.iloc[val_idx]
        pred = smooth(np.clip(fit_lgbm(tr, feats).predict(val[feats]), 0, None), SMOOTH_PRED)
        maes.append(mae_int(val["total_cases"].values, pred))
    return float(np.mean(maes))


def load():
    feat = pd.read_csv(DATA_DIR / "dengue_features_train.csv", parse_dates=["week_start_date"])
    lab = pd.read_csv(DATA_DIR / "dengue_labels_train.csv")
    test = pd.read_csv(DATA_DIR / "dengue_features_test.csv", parse_dates=["week_start_date"])
    return feat.merge(lab, on=["city", "year", "weekofyear"], how="left"), test


def main():
    train, test = load()
    print("=" * 64)
    print("DengAI — pipeline final (LightGBM lag-opt por ciudad)")
    print("=" * 64)

    submissions = []
    for city in ["sj", "iq"]:
        trc = train[train.city == city].copy(); trc["_is_test"] = False
        tec = test[test.city == city].copy(); tec["_is_test"] = True
        full = prep_city(pd.concat([trc, tec], ignore_index=True, sort=False))
        tr_f = full[~full._is_test].copy()
        lags = select_lags(tr_f)
        feats = add_lagged(full, lags)
        tr_f = full[~full._is_test].copy()
        te_f = full[full._is_test].copy()

        print(f"\n[{city}] lag óptimo (semanas): " +
              ", ".join(f"{d.split('_')[-1] if False else d}={L}" for d, L in lags.items()))
        print(f"  MAE rolling-origin = {rolling_mae(tr_f, feats):.2f}")

        pred = smooth(np.clip(fit_lgbm(tr_f, feats).predict(te_f[feats]), 0, None), SMOOTH_PRED)
        s = te_f[["city", "year", "weekofyear"]].copy()
        s["total_cases"] = np.clip(np.round(pred), 0, None).astype(int)
        submissions.append(s)

    fmt = pd.read_csv(DATA_DIR / "submission_format.csv")
    out = fmt[["city", "year", "weekofyear"]].merge(
        pd.concat(submissions, ignore_index=True), on=["city", "year", "weekofyear"], how="left")
    assert out["total_cases"].notna().all()
    out["total_cases"] = out["total_cases"].astype(int)
    SUB_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(SUB_DIR / "submission.csv", index=False)
    print(f"\nSubmission escrita: {SUB_DIR / 'submission.csv'}  ({len(out)} filas)")


if __name__ == "__main__":
    main()
