"""
Proyecto 4 — DengAI: baseline
=============================

Decisiones de diseño (justificadas en el EDA):
  - Un modelo POR CIUDAD (sj e iq tienen escala/estacionalidad/histórico distintos).
  - Validación TEMPORAL (últimas semanas como holdout). CV aleatoria = leakage.
  - Sin lags del target: el test es un bloque futuro largo y no los tendríamos.
    La señal usable es el CLIMA (dado en test) + estacionalidad. El clima precede
    a los casos → usamos rolling/lags sobre variables climáticas.
  - Objetivo L1 (MAE, la métrica oficial). Predicción clip>=0 y redondeada a int.
  - Referencia honesta: seasonal-naive (media histórica por weekofyear).

Uso:
    python 04_dengai/pipeline.py
Salida:
    04_dengai/submissions/submission.csv  (subir manualmente a DrivenData)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "04_dengai"
OUT_DIR = Path(__file__).resolve().parent
SUB_DIR = OUT_DIR / "submissions"

BASE_FEATURES = [
    "ndvi_ne", "ndvi_nw", "ndvi_se", "ndvi_sw",
    "precipitation_amt_mm",
    "reanalysis_air_temp_k", "reanalysis_avg_temp_k", "reanalysis_dew_point_temp_k",
    "reanalysis_max_air_temp_k", "reanalysis_min_air_temp_k",
    "reanalysis_precip_amt_kg_per_m2", "reanalysis_relative_humidity_percent",
    "reanalysis_sat_precip_amt_mm", "reanalysis_specific_humidity_g_per_kg",
    "reanalysis_tdtr_k",
    "station_avg_temp_c", "station_diur_temp_rng_c",
    "station_max_temp_c", "station_min_temp_c", "station_precip_mm",
]

# Variables que la literatura asocia a transmisión de dengue (mosquito Aedes):
# humedad, temperatura y precipitación. Sobre estas calculamos rolling.
DRIVERS = [
    "reanalysis_specific_humidity_g_per_kg",
    "reanalysis_dew_point_temp_k",
    "reanalysis_relative_humidity_percent",
    "station_avg_temp_c",
    "reanalysis_air_temp_k",
    "precipitation_amt_mm",
    "reanalysis_precip_amt_kg_per_m2",
]
ROLL_WINDOWS = [4, 8, 12]

LGB_PARAMS = dict(
    objective="regression_l1",   # MAE directo (métrica oficial)
    n_estimators=2000,
    learning_rate=0.02,
    num_leaves=15,
    min_child_samples=15,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    verbose=-1,
)


def build_features(df_city: pd.DataFrame) -> pd.DataFrame:
    """Features sobre la serie COMPLETA de la ciudad (train+test concatenados).
    Todo lo que usamos (clima y derivados) está observado también en test, así
    que no hay leakage del futuro: rolling de clima en t usa clima de t-k..t."""
    df = df_city.sort_values("week_start_date").reset_index(drop=True).copy()

    # Imputación temporal del clima (suave): interpolación + relleno en bordes.
    df[BASE_FEATURES] = (
        df[BASE_FEATURES].interpolate(method="linear", limit_direction="both")
    )

    # Estacionalidad: weekofyear como señal cíclica.
    woy = df["weekofyear"].clip(upper=52)
    df["woy_sin"] = np.sin(2 * np.pi * woy / 52.0)
    df["woy_cos"] = np.cos(2 * np.pi * woy / 52.0)

    # Rolling de los drivers climáticos (el clima precede a los casos).
    feat_cols = list(BASE_FEATURES) + ["woy_sin", "woy_cos"]
    for col in DRIVERS:
        for w in ROLL_WINDOWS:
            name = f"{col}_roll{w}"
            df[name] = df[col].rolling(window=w, min_periods=1).mean()
            feat_cols.append(name)

    df.attrs["feat_cols"] = feat_cols
    return df


def temporal_val_mae(X, y, frac=0.75):
    """Holdout temporal: primeras frac semanas para train, resto validación."""
    n = len(X)
    cut = int(n * frac)
    Xtr, Xval = X.iloc[:cut], X.iloc[cut:]
    ytr, yval = y.iloc[:cut], y.iloc[cut:]
    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(
        Xtr, ytr,
        eval_set=[(Xval, yval)],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    pred = np.clip(np.round(model.predict(Xval)), 0, None)
    mae = mean_absolute_error(yval, pred)
    best_iter = model.best_iteration_ or LGB_PARAMS["n_estimators"]
    return mae, best_iter, (yval, pred)


def seasonal_naive_mae(train_city, frac=0.75):
    """Referencia: predecir la media histórica por weekofyear (de la parte de train)."""
    n = len(train_city)
    cut = int(n * frac)
    tr, val = train_city.iloc[:cut], train_city.iloc[cut:]
    means = tr.groupby("weekofyear")["total_cases"].mean()
    overall = tr["total_cases"].mean()
    pred = val["weekofyear"].map(means).fillna(overall)
    pred = np.clip(np.round(pred), 0, None)
    return mean_absolute_error(val["total_cases"], pred)


def main():
    feat = pd.read_csv(DATA_DIR / "dengue_features_train.csv", parse_dates=["week_start_date"])
    lab = pd.read_csv(DATA_DIR / "dengue_labels_train.csv")
    test = pd.read_csv(DATA_DIR / "dengue_features_test.csv", parse_dates=["week_start_date"])
    train = feat.merge(lab, on=["city", "year", "weekofyear"], how="left")

    print("=" * 60)
    print("DengAI — baseline (LightGBM L1 por ciudad)")
    print("=" * 60)

    submissions = []
    val_records = []
    for city in ["sj", "iq"]:
        tr_city = train[train.city == city].copy()
        te_city = test[test.city == city].copy()

        # Concatenar para features (clima observado en ambos); marcar origen.
        tr_city["_is_test"] = False
        te_city["_is_test"] = True
        full = pd.concat([tr_city, te_city], ignore_index=True, sort=False)
        full = build_features(full)
        feat_cols = full.attrs["feat_cols"]

        tr_f = full[~full._is_test].copy()
        te_f = full[full._is_test].copy()
        X, y = tr_f[feat_cols], tr_f["total_cases"]

        # 1) Referencia seasonal-naive y 2) modelo, ambos en holdout temporal.
        naive_mae = seasonal_naive_mae(tr_f.sort_values("week_start_date"))
        model_mae, best_iter, _ = temporal_val_mae(X, y)
        val_records.append((city, naive_mae, model_mae, len(tr_f), len(te_f)))
        print(f"\n[{city}]  val MAE  seasonal-naive={naive_mae:.2f}  |  LightGBM={model_mae:.2f}  "
              f"(best_iter={best_iter})")

        # Refit en todo el train de la ciudad con el nº de árboles del holdout.
        params = dict(LGB_PARAMS); params["n_estimators"] = best_iter
        final = lgb.LGBMRegressor(**params)
        final.fit(X, y)
        pred = np.clip(np.round(final.predict(te_f[feat_cols])), 0, None).astype(int)

        sub = te_f[["city", "year", "weekofyear"]].copy()
        sub["total_cases"] = pred
        submissions.append(sub)

    print("\n" + "-" * 60)
    print("Resumen validación temporal (MAE; menor = mejor):")
    print(f"{'ciudad':<8}{'naive':>10}{'lgbm':>10}{'mejora':>10}")
    for city, nm, mm, _, _ in val_records:
        print(f"{city:<8}{nm:>10.2f}{mm:>10.2f}{nm - mm:>10.2f}")

    # Submission: respetar el ORDEN exacto del submission_format.
    sub_format = pd.read_csv(DATA_DIR / "submission_format.csv")
    full_sub = pd.concat(submissions, ignore_index=True)
    out = sub_format[["city", "year", "weekofyear"]].merge(
        full_sub, on=["city", "year", "weekofyear"], how="left"
    )
    assert out["total_cases"].notna().all(), "Faltan predicciones para alguna fila del test"
    out["total_cases"] = out["total_cases"].astype(int)

    SUB_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SUB_DIR / "submission.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSubmission escrita en {out_path}  ({len(out)} filas)")


if __name__ == "__main__":
    main()
