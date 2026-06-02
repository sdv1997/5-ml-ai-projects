"""
Proyecto 4 — DengAI: pipeline con selección de modelo por ciudad
================================================================

Sobre el baseline (LightGBM L1 + rolling de clima + estacionalidad) añadimos y
COMPARAMOS, por ciudad y en holdout temporal:
  - objetivo: L1 (MAE) vs Poisson vs Tweedie  (datos de conteo)
  - lags explícitos del clima (no solo medias móviles)
  - suavizado de la predicción (los casos reales no saltan semana a semana)
  - blend con seasonal-naive (útil donde el modelo no gana a la media estacional, p.ej. iq)

Elegimos la mejor combinación por ciudad minimizando MAE de validación, reentrenamos
en todo el train de la ciudad y generamos la submission.

Uso:    python 04_dengai/pipeline.py
Salida: 04_dengai/submissions/submission.csv
"""
import sys
import itertools
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

# Drivers epidemiológicos (humedad/temp/precip): el clima precede a los casos.
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
LAGS = [1, 2, 3, 4, 8, 12]

PARAMS = dict(
    n_estimators=3000,
    learning_rate=0.02,
    num_leaves=15,
    min_child_samples=15,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.6,
    reg_lambda=1.0,
    random_state=42,
    verbose=-1,
)
OBJECTIVES = {
    "l1": dict(objective="regression_l1"),
    "poisson": dict(objective="poisson"),
    "tweedie": dict(objective="tweedie", tweedie_variance_power=1.3),
}
SMOOTH_WINDOWS = [1, 3, 5]
BLEND_ALPHAS = [0.0, 0.25, 0.5]


def build_features(df_city: pd.DataFrame) -> pd.DataFrame:
    """Features sobre la serie completa de la ciudad (train+test son semanas
    consecutivas → timeline climática continua, sin leakage del target)."""
    df = df_city.sort_values("week_start_date").reset_index(drop=True).copy()
    df[BASE_FEATURES] = df[BASE_FEATURES].interpolate(method="linear", limit_direction="both")

    woy = df["weekofyear"].clip(upper=52)
    df["woy_sin"] = np.sin(2 * np.pi * woy / 52.0)
    df["woy_cos"] = np.cos(2 * np.pi * woy / 52.0)

    feat_cols = list(BASE_FEATURES) + ["woy_sin", "woy_cos"]
    for col in DRIVERS:
        for w in ROLL_WINDOWS:
            name = f"{col}_roll{w}"
            df[name] = df[col].rolling(window=w, min_periods=1).mean()
            feat_cols.append(name)
        for lag in LAGS:
            name = f"{col}_lag{lag}"
            df[name] = df[col].shift(lag)   # NaN al inicio → LightGBM los maneja
            feat_cols.append(name)

    df.attrs["feat_cols"] = feat_cols
    return df


def fit(X, y, objective_key, n_est=None, eval_set=None):
    params = dict(PARAMS); params.update(OBJECTIVES[objective_key])
    if n_est is not None:
        params["n_estimators"] = n_est
    model = lgb.LGBMRegressor(**params)
    if eval_set is not None:
        model.fit(X, y, eval_set=[eval_set], eval_metric="l1",
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    else:
        model.fit(X, y)
    return model


def smooth(arr, w):
    if w <= 1:
        return np.asarray(arr, dtype=float)
    s = pd.Series(arr).rolling(window=w, center=True, min_periods=1).mean()
    return s.values


def mae_int(y, pred):
    return mean_absolute_error(y, np.clip(np.round(pred), 0, None))


def seasonal_means(df):
    return df.groupby("weekofyear")["total_cases"].mean()


def select_config(tr_f, feat_cols, frac=0.75):
    """Barre objetivo × suavizado × blend en holdout temporal. Devuelve la mejor."""
    tr_f = tr_f.sort_values("week_start_date").reset_index(drop=True)
    n = len(tr_f); cut = int(n * frac)
    tr, val = tr_f.iloc[:cut], tr_f.iloc[cut:]
    Xtr, ytr = tr[feat_cols], tr["total_cases"]
    Xval, yval = val[feat_cols], val["total_cases"]

    naive_means = seasonal_means(tr)
    naive_val = val["weekofyear"].map(naive_means).fillna(ytr.mean()).values

    rows = []
    for obj in OBJECTIVES:
        model = fit(Xtr, ytr, obj, eval_set=(Xval, yval))
        bi = model.best_iteration_ or PARAMS["n_estimators"]
        raw = np.clip(model.predict(Xval), 0, None)
        for w, alpha in itertools.product(SMOOTH_WINDOWS, BLEND_ALPHAS):
            blended = (1 - alpha) * smooth(raw, w) + alpha * naive_val
            rows.append({"obj": obj, "smooth": w, "alpha": alpha,
                         "best_iter": bi, "mae": mae_int(yval, blended)})
    res = pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)
    return res


def main():
    feat = pd.read_csv(DATA_DIR / "dengue_features_train.csv", parse_dates=["week_start_date"])
    lab = pd.read_csv(DATA_DIR / "dengue_labels_train.csv")
    test = pd.read_csv(DATA_DIR / "dengue_features_test.csv", parse_dates=["week_start_date"])
    train = feat.merge(lab, on=["city", "year", "weekofyear"], how="left")

    print("=" * 64)
    print("DengAI — selección de modelo por ciudad (holdout temporal)")
    print("=" * 64)

    submissions = []
    summary = []
    for city in ["sj", "iq"]:
        tr_city = train[train.city == city].copy(); tr_city["_is_test"] = False
        te_city = test[test.city == city].copy(); te_city["_is_test"] = True
        full = build_features(pd.concat([tr_city, te_city], ignore_index=True, sort=False))
        feat_cols = full.attrs["feat_cols"]
        tr_f = full[~full._is_test].copy()
        te_f = full[full._is_test].copy()

        res = select_config(tr_f, feat_cols)
        best = res.iloc[0]
        print(f"\n[{city}]  top configs (MAE val):")
        print(res.head(5).to_string(index=False))
        print(f"  → elegido: obj={best.obj}  smooth={int(best['smooth'])}  "
              f"alpha={best.alpha}  MAE={best.mae:.2f}")
        summary.append((city, best))

        # Refit en todo el train de la ciudad con la config elegida.
        X, y = tr_f[feat_cols], tr_f["total_cases"]
        model = fit(X, y, best.obj, n_est=int(best.best_iter))
        raw = np.clip(model.predict(te_f[feat_cols]), 0, None)
        naive_means = seasonal_means(tr_f)
        naive_te = te_f["weekofyear"].map(naive_means).fillna(y.mean()).values
        pred = (1 - best.alpha) * smooth(raw, int(best["smooth"])) + best.alpha * naive_te
        pred = np.clip(np.round(pred), 0, None).astype(int)

        sub = te_f[["city", "year", "weekofyear"]].copy()
        sub["total_cases"] = pred
        submissions.append(sub)

    print("\n" + "-" * 64)
    print("Config final por ciudad:")
    for city, b in summary:
        print(f"  {city}: obj={b.obj:<8} smooth={int(b['smooth'])} alpha={b.alpha}  "
              f"val MAE={b.mae:.2f}")

    # Submission en el ORDEN exacto del formato.
    sub_format = pd.read_csv(DATA_DIR / "submission_format.csv")
    full_sub = pd.concat(submissions, ignore_index=True)
    out = sub_format[["city", "year", "weekofyear"]].merge(
        full_sub, on=["city", "year", "weekofyear"], how="left")
    assert out["total_cases"].notna().all(), "Faltan predicciones para alguna fila"
    out["total_cases"] = out["total_cases"].astype(int)

    SUB_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SUB_DIR / "submission.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSubmission escrita en {out_path}  ({len(out)} filas)")


if __name__ == "__main__":
    main()
