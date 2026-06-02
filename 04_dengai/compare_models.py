"""
Proyecto 4 — DengAI: comparación de modelos con validación rolling-origin
=========================================================================

Motivación: la submission 1 reveló que un único holdout temporal era optimista
(val ~13 vs LB ~24). Aquí usamos VALIDACIÓN ROLLING-ORIGIN (varios orígenes que
expanden el train y validan el siguiente bloque) para que el MAE offline track-ee
el leaderboard, y comparamos cuatro modelos POR CIUDAD:

  - seasonal-naive   : media histórica por weekofyear (referencia honesta)
  - LightGBM (L1)    : gradient boosting, objetivo MAE (referencia del slot)
  - NegBin GLM       : el enfoque CANÓNICO del benchmark oficial de DengAI
                       (conteos sobredispersos: var >> media)
  - SARIMAX          : dinámica temporal (AR/I/MA) + clima como regresores exógenos
                       + estacionalidad anual vía términos de Fourier (s=52 es
                       demasiado caro como seasonal_order → se mete en el exog).

Uso:    python 04_dengai/compare_models.py
Salida: tabla comparativa + 04_dengai/submissions/submission.csv (mejor por ciudad)
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")
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
DRIVERS = [
    "reanalysis_specific_humidity_g_per_kg", "reanalysis_dew_point_temp_k",
    "reanalysis_relative_humidity_percent", "station_avg_temp_c",
    "reanalysis_air_temp_k", "precipitation_amt_mm", "reanalysis_precip_amt_kg_per_m2",
]
ROLL_WINDOWS = [4, 8, 12]
LAGS = [1, 2, 3, 4, 8, 12]

# Subconjunto curado para los modelos estadísticos (sensibles a colinealidad/escala).
STAT_BASE = [
    "reanalysis_specific_humidity_g_per_kg", "reanalysis_dew_point_temp_k",
    "station_avg_temp_c", "reanalysis_air_temp_k", "precipitation_amt_mm",
]
FOURIER = ["woy_sin", "woy_cos", "woy_sin2", "woy_cos2"]

LGB_PARAMS = dict(
    objective="regression_l1", n_estimators=3000, learning_rate=0.02,
    num_leaves=15, min_child_samples=15, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.6, reg_lambda=1.0, random_state=42, verbose=-1,
)


def build_features(df_city):
    df = df_city.sort_values("week_start_date").reset_index(drop=True).copy()
    df[BASE_FEATURES] = df[BASE_FEATURES].interpolate(method="linear", limit_direction="both")
    woy = df["weekofyear"].clip(upper=52)
    df["woy_sin"] = np.sin(2 * np.pi * woy / 52.0)
    df["woy_cos"] = np.cos(2 * np.pi * woy / 52.0)
    df["woy_sin2"] = np.sin(4 * np.pi * woy / 52.0)
    df["woy_cos2"] = np.cos(4 * np.pi * woy / 52.0)
    feat = list(BASE_FEATURES) + FOURIER
    for col in DRIVERS:
        for w in ROLL_WINDOWS:
            df[f"{col}_roll{w}"] = df[col].rolling(w, min_periods=1).mean(); feat.append(f"{col}_roll{w}")
        for lag in LAGS:
            df[f"{col}_lag{lag}"] = df[col].shift(lag); feat.append(f"{col}_lag{lag}")
    df.attrs["lgb_feats"] = feat
    df.attrs["stat_feats"] = STAT_BASE + [f"{c}_roll4" for c in STAT_BASE] + FOURIER
    return df


def mae_int(y, p):
    return mean_absolute_error(y, np.clip(np.round(p), 0, None))


# ---- modelos -------------------------------------------------------------
def pred_naive(tr, val):
    means = tr.groupby("weekofyear")["total_cases"].mean()
    return val["weekofyear"].map(means).fillna(tr["total_cases"].mean()).values


def pred_lgbm(tr, val, feats):
    m = lgb.LGBMRegressor(**LGB_PARAMS)
    m.fit(tr[feats], tr["total_cases"])
    return np.clip(m.predict(val[feats]), 0, None)


def _standardize(tr, val, cols):
    mu = tr[cols].mean(); sd = tr[cols].std().replace(0, 1.0)
    return ((tr[cols] - mu) / sd).values, ((val[cols] - mu) / sd).values


def pred_negbin(tr, val, cols):
    Xtr, Xval = _standardize(tr, val, cols)
    Xtr = sm.add_constant(Xtr, has_constant="add")
    Xval = sm.add_constant(Xval, has_constant="add")
    ytr = tr["total_cases"].values
    best = None
    for alpha in [0.2, 0.5, 1.0, 2.0]:
        try:
            m = sm.GLM(ytr, Xtr, family=sm.families.NegativeBinomial(alpha=alpha)).fit(maxiter=200)
            if best is None or m.aic < best[0]:
                best = (m.aic, m)
        except Exception:
            continue
    if best is None:
        m = sm.GLM(ytr, Xtr, family=sm.families.Poisson()).fit()
        return np.clip(m.predict(Xval), 0, None)
    return np.clip(best[1].predict(Xval), 0, None)


def pred_sarimax(tr, val, cols):
    Xtr, Xval = _standardize(tr, val, cols)
    y_log = np.log1p(tr["total_cases"].values)
    best = None
    for order in [(1, 0, 1), (2, 0, 2), (1, 1, 1), (2, 1, 1)]:
        try:
            m = SARIMAX(y_log, exog=Xtr, order=order, seasonal_order=(0, 0, 0, 0),
                        enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=200)
            if best is None or m.aic < best[0]:
                best = (m.aic, m)
        except Exception:
            continue
    if best is None:
        return np.full(len(val), tr["total_cases"].mean())
    fc = best[1].get_forecast(steps=len(val), exog=Xval).predicted_mean
    return np.clip(np.expm1(np.asarray(fc)), 0, None)


MODELS = {
    "seasonal-naive": lambda tr, val, lf, sf: pred_naive(tr, val),
    "lightgbm-l1":    lambda tr, val, lf, sf: pred_lgbm(tr, val, lf),
    "negbin-glm":     lambda tr, val, lf, sf: pred_negbin(tr, val, sf),
    "sarimax":        lambda tr, val, lf, sf: pred_sarimax(tr, val, sf),
}


def rolling_origin_cv(tr_f, lgb_feats, stat_feats, n_splits=4):
    tr_f = tr_f.sort_values("week_start_date").reset_index(drop=True)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = {name: [] for name in MODELS}
    for tr_idx, val_idx in tscv.split(tr_f):
        tr, val = tr_f.iloc[tr_idx], tr_f.iloc[val_idx]
        for name, fn in MODELS.items():
            try:
                pred = fn(tr, val, lgb_feats, stat_feats)
                scores[name].append(mae_int(val["total_cases"].values, pred))
            except Exception as e:
                scores[name].append(np.nan)
    return {name: np.nanmean(v) for name, v in scores.items()}


def refit_predict(model_name, tr_f, te_f, lgb_feats, stat_feats):
    return MODELS[model_name](tr_f, te_f, lgb_feats, stat_feats)


def main():
    feat = pd.read_csv(DATA_DIR / "dengue_features_train.csv", parse_dates=["week_start_date"])
    lab = pd.read_csv(DATA_DIR / "dengue_labels_train.csv")
    test = pd.read_csv(DATA_DIR / "dengue_features_test.csv", parse_dates=["week_start_date"])
    train = feat.merge(lab, on=["city", "year", "weekofyear"], how="left")

    print("=" * 70)
    print("DengAI — comparación de modelos (validación rolling-origin, 4 folds)")
    print("=" * 70)

    submissions = []
    chosen = []
    for city in ["sj", "iq"]:
        tr_city = train[train.city == city].copy(); tr_city["_is_test"] = False
        te_city = test[test.city == city].copy(); te_city["_is_test"] = True
        full = build_features(pd.concat([tr_city, te_city], ignore_index=True, sort=False))
        lgb_feats, stat_feats = full.attrs["lgb_feats"], full.attrs["stat_feats"]
        tr_f = full[~full._is_test].copy()
        te_f = full[full._is_test].copy()

        cv = rolling_origin_cv(tr_f, lgb_feats, stat_feats)
        ranked = sorted(cv.items(), key=lambda kv: kv[1])
        best_name = ranked[0][0]
        chosen.append((city, best_name, cv))

        print(f"\n[{city}]  MAE medio rolling-origin:")
        for name, mae in ranked:
            mark = "  <- mejor" if name == best_name else ""
            print(f"    {name:<16} {mae:6.2f}{mark}")

        pred = np.clip(np.round(refit_predict(best_name, tr_f, te_f, lgb_feats, stat_feats)), 0, None).astype(int)
        sub = te_f[["city", "year", "weekofyear"]].copy()
        sub["total_cases"] = pred
        submissions.append(sub)

    print("\n" + "-" * 70)
    print("Modelo elegido por ciudad:")
    for city, name, cv in chosen:
        print(f"  {city}: {name}  (MAE rolling {cv[name]:.2f})")

    sub_format = pd.read_csv(DATA_DIR / "submission_format.csv")
    out = sub_format[["city", "year", "weekofyear"]].merge(
        pd.concat(submissions, ignore_index=True), on=["city", "year", "weekofyear"], how="left")
    assert out["total_cases"].notna().all(), "Faltan predicciones"
    out["total_cases"] = out["total_cases"].astype(int)
    SUB_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(SUB_DIR / "submission.csv", index=False)
    print(f"\nSubmission escrita en {SUB_DIR / 'submission.csv'}  ({len(out)} filas)")


if __name__ == "__main__":
    main()
