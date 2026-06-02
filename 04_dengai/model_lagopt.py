"""
Proyecto 4 — DengAI it.4: pocas features, lag óptimo, suavizado
================================================================

Hipótesis (de diagnóstico propio + literatura): estábamos SOBREAJUSTANDO con ~90
features sobre 936/520 filas. Las soluciones top usan pocas features climáticas,
con el LAG correcto (el clima precede a los casos ~1-2 meses), SUAVIZADAS, y
predicciones también suavizadas. En un holdout de futuro largo, lo simple generaliza.

Receta:
  1) Pocos drivers climáticos (sin fuentes duplicadas).
  2) Suavizar cada driver (media móvil) → quita el ruido semanal.
  3) Lag ÓPTIMO por driver y ciudad: el que maximiza |corr| con casos en train.
  4) Modelo simple (NegBin GLM / LightGBM regularizado), elegido por rolling-origin.
  5) Suavizar la predicción final.

Uso:    python 04_dengai/model_lagopt.py
Salida: tabla + 04_dengai/submissions/submission.csv
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import statsmodels.api as sm
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

# Pocos drivers, sin duplicar fuentes (1 humedad específica, 1 dew point, 1 temp, 1 precip).
DRIVERS = [
    "reanalysis_specific_humidity_g_per_kg",
    "reanalysis_dew_point_temp_k",
    "station_avg_temp_c",
    "station_min_temp_c",
    "precipitation_amt_mm",
]
SMOOTH_FEAT = 5     # ventana de suavizado de las features (semanas)
LMAX = 20           # rango de búsqueda de lag (semanas)
SMOOTH_PRED = 3     # suavizado de la predicción
BASE_TO_FILL = DRIVERS  # interpolación de NaNs

LGB_PARAMS = dict(
    objective="regression_l1", n_estimators=2000, learning_rate=0.02,
    num_leaves=7, min_child_samples=25, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.9, reg_lambda=5.0, random_state=42, verbose=-1,
)


def prep_city(df_city):
    """Serie completa de la ciudad (train+test), interpola y suaviza drivers."""
    df = df_city.sort_values("week_start_date").reset_index(drop=True).copy()
    df[BASE_TO_FILL] = df[BASE_TO_FILL].interpolate(method="linear", limit_direction="both")
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


def mae_int(y, p):
    return mean_absolute_error(y, np.clip(np.round(p), 0, None))


def smooth(arr, w):
    if w <= 1:
        return np.asarray(arr, float)
    return pd.Series(arr).rolling(w, center=True, min_periods=1).mean().values


def pred_naive(tr, val):
    means = tr.groupby("weekofyear")["total_cases"].mean()
    return val["weekofyear"].map(means).fillna(tr["total_cases"].mean()).values


def pred_lgbm(tr, val, feats):
    m = lgb.LGBMRegressor(**LGB_PARAMS)
    m.fit(tr[feats], tr["total_cases"])
    return np.clip(m.predict(val[feats]), 0, None)


def pred_negbin(tr, val, feats):
    mu = tr[feats].mean(); sd = tr[feats].std().replace(0, 1.0)
    Xtr = sm.add_constant(((tr[feats] - mu) / sd).values, has_constant="add")
    Xval = sm.add_constant(((val[feats] - mu) / sd).values, has_constant="add")
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
        return np.full(len(val), ytr.mean())
    return np.clip(best[1].predict(Xval), 0, None)


MODELS = {
    "seasonal-naive": lambda tr, val, f: pred_naive(tr, val),
    "negbin-lagopt":  lambda tr, val, f: pred_negbin(tr, val, f),
    "lgbm-small":     lambda tr, val, f: pred_lgbm(tr, val, f),
}


def rolling_cv(tr_f, feats, n_splits=4):
    tr_f = tr_f.sort_values("week_start_date").reset_index(drop=True)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = {n: [] for n in MODELS}
    for tr_idx, val_idx in tscv.split(tr_f):
        tr, val = tr_f.iloc[tr_idx], tr_f.iloc[val_idx]
        for name, fn in MODELS.items():
            try:
                pred = smooth(fn(tr, val, feats), SMOOTH_PRED)
                scores[name].append(mae_int(val["total_cases"].values, pred))
            except Exception:
                scores[name].append(np.nan)
    return {n: np.nanmean(v) for n, v in scores.items()}


def main():
    feat = pd.read_csv(DATA_DIR / "dengue_features_train.csv", parse_dates=["week_start_date"])
    lab = pd.read_csv(DATA_DIR / "dengue_labels_train.csv")
    test = pd.read_csv(DATA_DIR / "dengue_features_test.csv", parse_dates=["week_start_date"])
    train = feat.merge(lab, on=["city", "year", "weekofyear"], how="left")

    print("=" * 70)
    print("DengAI it.4 — pocas features + lag óptimo + suavizado (rolling-origin)")
    print("=" * 70)

    submissions, chosen = [], []
    for city in ["sj", "iq"]:
        trc = train[train.city == city].copy(); trc["_is_test"] = False
        tec = test[test.city == city].copy(); tec["_is_test"] = True
        full = prep_city(pd.concat([trc, tec], ignore_index=True, sort=False))
        tr_f = full[~full._is_test].copy()

        lags = select_lags(tr_f)
        feats = add_lagged(full, lags)
        tr_f = full[~full._is_test].copy()
        te_f = full[full._is_test].copy()

        print(f"\n[{city}] lag óptimo por driver (semanas):")
        for d, L in lags.items():
            print(f"    {d:<42} lag={L}")

        cv = rolling_cv(tr_f, feats)
        ranked = sorted(cv.items(), key=lambda kv: kv[1])
        best_name = ranked[0][0]
        # Tie-break: si gana el naive pero un modelo está empatado (<EPS), prefiere
        # el modelo — el naive aplana los brotes y eso castiga en el test real.
        EPS = 0.15
        if best_name == "seasonal-naive":
            for n, m in ranked:
                if n != "seasonal-naive" and m - cv["seasonal-naive"] < EPS:
                    best_name = n
                    break
        chosen.append((city, best_name, cv))
        print(f"  MAE rolling-origin: " + "  ".join(f"{n}={m:.2f}" for n, m in ranked))
        print(f"  → elegido: {best_name} ({cv[best_name]:.2f})")

        pred = smooth(MODELS[best_name](tr_f, te_f, feats), SMOOTH_PRED)
        pred = np.clip(np.round(pred), 0, None).astype(int)
        s = te_f[["city", "year", "weekofyear"]].copy(); s["total_cases"] = pred
        submissions.append(s)

    print("\n" + "-" * 70)
    print("Elegido por ciudad:")
    for city, name, cv in chosen:
        print(f"  {city}: {name}  (rolling MAE {cv[name]:.2f})")

    fmt = pd.read_csv(DATA_DIR / "submission_format.csv")
    out = fmt[["city", "year", "weekofyear"]].merge(
        pd.concat(submissions, ignore_index=True), on=["city", "year", "weekofyear"], how="left")
    out["total_cases"] = out["total_cases"].astype(int)
    SUB_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(SUB_DIR / "submission.csv", index=False)
    print(f"\nSubmission escrita: {SUB_DIR / 'submission.csv'}  ({len(out)} filas)")


if __name__ == "__main__":
    main()
