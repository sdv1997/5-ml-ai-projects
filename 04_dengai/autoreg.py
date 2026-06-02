"""
Proyecto 4 — DengAI it.6: autorregresión recursiva (usar los casos recientes)
=============================================================================

La señal más fuerte de los datos es la AUTOCORRELACIÓN de los casos (sj: 0.96 con
la semana previa) y hasta ahora la tirábamos porque el test es futuro. Aquí la
usamos de forma honesta: el test va justo detrás del train, así que las primeras
semanas tienen casos reales recientes. Predecimos RECURSIVAMENTE (predigo t →
lo uso como lag para t+1...), anclando al pasado real y dejando que el clima +
estacionalidad tomen el relevo a horizonte largo.

Features: clima (5 drivers con lag óptimo + suavizado, de model_lagopt) + estacionalidad
+ lags del propio total_cases [1,2,3,4,52]. Modelo LightGBM L1. Validación rolling-origin
con la MISMA recursión que en test (sin leakage: en cada fold solo se conocen los casos
del tramo de train).

Uso:    python 04_dengai/autoreg.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_lagopt as ml

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

CASE_LAGS = [1, 2, 3, 4, 52]


def add_caselags(df):
    cols = []
    for L in CASE_LAGS:
        c = f"cl{L}"
        df[c] = df["total_cases"].shift(L)
        cols.append(c)
    return cols


def fit_model(df_rows, feats):
    m = lgb.LGBMRegressor(**ml.LGB_PARAMS)
    m.fit(df_rows[feats], df_rows["total_cases"])
    return m


def recursive_predict(model, static_mat, cases_known, start, end, feats_order_static):
    """Predice posiciones [start, end) recursivamente. cases_known: array float con
    casos reales hasta start (resto NaN). Devuelve el array completo relleno."""
    y = cases_known.copy()
    for pos in range(start, end):
        feat = list(static_mat[pos])
        for L in CASE_LAGS:
            feat.append(y[pos - L] if pos - L >= 0 else np.nan)
        p = model.predict(np.array(feat, dtype=float).reshape(1, -1))[0]
        y[pos] = max(round(p), 0)
    return y


def rolling_cv_ar(tr_f, static_feats):
    tr_f = tr_f.sort_values("week_start_date").reset_index(drop=True)
    lag_cols = add_caselags(tr_f)
    feats = static_feats + lag_cols
    static_mat = tr_f[static_feats].to_numpy()
    cases = tr_f["total_cases"].to_numpy(dtype=float)

    maes = []
    for tr_idx, val_idx in TimeSeriesSplit(n_splits=4).split(tr_f):
        k = tr_idx[-1] + 1  # primeras k filas conocidas
        train_rows = tr_f.iloc[tr_idx].dropna(subset=lag_cols)
        model = fit_model(train_rows, feats)
        known = cases.copy(); known[k:] = np.nan
        s, e = val_idx[0], val_idx[-1] + 1
        y = recursive_predict(model, static_mat, known, s, e, static_feats)
        maes.append(ml.mae_int(cases[s:e], y[s:e]))
    return float(np.mean(maes))


def rolling_cv_noar(tr_f, static_feats):
    """Referencia: mismo modelo SIN lags de casos (como it.4)."""
    tr_f = tr_f.sort_values("week_start_date").reset_index(drop=True)
    maes = []
    for tr_idx, val_idx in TimeSeriesSplit(n_splits=4).split(tr_f):
        tr, val = tr_f.iloc[tr_idx], tr_f.iloc[val_idx]
        m = lgb.LGBMRegressor(**ml.LGB_PARAMS)
        m.fit(tr[static_feats], tr["total_cases"])
        pred = ml.smooth(np.clip(m.predict(val[static_feats]), 0, None), ml.SMOOTH_PRED)
        maes.append(ml.mae_int(val["total_cases"].values, pred))
    return float(np.mean(maes))


def main():
    feat = pd.read_csv(ml.DATA_DIR / "dengue_features_train.csv", parse_dates=["week_start_date"])
    lab = pd.read_csv(ml.DATA_DIR / "dengue_labels_train.csv")
    test = pd.read_csv(ml.DATA_DIR / "dengue_features_test.csv", parse_dates=["week_start_date"])
    train = feat.merge(lab, on=["city", "year", "weekofyear"], how="left")

    print("=" * 70)
    print("DengAI it.6 — autorregresión recursiva (rolling-origin)")
    print("=" * 70)

    submissions = []
    for city in ["sj", "iq"]:
        trc = train[train.city == city].copy(); trc["_is_test"] = False
        tec = test[test.city == city].copy(); tec["_is_test"] = True
        full = ml.prep_city(pd.concat([trc, tec], ignore_index=True, sort=False))
        tr_only = full[~full._is_test].copy()
        lags = ml.select_lags(tr_only)
        static_feats = ml.add_lagged(full, lags)

        tr_f = full[~full._is_test].copy().sort_values("week_start_date").reset_index(drop=True)
        mae_noar = rolling_cv_noar(tr_f, static_feats)
        mae_ar = rolling_cv_ar(tr_f, static_feats)
        print(f"\n[{city}] rolling-origin MAE:  sin-AR(it.4)={mae_noar:.2f}   con-AR={mae_ar:.2f}")

        # Submission final con AR: train completo conocido, recursión sobre el test.
        full_sorted = full.sort_values("week_start_date").reset_index(drop=True)
        lag_cols = add_caselags(full_sorted)
        feats = static_feats + lag_cols
        train_rows = full_sorted[~full_sorted._is_test].dropna(subset=lag_cols)
        model = fit_model(train_rows, feats)
        static_mat = full_sorted[static_feats].to_numpy()
        cases = full_sorted["total_cases"].to_numpy(dtype=float)
        test_pos = np.where(full_sorted["_is_test"].to_numpy())[0]
        s, e = test_pos[0], test_pos[-1] + 1
        known = cases.copy(); known[s:] = np.nan
        y = recursive_predict(model, static_mat, known, s, e, static_feats)
        sub = full_sorted.loc[full_sorted._is_test, ["city", "year", "weekofyear"]].copy()
        sub["total_cases"] = np.clip(np.round(y[s:e]), 0, None).astype(int)
        submissions.append(sub)

    fmt = pd.read_csv(ml.DATA_DIR / "submission_format.csv")
    out = fmt[["city", "year", "weekofyear"]].merge(
        pd.concat(submissions, ignore_index=True), on=["city", "year", "weekofyear"], how="left")
    out["total_cases"] = out["total_cases"].astype(int)
    ml.SUB_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(ml.SUB_DIR / "submission_ar.csv", index=False)
    print(f"\nSubmission (AR) escrita: {ml.SUB_DIR / 'submission_ar.csv'}  ({len(out)} filas)")
    print("(no sobrescribe submission.csv; revisa el MAE rolling antes de decidir)")


if __name__ == "__main__":
    main()
