"""
Proyecto 4 — DengAI: experimentos (el recorrido, con sus negativos)
===================================================================

Consolida las alternativas que probamos y descartamos, todas evaluadas con la
MISMA validación rolling-origin que el pipeline final. Comparte la ingeniería de
features con pipeline.py (pocas features + lag óptimo + suavizado).

Bloques:
  A) Comparación de familias de modelo : seasonal-naive, LightGBM, NegBin GLM, SARIMAX
  B) Ensemble por ciudad (pesos por rolling-origin)
  C) Autorregresión recursiva (NEGATIVO: el error se acumula a horizonte largo)

Resumen de conclusiones (detalle y números en el README):
  - LightGBM con pocas features + lag óptimo = mejor y más robusto (→ pipeline.py).
  - SARIMAX gana en CV pero pierde en el leaderboard (extrapola mal a 260 semanas).
  - NegBin (el benchmark oficial) se queda corto sin tuning ni dinámica temporal.
  - Ensemble ayuda algo en iq, nada en sj.
  - Autorregresión recursiva empeora muchísimo sj (realimenta su propio error).

Uso:    python 04_dengai/experiments.py
"""
import sys
import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from patsy import dmatrix
from sklearn.model_selection import TimeSeriesSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline as pl  # prep_city, select_lags, add_lagged, smooth, mae_int, fit_lgbm, load, ...

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def build_city(city, train, test):
    trc = train[train.city == city].copy(); trc["_is_test"] = False
    tec = test[test.city == city].copy(); tec["_is_test"] = True
    full = pl.prep_city(pd.concat([trc, tec], ignore_index=True, sort=False))
    tr_f = full[~full._is_test].copy()
    feats = pl.add_lagged(full, pl.select_lags(tr_f))
    return full, full[~full._is_test].copy(), full[full._is_test].copy(), feats


# ---- modelos --------------------------------------------------------------
def pred_naive(tr, val, feats):
    means = tr.groupby("weekofyear")["total_cases"].mean()
    return val["weekofyear"].map(means).fillna(tr["total_cases"].mean()).values


def pred_lgbm(tr, val, feats):
    return np.clip(pl.fit_lgbm(tr, feats).predict(val[feats]), 0, None)


def _xy(tr, val, feats):
    mu = tr[feats].mean()
    Xtr = tr[feats].fillna(mu); Xval = val[feats].fillna(mu)
    sd = Xtr.std().replace(0, 1.0)
    return ((Xtr - mu) / sd).values, ((Xval - mu) / sd).values


def pred_negbin(tr, val, feats):
    Xtr, Xval = _xy(tr, val, feats)
    Xtr = sm.add_constant(Xtr, has_constant="add"); Xval = sm.add_constant(Xval, has_constant="add")
    best = None
    for alpha in [0.2, 0.5, 1.0, 2.0]:
        try:
            m = sm.GLM(tr["total_cases"].values, Xtr,
                       family=sm.families.NegativeBinomial(alpha=alpha)).fit(maxiter=200)
            if best is None or m.aic < best[0]:
                best = (m.aic, m)
        except Exception:
            continue
    return np.clip(best[1].predict(Xval), 0, None) if best else np.full(len(val), tr["total_cases"].mean())


def pred_sarimax(tr, val, feats):
    Xtr, Xval = _xy(tr, val, feats)
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
    return np.clip(np.expm1(np.asarray(best[1].get_forecast(steps=len(val), exog=Xval).predicted_mean)), 0, None)


def pred_negbin_gam(tr, val, feats):
    """NegBin GAM estilo DLNM: splines no lineales cr() sobre los drivers con lag
    óptimo + estacionalidad cíclica cc(weekofyear). El modelo de libro en
    epidemiología climática. Pierde igualmente: optimiza verosimilitud (media),
    no MAE (mediana)."""
    lag_cols = [c for c in feats if c not in ("woy_sin", "woy_cos")]
    def design_df(d):
        out = pd.DataFrame({f"g{i}": d[c].values for i, c in enumerate(lag_cols)})
        out["woy"] = d["weekofyear"].clip(upper=52).values
        return out.interpolate(limit_direction="both").fillna(out.mean())
    Dtr, Dval = design_df(tr), design_df(val)
    terms = "cc(woy, df=6) + " + " + ".join(f"cr(g{i}, df=4)" for i in range(len(lag_cols)))
    dm_tr = dmatrix(terms, Dtr, return_type="dataframe")
    dm_val = dmatrix(dm_tr.design_info, Dval, return_type="dataframe")
    best = None
    for alpha in [0.3, 0.6, 1.0, 1.5]:
        try:
            m = sm.GLM(tr["total_cases"].values, dm_tr,
                       family=sm.families.NegativeBinomial(alpha=alpha)).fit(maxiter=300)
            if best is None or m.aic < best[0]:
                best = (m.aic, m)
        except Exception:
            continue
    return np.clip(best[1].predict(dm_val), 0, None) if best else np.full(len(val), tr["total_cases"].mean())


MODELS = {"seasonal-naive": pred_naive, "lightgbm": pred_lgbm, "negbin-glm": pred_negbin,
          "negbin-gam": pred_negbin_gam, "sarimax": pred_sarimax}


def rolling_cv(tr_f, feats, fns, n_splits=4):
    tr_f = tr_f.sort_values("week_start_date").reset_index(drop=True)
    scores = {n: [] for n in fns}
    for tr_idx, val_idx in TimeSeriesSplit(n_splits=n_splits).split(tr_f):
        tr, val = tr_f.iloc[tr_idx], tr_f.iloc[val_idx]
        for n, fn in fns.items():
            try:
                pred = pl.smooth(fn(tr, val, feats), pl.SMOOTH_PRED)
                scores[n].append(pl.mae_int(val["total_cases"].values, pred))
            except Exception:
                scores[n].append(np.nan)
    return {n: float(np.nanmean(v)) for n, v in scores.items()}


# ---- C) autorregresión recursiva -----------------------------------------
CASE_LAGS = [1, 2, 3, 4, 52]


def _recursive(model, static_mat, cases_known, start, end):
    y = cases_known.copy()
    for pos in range(start, end):
        feat = list(static_mat[pos]) + [y[pos - L] if pos - L >= 0 else np.nan for L in CASE_LAGS]
        y[pos] = max(round(model.predict(np.array(feat, float).reshape(1, -1))[0]), 0)
    return y


def autoreg_rolling(tr_f, static_feats, n_splits=4):
    tr_f = tr_f.sort_values("week_start_date").reset_index(drop=True)
    for L in CASE_LAGS:
        tr_f[f"cl{L}"] = tr_f["total_cases"].shift(L)
    feats = static_feats + [f"cl{L}" for L in CASE_LAGS]
    static_mat = tr_f[static_feats].to_numpy()
    cases = tr_f["total_cases"].to_numpy(float)
    maes = []
    for tr_idx, val_idx in TimeSeriesSplit(n_splits=n_splits).split(tr_f):
        k = tr_idx[-1] + 1
        model = pl.fit_lgbm(tr_f.iloc[tr_idx].dropna(subset=[f"cl{L}" for L in CASE_LAGS]), feats)
        known = cases.copy(); known[k:] = np.nan
        s, e = val_idx[0], val_idx[-1] + 1
        y = _recursive(model, static_mat, known, s, e)
        maes.append(pl.mae_int(cases[s:e], y[s:e]))
    return float(np.mean(maes))


def weight_grid(step=0.25):
    rng = [round(i * step, 3) for i in range(int(round(1 / step)) + 1)]
    return [(a, b, round(1 - a - b, 3)) for a in rng for b in rng if -1e-9 <= 1 - a - b <= 1 + 1e-9]


def main():
    train, test = pl.load()
    for city in ["sj", "iq"]:
        full, tr_f, te_f, feats = build_city(city, train, test)
        print("=" * 64)
        print(f"Ciudad {city}")
        print("=" * 64)

        # A) comparación de familias
        cv = rolling_cv(tr_f, feats, MODELS)
        print("A) MAE rolling-origin por modelo:")
        for n, m in sorted(cv.items(), key=lambda kv: kv[1]):
            print(f"     {n:<16} {m:.2f}")

        # B) ensemble (lgbm + negbin + naive)
        tr_s = tr_f.sort_values("week_start_date").reset_index(drop=True)
        folds = []
        for tr_idx, val_idx in TimeSeriesSplit(n_splits=4).split(tr_s):
            tr, val = tr_s.iloc[tr_idx], tr_s.iloc[val_idx]
            folds.append((val["total_cases"].values,
                          {n: np.clip(MODELS[n](tr, val, feats), 0, None)
                           for n in ["lightgbm", "negbin-glm", "seasonal-naive"]}))
        best_w, best = None, np.inf
        for w in weight_grid():
            maes = [pl.mae_int(y, pl.smooth(w[0]*p["lightgbm"]+w[1]*p["negbin-glm"]+w[2]*p["seasonal-naive"], pl.SMOOTH_PRED))
                    for y, p in folds]
            if np.mean(maes) < best:
                best, best_w = float(np.mean(maes)), w
        print(f"B) ensemble mejor w(lgbm,negbin,naive)={best_w}  MAE={best:.2f}")

        # C) autorregresión recursiva
        print(f"C) autorregresión recursiva: MAE={autoreg_rolling(tr_f, feats):.2f}  (vs lightgbm {cv['lightgbm']:.2f})")
        print()


if __name__ == "__main__":
    main()
