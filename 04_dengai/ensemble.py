"""
Proyecto 4 — DengAI it.5: ensemble por ciudad con pesos por rolling-origin
==========================================================================

Sobre it.4 (5 features + lag óptimo + suavizado), promediamos modelos poco
correlacionados — el siguiente paso clásico para bajar MAE en targets ruidosos:

  ensemble = w_lgb · lgbm-small  +  w_nb · NegBin  +  w_naive · seasonal-naive

Los pesos (simplex, paso 0.25) se eligen POR CIUDAD minimizando el MAE de
validación rolling-origin. Grid grueso a propósito para no sobreajustar a los
4 folds. Reutiliza la ingeniería de features de model_lagopt.

Uso:    python 04_dengai/ensemble.py
Salida: tabla + 04_dengai/submissions/submission.csv
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_lagopt as ml

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = ["lgbm-small", "negbin", "naive"]
BASE_FN = {
    "lgbm-small": lambda tr, val, f: ml.pred_lgbm(tr, val, f),
    "negbin":     lambda tr, val, f: ml.pred_negbin(tr, val, f),
    "naive":      lambda tr, val, f: ml.pred_naive(tr, val),
}


def weight_grid(step=0.25):
    rng = [round(i * step, 3) for i in range(int(round(1 / step)) + 1)]
    out = []
    for a in rng:
        for b in rng:
            c = round(1 - a - b, 3)
            if -1e-9 <= c <= 1 + 1e-9:
                out.append((a, b, max(c, 0.0)))
    return out


def base_preds(tr, val, feats):
    return {n: np.clip(BASE_FN[n](tr, val, feats), 0, None) for n in BASE}


def main():
    feat = pd.read_csv(ml.DATA_DIR / "dengue_features_train.csv", parse_dates=["week_start_date"])
    lab = pd.read_csv(ml.DATA_DIR / "dengue_labels_train.csv")
    test = pd.read_csv(ml.DATA_DIR / "dengue_features_test.csv", parse_dates=["week_start_date"])
    train = feat.merge(lab, on=["city", "year", "weekofyear"], how="left")

    print("=" * 70)
    print("DengAI it.5 — ensemble por ciudad (pesos por rolling-origin)")
    print("=" * 70)

    grid = weight_grid(0.25)
    submissions, chosen = [], []
    for city in ["sj", "iq"]:
        trc = train[train.city == city].copy(); trc["_is_test"] = False
        tec = test[test.city == city].copy(); tec["_is_test"] = True
        full = ml.prep_city(pd.concat([trc, tec], ignore_index=True, sort=False))
        tr_f = full[~full._is_test].copy()
        lags = ml.select_lags(tr_f)
        feats = ml.add_lagged(full, lags)
        tr_f = full[~full._is_test].copy()
        te_f = full[full._is_test].copy()

        # Precalcular predicciones base por fold (una vez), luego barrer pesos.
        tr_f = tr_f.sort_values("week_start_date").reset_index(drop=True)
        folds = []
        for tr_idx, val_idx in TimeSeriesSplit(n_splits=4).split(tr_f):
            tr, val = tr_f.iloc[tr_idx], tr_f.iloc[val_idx]
            folds.append((val["total_cases"].values, base_preds(tr, val, feats)))

        best_w, best_mae = None, np.inf
        for w in grid:
            maes = []
            for yval, preds in folds:
                ens = w[0] * preds["lgbm-small"] + w[1] * preds["negbin"] + w[2] * preds["naive"]
                maes.append(ml.mae_int(yval, ml.smooth(ens, ml.SMOOTH_PRED)))
            m = float(np.mean(maes))
            if m < best_mae:
                best_mae, best_w = m, w
        chosen.append((city, best_w, best_mae))
        # MAE de cada base en solitario, para contexto.
        solo = {}
        for n in BASE:
            solo[n] = float(np.mean([ml.mae_int(y, ml.smooth(p[n], ml.SMOOTH_PRED)) for y, p in folds]))
        print(f"\n[{city}] MAE rolling base: " + "  ".join(f"{n}={solo[n]:.2f}" for n in BASE))
        print(f"  → ensemble w(lgbm,negbin,naive)={best_w}  MAE={best_mae:.2f}")

        # Refit en todo el train, predecir test, ensemble.
        preds_te = base_preds(tr_f, te_f, feats)
        ens = best_w[0] * preds_te["lgbm-small"] + best_w[1] * preds_te["negbin"] + best_w[2] * preds_te["naive"]
        pred = np.clip(np.round(ml.smooth(ens, ml.SMOOTH_PRED)), 0, None).astype(int)
        s = te_f[["city", "year", "weekofyear"]].copy(); s["total_cases"] = pred
        submissions.append(s)

    print("\n" + "-" * 70)
    print("Pesos elegidos por ciudad (lgbm, negbin, naive):")
    for city, w, m in chosen:
        print(f"  {city}: {w}  rolling MAE={m:.2f}")

    fmt = pd.read_csv(ml.DATA_DIR / "submission_format.csv")
    out = fmt[["city", "year", "weekofyear"]].merge(
        pd.concat(submissions, ignore_index=True), on=["city", "year", "weekofyear"], how="left")
    out["total_cases"] = out["total_cases"].astype(int)
    ml.SUB_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(ml.SUB_DIR / "submission.csv", index=False)
    print(f"\nSubmission escrita: {ml.SUB_DIR / 'submission.csv'}  ({len(out)} filas)")


if __name__ == "__main__":
    main()
