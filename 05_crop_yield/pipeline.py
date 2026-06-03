"""
Proyecto 5 — CGIAR Crop Yield: pipeline (CPU)
=============================================

Satélite + geoespacial SIN GPU: features tabulares de los parches Sentinel-2 →
LightGBM (solución ligera, dentro de las reglas de Zindi). Métrica oficial: RMSE.

Incorpora ideas de la solución 4ª clasificada (CV ~1.59), validadas aquí:
  - **MEDIANA** del parche por mes (robusta a nubes; sin máscara QA).
  - Índices de vegetación NDVI, EVI, **SAVI** + min/max en la temporada de maíz.
  - **Ratios red-edge** B7/B5 y B7/B6 (median por mes).
  - Clima de la **temporada (meses 3–9)** desde additional_info (pr/tmmn/tmmx, media 4 años) + suelo ISRIC.
  - **Filtro de calidad de etiqueta** (entrenar solo con Quality ∈ {1,3}).
  - Doble validación: KFold aleatorio (≈LB) + GroupKFold por año (estrés).

Uso:  python 05_crop_yield/pipeline.py            # usa cache si existe
      python 05_crop_yield/pipeline.py --refeat   # re-extrae features de array
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import mean_squared_error

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT_DIR = Path(__file__).resolve().parent
DATA_DIR = OUT_DIR / "data"
TRAIN_ARR = DATA_DIR / "image_arrays_train" / "image_arrays_train"
TEST_ARR = DATA_DIR / "image_arrays_test" / "image_arrays_test"
SUB_DIR = OUT_DIR / "submissions"

S2_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
GROW = [3, 4, 5, 6, 7, 8]          # temporada de maíz (Kenia)
SEED = 42


def band_index():
    lines = [l.strip() for l in open(DATA_DIR / "bandnames.txt") if l.strip()]
    idx, steps = {}, set()
    for i, l in enumerate(lines):
        t, rest = l.split("_", 1)
        idx[(int(t), rest)] = i; steps.add(int(t))
    return idx, sorted(steps)


IDX, STEPS = band_index()


def field_features(arr):
    """Features por campo usando la MEDIANA del parche (robusta a nubes)."""
    arr = arr.astype(np.float32)
    feat = {}
    ndvi_series = []
    for t in STEPS:
        def med(name):
            return float(np.median(arr[IDX[(t, f"S2_{name}")]]))
        b = {name: arr[IDX[(t, f"S2_{name}")]] for name in S2_BANDS}
        b8, b4, b2, b3 = b["B8"], b["B4"], b["B2"], b["B3"]
        ndvi = (b8 - b4) / (b8 + b4 + 1e-6)
        evi = 2.5 * (b8 / 1e4 - b4 / 1e4) / (b8 / 1e4 + 6 * b4 / 1e4 - 7.5 * b2 / 1e4 + 1)
        savi = (b8 - b4) / (b8 + b4 + 0.725) * 1.725
        feat[f"NDVI_med_{t}"] = float(np.median(ndvi)); ndvi_series.append(feat[f"NDVI_med_{t}"])
        feat[f"EVI_med_{t}"] = float(np.median(evi))
        # ratios red-edge
        feat[f"B7B5_med_{t}"] = med("B7") / (med("B5") + 1e-6)
        feat[f"B7B6_med_{t}"] = med("B7") / (med("B6") + 1e-6)
        if t in GROW:
            feat[f"NDVI_max_{t}"] = float(ndvi.max()); feat[f"NDVI_min_{t}"] = float(ndvi.min())
            feat[f"EVI_max_{t}"] = float(evi.max()); feat[f"SAVI_max_{t}"] = float(savi.max())
    nv = np.array(ndvi_series, dtype=np.float32)
    feat["NDVI_mean"] = nv.mean(); feat["NDVI_std"] = nv.std()
    feat["NDVI_max"] = nv.max(); feat["NDVI_integral"] = float(nv.sum())
    feat["NDVI_peak_month"] = int(np.argmax(nv)); feat["NDVI_range"] = float(nv.max() - nv.min())
    return feat


def extract(field_ids, arr_dir, desc):
    rows = []
    for i, fid in enumerate(field_ids):
        f = field_features(np.load(arr_dir / f"{fid}.npy")); f["Field_ID"] = fid; rows.append(f)
        if (i + 1) % 500 == 0:
            print(f"  {desc}: {i+1}/{len(field_ids)}", flush=True)
    return pd.DataFrame(rows)


def array_features(refeat=False):
    ftr, fte = DATA_DIR / "features_med_train.csv", DATA_DIR / "features_med_test.csv"
    if not refeat and ftr.exists() and fte.exists():
        return pd.read_csv(ftr), pd.read_csv(fte)
    print("Extrayendo features (mediana del parche) — unos minutos…")
    tr_ids = pd.read_csv(DATA_DIR / "Train.csv").Field_ID
    te_ids = pd.read_csv(DATA_DIR / "SampleSubmission.csv").Field_ID
    Xtr = extract(tr_ids, TRAIN_ARR, "train"); Xte = extract(te_ids, TEST_ARR, "test")
    Xtr.to_csv(ftr, index=False); Xte.to_csv(fte, index=False)
    return Xtr, Xte


def tabular_extras():
    """Clima de temporada (meses 3–9, pr/tmmn/tmmx, media 4 años) + suelo ISRIC."""
    fa = pd.read_csv(DATA_DIR / "fields_w_additional_info.csv")
    out = fa[["Field_ID"] + [c for c in fa.columns if c.startswith("soil_")]].copy()
    for m in GROW:
        for v in ["pr", "tmmn", "tmmx"]:
            cc = [f"climate_{y}_{m}_{v}" for y in [2016, 2017, 2018, 2019]]
            cc = [c for c in cc if c in fa.columns]
            if cc:
                out[f"clim_{m}_{v}"] = fa[cc].mean(axis=1)
    return out


def get_data(refeat=False):
    Xtr, Xte = array_features(refeat)
    tr = pd.read_csv(DATA_DIR / "Train.csv")
    te = pd.read_csv(DATA_DIR / "SampleSubmission.csv")[["Field_ID"]].merge(
        pd.read_csv(DATA_DIR / "test_field_ids_with_year.csv"), on="Field_ID", how="left")
    extras = tabular_extras()
    Xtr = Xtr.merge(tr, on="Field_ID").merge(extras, on="Field_ID", how="left")
    Xte = Xte.merge(te, on="Field_ID", how="left").merge(extras, on="Field_ID", how="left")
    return Xtr, Xte


LGB = dict(objective="regression", n_estimators=3000, learning_rate=0.02, num_leaves=31,
           min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
           reg_lambda=3.0, random_state=SEED, verbose=-1)


def cv_rmse(X, feats, y, splitter, groups=None, log=True):
    oof = np.zeros(len(X)); yt = np.log1p(y) if log else y
    for tr_i, va_i in splitter.split(X[feats], yt, groups):
        m = lgb.LGBMRegressor(**LGB)
        m.fit(X[feats].iloc[tr_i], yt[tr_i], eval_set=[(X[feats].iloc[va_i], yt[va_i])],
              eval_metric="rmse", callbacks=[lgb.early_stopping(100, verbose=False)])
        p = m.predict(X[feats].iloc[va_i]); oof[va_i] = np.expm1(p) if log else p
    return mean_squared_error(y, np.clip(oof, 0, None)) ** 0.5


def main(refeat=False):
    Xtr, Xte = get_data(refeat)
    Xtr = Xtr[Xtr.Quality.isin([1, 3])].reset_index(drop=True)   # filtro de calidad (truco 4º)
    feats = [c for c in Xtr.columns if c not in ("Field_ID", "Yield", "Quality") and c in Xte.columns]
    y = Xtr["Yield"].values
    print(f"\n{len(Xtr)} train (Quality 1,3) · {len(Xte)} test · {len(feats)} features")
    base = mean_squared_error(y, np.full_like(y, y.mean())) ** 0.5

    cv = {log: cv_rmse(Xtr, feats, y, KFold(5, shuffle=True, random_state=SEED), log=log)
          for log in (False, True)}
    best_log = min(cv, key=cv.get)
    for log in (False, True):
        print(f"  KFold-aleatorio (≈LB) log={log}: RMSE {cv[log]:.4f}" + ("  ← elegido" if log == best_log else ""))
    rmse_year = cv_rmse(Xtr, feats, y, GroupKFold(Xtr.Year.nunique()), Xtr.Year.values, log=best_log)
    print(f"  GroupKFold-año: RMSE {rmse_year:.4f}  |  baseline media {base:.4f}")

    # refit (3 seeds) con la mejor config
    yt = np.log1p(y) if best_log else y
    preds = np.zeros(len(Xte))
    for s in range(3):
        p = dict(LGB); p["random_state"] = SEED + s
        m = lgb.LGBMRegressor(**p); m.fit(Xtr[feats], yt)
        pr = m.predict(Xte[feats]); preds += (np.expm1(pr) if best_log else pr) / 3
    SUB_DIR.mkdir(parents=True, exist_ok=True)
    sub = Xte[["Field_ID"]].copy(); sub["Yield"] = np.clip(preds, 0, None)
    sub.to_csv(SUB_DIR / "submission.csv", index=False)
    print(f"Submission escrita: {SUB_DIR/'submission.csv'}  ({len(sub)} filas)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--refeat", action="store_true")
    main(ap.parse_args().refeat)
