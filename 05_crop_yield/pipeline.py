"""
Proyecto 5 — CGIAR Crop Yield: pipeline (CPU)
=============================================

Satélite + geoespacial SIN GPU: en vez de meter las imágenes en una CNN, agregamos
los parches Sentinel-2 (41×41) a FEATURES TABULARES y modelamos con LightGBM —
enfoque competitivo real en rendimiento de cultivos.

Pasos:
  1) Extracción de features por campo desde el array (360,41,41) = 12 meses × 30 canales:
       - máscara de nubes con la banda QA60 (bits 10/11)
       - por mes: media (sobre píxeles válidos) de cada banda S2 + NDVI/EVI/NDWI + clima
       - features temporales (mean/std/min/max por señal) + fenología del NDVI
         (mes de pico, integral, rango) + las 12 NDVI mensuales
       - + suelo (ISRIC) de fields_w_additional_info.csv
     (se cachea en data/features_*.csv para no re-extraer)
  2) Validación GroupKFold por AÑO (2016–2019) → RMSE honesto.
  3) LightGBM regresión (L2). Refit en todo el train → submission.

Uso:    python 05_crop_yield/pipeline.py          # extrae (si falta), CV y submission
        python 05_crop_yield/pipeline.py --refeat  # fuerza re-extracción de features
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
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

S2_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
CLIM_VARS = ["aet", "def", "pdsi", "pet", "pr", "ro", "soil", "srad", "swe", "tmmn", "tmmx", "vap", "vpd", "vs"]
CLOUD_BITS = (1 << 10) | (1 << 11)   # QA60: opaque clouds + cirrus


def band_index():
    lines = [l.strip() for l in open(DATA_DIR / "bandnames.txt") if l.strip()]
    idx = {}
    steps = set()
    for i, l in enumerate(lines):
        t, rest = l.split("_", 1)
        idx[(int(t), rest)] = i
        steps.add(int(t))
    return idx, sorted(steps)


IDX, STEPS = band_index()


def field_features(arr):
    """arr (360,41,41) -> dict de features para un campo."""
    arr = arr.astype(np.float32)
    monthly = {f"S2_{b}": [] for b in S2_BANDS}
    monthly.update({"NDVI": [], "EVI": [], "NDWI": []})
    monthly.update({f"CLIM_{c}": [] for c in CLIM_VARS})

    for t in STEPS:
        qa = arr[IDX[(t, "S2_QA60")]].astype(np.int64)
        valid = (qa & CLOUD_BITS) == 0
        if valid.sum() < 20:           # casi todo nubes → usa todo el parche
            valid = np.ones_like(valid, dtype=bool)

        def bmean(name):
            return float(arr[IDX[(t, name)]][valid].mean())

        for b in S2_BANDS:
            monthly[f"S2_{b}"].append(bmean(f"S2_{b}"))
        b8 = arr[IDX[(t, "S2_B8")]]; b4 = arr[IDX[(t, "S2_B4")]]
        b2 = arr[IDX[(t, "S2_B2")]]; b3 = arr[IDX[(t, "S2_B3")]]; b11 = arr[IDX[(t, "S2_B11")]]
        ndvi = (b8 - b4) / (b8 + b4 + 1e-6)
        # EVI con reflectancia (bandas en DN ~*1e4 → se escala a 0-1)
        evi = 2.5 * (b8 / 1e4 - b4 / 1e4) / (b8 / 1e4 + 6 * b4 / 1e4 - 7.5 * b2 / 1e4 + 1)
        ndwi = (b3 - b8) / (b3 + b8 + 1e-6)
        monthly["NDVI"].append(float(ndvi[valid].mean()))
        monthly["EVI"].append(float(evi[valid].mean()))
        monthly["NDWI"].append(float(ndwi[valid].mean()))
        for c in CLIM_VARS:
            monthly[f"CLIM_{c}"].append(bmean(f"CLIM_{c}"))

    feat = {}
    for sig, vals in monthly.items():
        v = np.array(vals, dtype=np.float32)
        feat[f"{sig}_mean"] = v.mean(); feat[f"{sig}_std"] = v.std()
        feat[f"{sig}_min"] = v.min(); feat[f"{sig}_max"] = v.max()
    # fenología del NDVI
    ndvi = np.array(monthly["NDVI"], dtype=np.float32)
    feat["NDVI_peak_month"] = int(np.argmax(ndvi))
    feat["NDVI_integral"] = float(ndvi.sum())
    feat["NDVI_range"] = float(ndvi.max() - ndvi.min())
    for t in STEPS:
        feat[f"NDVI_m{t}"] = float(ndvi[t])
    return feat


def extract(field_ids, arr_dir, desc):
    rows = []
    for i, fid in enumerate(field_ids):
        f = field_features(np.load(arr_dir / f"{fid}.npy"))
        f["Field_ID"] = fid
        rows.append(f)
        if (i + 1) % 500 == 0:
            print(f"  {desc}: {i+1}/{len(field_ids)}", flush=True)
    return pd.DataFrame(rows)


def get_features(refeat=False):
    ftr, fte = DATA_DIR / "features_train.csv", DATA_DIR / "features_test.csv"
    if not refeat and ftr.exists() and fte.exists():
        print("Cargando features cacheadas…")
        return pd.read_csv(ftr), pd.read_csv(fte)
    print("Extrayendo features de los parches satélite (puede tardar unos minutos)…")
    tr = pd.read_csv(DATA_DIR / "Train.csv")
    te = pd.read_csv(DATA_DIR / "SampleSubmission.csv")[["Field_ID"]].merge(
        pd.read_csv(DATA_DIR / "test_field_ids_with_year.csv"), on="Field_ID", how="left")
    Xtr = extract(tr.Field_ID, TRAIN_ARR, "train").merge(tr, on="Field_ID")
    Xte = extract(te.Field_ID, TEST_ARR, "test").merge(te, on="Field_ID", how="left")
    # suelo (ISRIC) de la info adicional
    fa = pd.read_csv(DATA_DIR / "fields_w_additional_info.csv")
    soil = fa[["Field_ID"] + [c for c in fa.columns if c.startswith("soil_")]]
    Xtr = Xtr.merge(soil, on="Field_ID", how="left"); Xte = Xte.merge(soil, on="Field_ID", how="left")
    Xtr.to_csv(ftr, index=False); Xte.to_csv(fte, index=False)
    return Xtr, Xte


LGB = dict(objective="regression", n_estimators=2000, learning_rate=0.02, num_leaves=31,
           min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
           reg_lambda=2.0, random_state=42, verbose=-1)


def main(refeat=False):
    Xtr, Xte = get_features(refeat)
    drop = ["Field_ID", "Yield", "Quality"]
    feats = [c for c in Xtr.columns if c not in drop and c in Xte.columns]
    print(f"\n{len(Xtr)} train · {len(Xte)} test · {len(feats)} features")

    y = Xtr["Yield"].values
    groups = Xtr["Year"].values
    oof = np.zeros(len(Xtr)); test_pred = np.zeros(len(Xte)); n_splits = Xtr.Year.nunique()
    gkf = GroupKFold(n_splits=n_splits)
    print(f"\nGroupKFold por año ({n_splits} folds):")
    for k, (tr_i, va_i) in enumerate(gkf.split(Xtr[feats], y, groups)):
        m = lgb.LGBMRegressor(**LGB)
        m.fit(Xtr[feats].iloc[tr_i], y[tr_i],
              eval_set=[(Xtr[feats].iloc[va_i], y[va_i])], eval_metric="rmse",
              callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va_i] = m.predict(Xtr[feats].iloc[va_i])
        test_pred += m.predict(Xte[feats]) / n_splits
        yr = int(Xtr.Year.iloc[va_i[0]])
        rmse = mean_squared_error(y[va_i], oof[va_i]) ** 0.5
        print(f"  fold año={yr}: RMSE {rmse:.4f}  (best_iter {m.best_iteration_})")

    cv = mean_squared_error(y, oof) ** 0.5
    base = mean_squared_error(y, np.full_like(y, y.mean())) ** 0.5
    print(f"\nRMSE OOF (GroupKFold año): {cv:.4f}   |   baseline media: {base:.4f}")

    SUB_DIR.mkdir(parents=True, exist_ok=True)
    sub = Xte[["Field_ID"]].copy(); sub["Yield"] = np.clip(test_pred, 0, None)
    sub.to_csv(SUB_DIR / "submission.csv", index=False)
    print(f"Submission escrita: {SUB_DIR/'submission.csv'}  ({len(sub)} filas)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--refeat", action="store_true")
    main(ap.parse_args().refeat)
