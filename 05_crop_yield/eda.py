"""
Proyecto 5 — CGIAR Crop Yield: EDA
==================================

Explora:
  - el target Yield (distribución, por Quality/Year)
  - la estructura de los arrays (360, 41, 41) = 12 meses × 30 capas (Sentinel-2 + clima)
  - NDVI de ejemplo a lo largo del año (fenología) en un campo
  - fields_w_additional_info.csv: ¿hay coordenadas? cobertura train/test
Salida: 05_crop_yield/eda.png + resumen por consola.
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT_DIR = Path(__file__).resolve().parent
DATA_DIR = OUT_DIR / "data"
TRAIN_ARR = DATA_DIR / "image_arrays_train" / "image_arrays_train"


def parse_bands():
    """bandnames.txt: línea i = capa i, formato '{t}_{src}_{name}'. Devuelve listas
    de índices por (timestep, nombre) y el nº de timesteps."""
    lines = [l.strip() for l in open(DATA_DIR / "bandnames.txt") if l.strip()]
    idx = {}            # (t, name) -> layer index
    steps = set()
    for i, l in enumerate(lines):
        t, rest = l.split("_", 1)
        idx[(int(t), rest)] = i
        steps.add(int(t))
    return idx, sorted(steps), lines


def main():
    df = pd.read_csv(DATA_DIR / "Train.csv")
    print("=" * 60); print("CGIAR Crop Yield — EDA"); print("=" * 60)
    print(f"\nTrain: {len(df)} campos | Test: {len(pd.read_csv(DATA_DIR/'SampleSubmission.csv'))} campos")
    print(f"\nYield: media={df.Yield.mean():.3f} std={df.Yield.std():.3f} "
          f"min={df.Yield.min():.3f} max={df.Yield.max():.3f}")
    print(f"Year: {df.Year.value_counts().to_dict()}")
    print(f"Quality (1=mejor?): {df.Quality.value_counts().sort_index().to_dict()}")
    print("Yield medio por Quality:")
    print(df.groupby("Quality").Yield.agg(["mean", "std", "count"]).round(3).to_string())

    idx, steps, lines = parse_bands()
    print(f"\nCapas: {len(lines)} = {len(steps)} timesteps × {len(lines)//len(steps)} canales/timestep")
    chans = sorted({l.split('_', 1)[1] for l in lines})
    print("Canales:", chans)

    # additional info: coordenadas?
    fa = pd.read_csv(DATA_DIR / "fields_w_additional_info.csv")
    coordish = [c for c in fa.columns if any(k in c.lower() for k in ["lat", "lon", "_x", "_y", "coord", "geom"])]
    print(f"\nfields_w_additional_info: {fa.shape[0]} campos × {fa.shape[1]} cols. "
          f"Columnas tipo coordenada: {coordish if coordish else 'NINGUNA'}")
    tr_ids = set(df.Field_ID); te_ids = set(pd.read_csv(DATA_DIR/'SampleSubmission.csv').Field_ID)
    cov = set(fa.Field_ID)
    print(f"  cobertura: train {len(tr_ids & cov)}/{len(tr_ids)}  test {len(te_ids & cov)}/{len(te_ids)}")

    # NDVI mensual (fenología) de unos campos
    def ndvi_series(fid):
        arr = np.load(TRAIN_ARR / f"{fid}.npy").astype(np.float32)
        out = []
        for t in steps:
            b8 = arr[idx[(t, "S2_B8")]]; b4 = arr[idx[(t, "S2_B4")]]
            ndvi = (b8 - b4) / (b8 + b4 + 1e-6)
            out.append(np.nanmean(ndvi))
        return np.array(out)

    # ---- plots ----
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    ax[0, 0].hist(df.Yield, bins=40, color="tab:green"); ax[0, 0].set_title("Distribución de Yield")
    ax[0, 0].set_xlabel("yield")

    df.groupby("Quality").Yield.mean().plot(kind="bar", ax=ax[0, 1], color="tab:orange")
    ax[0, 1].set_title("Yield medio por Quality")

    # NDVI temporal de los 5 campos con mayor y 5 con menor yield
    top = df.nlargest(5, "Yield"); bot = df.nsmallest(5, "Yield")
    for _, r in top.iterrows():
        ax[0, 2].plot(steps, ndvi_series(r.Field_ID), color="tab:green", alpha=0.6)
    for _, r in bot.iterrows():
        ax[0, 2].plot(steps, ndvi_series(r.Field_ID), color="tab:red", alpha=0.6)
    ax[0, 2].set_title("NDVI mensual (verde=top yield, rojo=bottom)")
    ax[0, 2].set_xlabel("mes"); ax[0, 2].set_ylabel("NDVI medio")

    # imagen RGB aproximada (B4,B3,B2) de un campo en un mes central
    fid = df.iloc[0].Field_ID
    arr = np.load(TRAIN_ARR / f"{fid}.npy").astype(np.float32)
    t = steps[len(steps)//2]
    rgb = np.stack([arr[idx[(t, "S2_B4")]], arr[idx[(t, "S2_B3")]], arr[idx[(t, "S2_B2")]]], -1)
    rgb = np.clip(rgb / (np.percentile(rgb, 99) + 1e-6), 0, 1)
    ax[1, 0].imshow(rgb); ax[1, 0].set_title(f"RGB parche 41×41 ({fid}, mes {t})"); ax[1, 0].axis("off")

    ndvi = (arr[idx[(t, "S2_B8")]] - arr[idx[(t, "S2_B4")]]) / (arr[idx[(t, "S2_B8")]] + arr[idx[(t, "S2_B4")]] + 1e-6)
    im = ax[1, 1].imshow(ndvi, cmap="RdYlGn", vmin=-0.2, vmax=0.9); ax[1, 1].set_title("NDVI del parche"); ax[1, 1].axis("off")
    plt.colorbar(im, ax=ax[1, 1], fraction=0.046)

    # NDVI medio del dataset por mes (curva estacional global)
    sample = df.sample(min(200, len(df)), random_state=42)
    curves = np.array([ndvi_series(f) for f in sample.Field_ID])
    ax[1, 2].plot(steps, np.nanmean(curves, 0), "b-o"); ax[1, 2].fill_between(
        steps, np.nanpercentile(curves, 25, 0), np.nanpercentile(curves, 75, 0), alpha=0.2)
    ax[1, 2].set_title("NDVI medio por mes (200 campos)"); ax[1, 2].set_xlabel("mes")

    fig.suptitle("CGIAR Crop Yield — EDA")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "eda.png", dpi=100)
    print(f"\nPlot guardado en {OUT_DIR/'eda.png'}")


if __name__ == "__main__":
    main()
