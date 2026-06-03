"""
Proyecto 5 — Understanding Clouds: EDA
======================================

Explora train.csv (máscaras RLE por imagen×clase) y muestrea imágenes con sus
máscaras superpuestas. Lo que importa para diseñar el modelo:
  - frecuencia de cada clase (Fish/Flower/Gravel/Sugar) y % de máscaras vacías
  - cuántas clases coexisten por imagen (multi-label)
  - tamaño de imagen / máscara
  - inspección visual: ¿son blobs gruesos? (→ se puede reescalar agresivo)

Salida: 05_understanding_clouds/eda.png + resumen por consola.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rle import rle_decode, CLASSES, IMG_SHAPE

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT_DIR = Path(__file__).resolve().parent
DATA_DIR = OUT_DIR / "data"


def find(name, kind="file"):
    """Busca recursivamente un fichero/carpeta dentro de data/ (estructura variable)."""
    for p in DATA_DIR.rglob(name):
        if (kind == "file" and p.is_file()) or (kind == "dir" and p.is_dir()):
            return p
    return None


def main():
    train_csv = find("train.csv")
    train_dir = find("train_images", "dir")
    if train_csv is None:
        print("No encuentro train.csv en data/. ¿Terminó la extracción?")
        print("Contenido de data/:", [p.name for p in DATA_DIR.glob("*")])
        return

    df = pd.read_csv(train_csv)
    df["image"] = df["Image_Label"].str.split("_").str[0]
    df["label"] = df["Image_Label"].str.split("_").str[1]
    df["has_mask"] = df["EncodedPixels"].notna()

    print("=" * 60)
    print("Understanding Clouds — EDA")
    print("=" * 60)
    n_img = df["image"].nunique()
    print(f"\nImágenes: {n_img}  |  filas (imagen×clase): {len(df)}")

    print("\nPresencia de cada clase (máscara no vacía):")
    pres = df[df.has_mask].groupby("label").size().reindex(CLASSES)
    for c in CLASSES:
        print(f"  {c:<8} {pres[c]:>5}  ({pres[c]/n_img*100:4.1f}% de las imágenes)")

    per_img = df[df.has_mask].groupby("image").size()
    per_img = per_img.reindex(df.image.unique(), fill_value=0)
    print("\nNº de clases por imagen:")
    for k in range(0, 5):
        print(f"  {k} clases: {(per_img == k).sum():>5} imágenes ({(per_img==k).mean()*100:4.1f}%)")
    print(f"  media: {per_img.mean():.2f} clases/imagen")

    # tamaño real de una imagen
    if train_dir is not None:
        sample_img = next(train_dir.glob("*.jpg"), None)
        if sample_img:
            w, h = Image.open(sample_img).size
            print(f"\nTamaño de imagen: {w}x{h} (esperado RLE shape {IMG_SHAPE[::-1]})")

    # ---- plots ----
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 4)

    ax = fig.add_subplot(gs[0, 0])
    ax.bar(CLASSES, [pres[c] for c in CLASSES], color="tab:blue")
    ax.set_title("Imágenes con cada clase"); ax.tick_params(axis="x", rotation=30)

    ax = fig.add_subplot(gs[0, 1])
    ax.bar(range(5), [(per_img == k).sum() for k in range(5)], color="tab:green")
    ax.set_title("Nº de clases por imagen"); ax.set_xlabel("clases presentes")

    # co-ocurrencia
    ax = fig.add_subplot(gs[0, 2])
    piv = (df.assign(v=df.has_mask.astype(int))
             .pivot_table(index="image", columns="label", values="v", fill_value=0)[CLASSES])
    co = piv.T.dot(piv)
    im = ax.imshow(co, cmap="Blues")
    ax.set_xticks(range(4)); ax.set_xticklabels(CLASSES, rotation=30)
    ax.set_yticks(range(4)); ax.set_yticklabels(CLASSES)
    ax.set_title("Co-ocurrencia de clases")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, int(co.iloc[i, j]), ha="center", va="center", fontsize=7)

    # muestras con máscaras superpuestas
    colors = {"Fish": (1, 0, 0), "Flower": (0, 1, 0), "Gravel": (0, 0, 1), "Sugar": (1, 1, 0)}
    if train_dir is not None:
        imgs_with = per_img[per_img > 0].index[:8]
        for idx, img_name in enumerate(imgs_with):
            r, c = 1 + idx // 4, idx % 4
            ax = fig.add_subplot(gs[r, c])
            img = np.array(Image.open(train_dir / img_name).convert("RGB").resize((525, 350)))
            overlay = img.copy()
            sub = df[(df.image == img_name) & df.has_mask]
            present = []
            for _, row in sub.iterrows():
                m = rle_decode(row["EncodedPixels"])
                m = np.array(Image.fromarray(m * 255).resize((525, 350))) > 127
                col = np.array(colors[row["label"]]) * 255
                overlay[m] = (0.5 * overlay[m] + 0.5 * col).astype(np.uint8)
                present.append(row["label"])
            ax.imshow(overlay); ax.axis("off")
            ax.set_title(" ".join(present), fontsize=7)

    fig.suptitle("Understanding Clouds — EDA (rojo=Fish verde=Flower azul=Gravel amarillo=Sugar)")
    fig.tight_layout()
    out = OUT_DIR / "eda.png"
    fig.savefig(out, dpi=100)
    print(f"\nPlot guardado en {out}")


if __name__ == "__main__":
    main()
