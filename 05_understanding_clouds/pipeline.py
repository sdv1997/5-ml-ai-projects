"""
Proyecto 5 — Understanding Clouds: baseline U-Net (entrenar en RunPod A5000)
===========================================================================

Segmentación multi-label de 4 patrones de nube (Fish/Flower/Gravel/Sugar).
Decisiones (justificadas en el EDA): las máscaras son bloques rectangulares
gruesos → reescalado agresivo (384×576) sin perder señal; lo que importa es
localizar el patrón y decidir si está → post-proceso (umbral + min-size) clave.

Pipeline:
  1) Dataset: jpg → resize 384×576, normaliza (ImageNet); máscaras vía RLE → resize.
  2) Modelo: smp.Unet(encoder resnet34, pretrained), 4 canales, salida sigmoid.
  3) Loss: BCE + Dice. Optim AdamW + CosineAnnealing, AMP.
  4) Validación: Dice OFICIAL a resolución original (sube la pred a 1400×2100),
     con búsqueda de umbral + min-size por clase.
  5) Inferencia test → post-proceso → RLE → submission.csv.

Uso (en el pod):
    python 05_understanding_clouds/pipeline.py --train      # entrena, guarda best.pt
    python 05_understanding_clouds/pipeline.py --predict    # genera submission.csv
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rle import rle_decode, rle_encode, dice, CLASSES, IMG_SHAPE  # IMG_SHAPE=(1400,2100)

OUT_DIR = Path(__file__).resolve().parent
DATA_DIR = OUT_DIR / "data"
CKPT = OUT_DIR / "best.pt"
SUB = OUT_DIR / "submissions"

H, W = 384, 576                      # resolución de trabajo (mantiene aspecto 2:3)
ENCODER = "resnet34"
BATCH = 16
EPOCHS = 24
LR = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MEAN = np.array([0.485, 0.456, 0.406]); STD = np.array([0.229, 0.224, 0.225])
SEED = 42


def load_df():
    df = pd.read_csv(DATA_DIR / "train.csv")
    df["image"] = df["Image_Label"].str.split("_").str[0]
    df["label"] = df["Image_Label"].str.split("_").str[1]
    return df


class CloudDS(Dataset):
    def __init__(self, images, df, img_dir, train=True):
        self.images = list(images); self.img_dir = img_dir; self.train = train
        self.by_img = {img: g for img, g in df.groupby("image")} if df is not None else {}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        name = self.images[i]
        img = cv2.imread(str(self.img_dir / name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
        if self.train:
            masks = np.zeros((len(CLASSES), H, W), dtype=np.float32)
            g = self.by_img.get(name)
            if g is not None:
                for _, row in g.iterrows():
                    if isinstance(row["EncodedPixels"], str) and row["EncodedPixels"].strip():
                        m = rle_decode(row["EncodedPixels"])  # (1400,2100)
                        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                        masks[CLASSES.index(row["label"])] = m
            if np.random.rand() < 0.5:
                img = img[:, ::-1].copy(); masks = masks[:, :, ::-1].copy()
            if np.random.rand() < 0.5:
                img = img[::-1].copy(); masks = masks[:, ::-1].copy()
            x = ((img / 255.0 - MEAN) / STD).transpose(2, 0, 1).astype(np.float32)
            return torch.from_numpy(x), torch.from_numpy(masks)
        x = ((img / 255.0 - MEAN) / STD).transpose(2, 0, 1).astype(np.float32)
        return torch.from_numpy(x), name


def build_model():
    return smp.Unet(encoder_name=ENCODER, encoder_weights="imagenet",
                    classes=len(CLASSES), activation=None).to(DEVICE)


def gt_masks_fullres(name, by_img):
    """Máscaras ground-truth a resolución original (para Dice oficial)."""
    out = []
    g = by_img.get(name)
    for c in CLASSES:
        m = np.zeros(IMG_SHAPE, dtype=np.uint8)
        if g is not None:
            row = g[g.label == c]
            if len(row) and isinstance(row.iloc[0]["EncodedPixels"], str) and row.iloc[0]["EncodedPixels"].strip():
                m = rle_decode(row.iloc[0]["EncodedPixels"])
        out.append(m)
    return out


def postprocess(prob_hw, thr, min_size):
    """prob (H,W) en [0,1] → sube a original, umbral, elimina si < min_size."""
    up = cv2.resize(prob_hw, (IMG_SHAPE[1], IMG_SHAPE[0]), interpolation=cv2.INTER_LINEAR)
    mask = (up > thr).astype(np.uint8)
    if mask.sum() < min_size:
        return np.zeros_like(mask)
    return mask


def train():
    torch.manual_seed(SEED); np.random.seed(SEED)
    df = load_df()
    images = df["image"].unique()
    rng = np.random.RandomState(SEED); rng.shuffle(images)
    n_val = int(len(images) * 0.15)
    val_imgs, tr_imgs = images[:n_val], images[n_val:]
    by_img = {img: g for img, g in df.groupby("image")}

    tl = DataLoader(CloudDS(tr_imgs, df, DATA_DIR / "train_images", True),
                    batch_size=BATCH, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    vl = DataLoader(CloudDS(val_imgs, None, DATA_DIR / "train_images", False),
                    batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)
    print(f"train {len(tr_imgs)} | val {len(val_imgs)} imágenes | device {DEVICE}", flush=True)

    model = build_model()
    bce = nn.BCEWithLogitsLoss(); dice_loss = smp.losses.DiceLoss(mode="multilabel")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler()

    best = -1
    for ep in range(EPOCHS):
        model.train()
        run = 0.0
        for x, y in tl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                out = model(x); loss = bce(out, y) + dice_loss(out, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            run += loss.item()
        sched.step()

        # validación: Dice oficial a resolución original (thr=0.5, min_size=15000 de partida)
        model.eval(); preds, gts = [], []
        with torch.no_grad():
            for x, names in vl:
                with torch.cuda.amp.autocast():
                    p = torch.sigmoid(model(x.to(DEVICE))).cpu().numpy()
                for b, name in enumerate(names):
                    gt = gt_masks_fullres(name, by_img)
                    for ci in range(len(CLASSES)):
                        preds.append(postprocess(p[b, ci], 0.5, 15000)); gts.append(gt[ci])
        vdice = float(np.mean([dice(a, b) for a, b in zip(preds, gts)]))
        print(f"ep {ep+1:02d}/{EPOCHS}  loss {run/len(tl):.4f}  val_dice {vdice:.4f}", flush=True)
        if vdice > best:
            best = vdice; torch.save(model.state_dict(), CKPT)
            print(f"  ✓ guardado best.pt (dice {best:.4f})", flush=True)
    print(f"Mejor val_dice: {best:.4f}")


@torch.no_grad()
def predict(thr=0.5, min_size=15000):
    model = build_model(); model.load_state_dict(torch.load(CKPT, map_location=DEVICE)); model.eval()
    test_imgs = sorted(p.name for p in (DATA_DIR / "test_images").glob("*.jpg"))
    dl = DataLoader(CloudDS(test_imgs, None, DATA_DIR / "test_images", False),
                    batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)
    rows = []
    for x, names in dl:
        with torch.cuda.amp.autocast():
            p = torch.sigmoid(model(x.to(DEVICE))).cpu().numpy()
        for b, name in enumerate(names):
            for ci, c in enumerate(CLASSES):
                mask = postprocess(p[b, ci], thr, min_size)
                rows.append({"Image_Label": f"{name}_{c}", "EncodedPixels": rle_encode(mask)})
    sub = pd.DataFrame(rows)
    SUB.mkdir(parents=True, exist_ok=True)
    sub.to_csv(SUB / "submission.csv", index=False)
    print(f"Submission escrita: {SUB/'submission.csv'}  ({len(sub)} filas, "
          f"{(sub.EncodedPixels!='').mean()*100:.1f}% con máscara)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--predict", action="store_true")
    a = ap.parse_args()
    if a.train: train()
    if a.predict: predict()
    if not (a.train or a.predict):
        print("Usa --train y/o --predict")
