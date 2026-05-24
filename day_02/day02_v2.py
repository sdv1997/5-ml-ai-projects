"""
Día 2 / 30 — Conser-vision v2 — MegaDetector + ConvNeXt V2 Base
================================================================
Mejora clave sobre v1: MegaDetectorV5 pre-recorta el animal antes de clasificar.
Elimina el fondo (que puede ser específico del site) y fuerza al clasificador
a fijarse en la especie. Crítico dado que train/test tienen sites disjuntos.

Cambios respecto a v1:
  - MegaDetectorV5 → bbox del animal (cacheado en JSON, solo corre una vez)
  - Dataset recorta al bbox + padding antes de cualquier transform
  - Blank images (sin detección): imagen completa → coherente con la lógica
  - EPOCHS_UNFROZEN: 12 (vs 10 en v1)
  - Checkpoint y submission con sufijo _v2 para no pisar v1

Pipeline:
    1) MegaDetectorV5 detecta bbox en todas las imágenes → JSON cache.
    2) Dataset recorta al bbox antes de las transformaciones.
    3) ConvNeXt V2 Base (ImageNet-22K), StratifiedGroupKFold por site (5 folds).
    4) Fase 1: cabeza congelada (3 épocas). Fase 2: fine-tune diferencial (12 épocas).
    5) Ensemble OOF (media de probabilidades) → submission.

Uso:
    python day02_v2.py                # detecta (si no hay caché) + entrena
    python day02_v2.py --detect-only  # solo MegaDetector, sin entrenar
    python day02_v2.py --train-only   # solo entrena (necesita caché previo)
"""

import argparse
import io
import json
import random
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedGroupKFold
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).parent
ROOT = HERE.parent
DATA = ROOT / "data"
TRAIN_DIR = DATA / "train_features"
TEST_DIR  = DATA / "test_features"
SUBS_DIR  = HERE / "submissions"
SUBS_DIR.mkdir(exist_ok=True)
CKPT_DIR  = HERE / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)
BOXES_CACHE = CKPT_DIR / "megadetector_boxes.json"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 42
N_FOLDS = 5
IMG_SIZE = 224
BATCH_TRAIN_FROZEN   = 256   # solo cabeza, poco uso de VRAM
BATCH_TRAIN_UNFROZEN = 64    # backprop completo — 24 GB A5000 a 224px
BATCH_VAL = 128
NUM_WORKERS = 8
EPOCHS_FROZEN   = 3
EPOCHS_UNFROZEN = 12         # +2 vs v1
LR_HEAD     = 1e-3
LR_BACKBONE = 1e-5
LR_MIN      = 1e-7
WEIGHT_DECAY    = 1e-4
LABEL_SMOOTHING = 0.1
GRAD_CLIP = 1.0
BOX_CONF_THRESH = 0.2        # umbral de confianza MegaDetector
BOX_PAD = 0.1                # padding proporcional al tamaño del bbox
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = [
    "antelope_duiker",
    "bird",
    "blank",
    "civet_genet",
    "hog",
    "leopard",
    "monkey_prosimian",
    "rodent",
]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# MegaDetector
# ---------------------------------------------------------------------------

def _load_megadetector():
    try:
        from PytorchWildlife.models import detection as pw_detection
    except ImportError:
        import subprocess
        print("  PytorchWildlife no encontrado — instalando...")
        subprocess.run(["pip", "install", "PytorchWildlife", "-q"], check=True)
        from PytorchWildlife.models import detection as pw_detection
    detector = pw_detection.MegaDetectorV5(device=str(DEVICE), pretrained=True)
    return detector


def _parse_detections(det) -> list | None:
    """Extrae el bbox del animal de mayor confianza de un supervision.Detections."""
    if det is None:
        return None
    try:
        if len(det.xyxy) == 0:
            return None
        # Clase 0 = animal en MegaDetector (1=persona, 2=vehículo)
        animal_mask = det.class_id == 0
        if not animal_mask.any():
            return None
        conf = det.confidence[animal_mask]
        xyxy = det.xyxy[animal_mask]
        best = int(conf.argmax())
        return [float(v) for v in xyxy[best]]
    except Exception:
        return None


def run_megadetector() -> dict:
    """
    Corre MegaDetectorV5 en train_features/ y test_features/.
    Devuelve dict img_id → [x1, y1, x2, y2] (coords pixel) | None.
    Carga desde caché si existe. Procesa imagen a imagen para tolerar
    archivos corruptos y guarda el caché incrementalmente cada 500 imágenes.
    """
    if BOXES_CACHE.exists():
        print(f"  Cargando caché de bboxes ({BOXES_CACHE.name})")
        with open(BOXES_CACHE) as f:
            boxes = json.load(f)
        detected = sum(1 for v in boxes.values() if v is not None)
        print(f"  {len(boxes):,} imágenes — {detected:,} con animal ({detected/len(boxes):.1%})")
        return boxes

    print("  Cargando MegaDetectorV5...")
    detector = _load_megadetector()

    boxes = {}
    skipped = 0
    t0 = time.time()
    all_paths = []
    for img_dir in [TRAIN_DIR, TEST_DIR]:
        all_paths += sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.JPG"))

    print(f"  {len(all_paths):,} imágenes totales — procesando...")
    for i, path in enumerate(all_paths):
        img_id = path.stem
        try:
            res = detector.single_image_detection(
                str(path), det_conf_thres=BOX_CONF_THRESH
            )
            boxes[img_id] = _parse_detections(res.get("detections"))
        except Exception:
            boxes[img_id] = None
            skipped += 1

        if (i + 1) % 500 == 0:
            elapsed = (time.time() - t0) / 60
            eta = elapsed / (i + 1) * (len(all_paths) - i - 1)
            det_so_far = sum(1 for v in boxes.values() if v is not None)
            print(f"  {i+1:,}/{len(all_paths):,} ({(i+1)/len(all_paths):.0%}) "
                  f"| det={det_so_far:,} | skip={skipped} | {elapsed:.1f}min | ETA {eta:.1f}min")
            with open(BOXES_CACHE, "w") as f:
                json.dump(boxes, f)

    elapsed = (time.time() - t0) / 60
    detected = sum(1 for v in boxes.values() if v is not None)
    print(f"  Detecciones: {detected:,}/{len(boxes):,} ({detected/len(boxes):.1%}) — {elapsed:.1f} min")

    with open(BOXES_CACHE, "w") as f:
        json.dump(boxes, f)
    print(f"  Caché guardado → {BOXES_CACHE}")
    return boxes


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

TRAIN_TRANSFORMS = T.Compose([
    T.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(p=0.1),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
    T.RandomGrayscale(p=0.05),
    T.RandomRotation(degrees=10),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    T.RandomErasing(p=0.15, scale=(0.02, 0.1)),
])

VAL_TRANSFORMS = T.Compose([
    T.Resize(int(IMG_SIZE * 1.14)),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def crop_to_box(img: Image.Image, box: list) -> Image.Image:
    """Recorta la imagen al bbox detectado con padding proporcional."""
    w, h = img.size
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, x1 - bw * BOX_PAD)
    y1 = max(0, y1 - bh * BOX_PAD)
    x2 = min(w, x2 + bw * BOX_PAD)
    y2 = min(h, y2 + bh * BOX_PAD)
    cropped = img.crop((int(x1), int(y1), int(x2), int(y2)))
    if cropped.width < 10 or cropped.height < 10:
        return img  # fallback si el recorte es degenerado
    return cropped


def preload_images(df: pd.DataFrame, img_dir: Path) -> dict:
    cache = {}
    ids = df["id"].tolist()
    print(f"  Precargando {len(ids):,} imágenes en RAM...", end="", flush=True)
    for img_id in ids:
        for ext in [".jpg", ".JPG", ".jpeg"]:
            p = img_dir / f"{img_id}{ext}"
            if p.exists():
                try:
                    cache[img_id] = p.read_bytes()
                except Exception:
                    cache[img_id] = None
                break
    ok = sum(1 for v in cache.values() if v is not None)
    print(f" {ok:,} OK, {len(ids)-ok} fallidas")
    return cache


class WildlifeDataset(Dataset):
    def __init__(self, df, img_dir, transform, is_test=False, cache=None, boxes=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test
        self.cache = cache
        self.boxes = boxes

    def __len__(self):
        return len(self.df)

    def _load(self, img_id: str) -> Image.Image:
        if self.cache is not None:
            raw = self.cache.get(img_id)
            if raw:
                try:
                    return Image.open(io.BytesIO(raw)).convert("RGB")
                except Exception:
                    pass
            return Image.new("RGB", (IMG_SIZE, IMG_SIZE), 0)
        for ext in [".jpg", ".JPG", ".jpeg"]:
            p = self.img_dir / f"{img_id}{ext}"
            if p.exists():
                return Image.open(p).convert("RGB")
        return Image.new("RGB", (IMG_SIZE, IMG_SIZE), 0)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row["id"]
        img = self._load(img_id)

        if self.boxes is not None:
            box = self.boxes.get(img_id)
            if box is not None:
                img = crop_to_box(img, box)

        img = self.transform(img)
        if self.is_test:
            return img, img_id
        label = torch.tensor(row[CLASSES].values.astype(float), dtype=torch.float32)
        return img, label


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(n_classes: int = 8) -> nn.Module:
    return timm.create_model("convnextv2_base", pretrained=True, num_classes=n_classes)


def freeze_backbone(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = "head" in name


def unfreeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


def make_optimizer_frozen(model: nn.Module):
    return torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_HEAD, weight_decay=WEIGHT_DECAY,
    )


def make_optimizer_unfrozen(model: nn.Module):
    head_ids = {id(p) for p in model.head.parameters()}
    return torch.optim.AdamW([
        {"params": list(model.head.parameters()), "lr": LR_HEAD},
        {"params": [p for p in model.parameters() if id(p) not in head_ids], "lr": LR_BACKBONE},
    ], weight_decay=WEIGHT_DECAY)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scaler, criterion, device):
    model.train()
    total_loss = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with autocast():
            loss = criterion(model(imgs), labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * len(imgs)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, all_probs, all_labels = 0.0, [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with autocast():
            logits = model(imgs)
            loss = criterion(logits, labels)
        total_loss += loss.item() * len(imgs)
        all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    probs = np.concatenate(all_probs)
    true  = np.concatenate(all_labels)
    ll = log_loss(true.argmax(axis=1), probs, labels=list(range(len(CLASSES))))
    return total_loss / len(loader.dataset), ll, probs


def make_loader(ds, batch_size, shuffle):
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
    )


def train_fold(model, tr_ds, val_ds, criterion):
    scaler = GradScaler()
    best_logloss = float("inf")
    best_state = None

    # --- Fase 1: backbone congelado ---
    print(f"  [Fase 1] {EPOCHS_FROZEN} épocas — solo cabeza (batch={BATCH_TRAIN_FROZEN})")
    freeze_backbone(model)
    tr_loader  = make_loader(tr_ds, BATCH_TRAIN_FROZEN, shuffle=True)
    val_loader = make_loader(val_ds, BATCH_VAL, shuffle=False)
    optimizer  = make_optimizer_frozen(model)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS_FROZEN, eta_min=LR_MIN
    )
    for epoch in range(1, EPOCHS_FROZEN + 1):
        tr_loss = train_one_epoch(model, tr_loader, optimizer, scaler, criterion, DEVICE)
        _, val_ll, _ = validate(model, val_loader, criterion, DEVICE)
        scheduler.step()
        print(f"    E{epoch:02d}/{EPOCHS_FROZEN} | tr={tr_loss:.4f} | val_ll={val_ll:.4f}")
        if val_ll < best_logloss:
            best_logloss = val_ll
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    del tr_loader, val_loader
    torch.cuda.empty_cache()

    # --- Fase 2: fine-tune diferencial ---
    print(f"  [Fase 2] {EPOCHS_UNFROZEN} épocas — lr diferencial "
          f"(head={LR_HEAD}, backbone={LR_BACKBONE}, batch={BATCH_TRAIN_UNFROZEN})")
    unfreeze_all(model)
    tr_loader  = make_loader(tr_ds, BATCH_TRAIN_UNFROZEN, shuffle=True)
    val_loader = make_loader(val_ds, BATCH_VAL, shuffle=False)
    optimizer  = make_optimizer_unfrozen(model)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS_UNFROZEN, eta_min=LR_MIN
    )
    for epoch in range(1, EPOCHS_UNFROZEN + 1):
        tr_loss = train_one_epoch(model, tr_loader, optimizer, scaler, criterion, DEVICE)
        _, val_ll, _ = validate(model, val_loader, criterion, DEVICE)
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]
        print(f"    E{epoch:02d}/{EPOCHS_UNFROZEN} | tr={tr_loss:.4f} | val_ll={val_ll:.4f} | lr={lr_now:.2e}")
        if val_ll < best_logloss:
            best_logloss = val_ll
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, best_logloss


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(detect_only: bool = False, train_only: bool = False) -> None:
    seed_everything(SEED)
    print(f"\nDevice: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")

    # --- Detección con MegaDetector ---
    if not train_only:
        print("\n[MegaDetector V5]")
        boxes = run_megadetector()
    else:
        if not BOXES_CACHE.exists():
            raise FileNotFoundError(
                f"No hay caché de bboxes en {BOXES_CACHE}. "
                "Ejecuta sin --train-only primero."
            )
        with open(BOXES_CACHE) as f:
            boxes = json.load(f)
        detected = sum(1 for v in boxes.values() if v is not None)
        print(f"[Boxes] {len(boxes):,} imágenes, {detected:,} con animal — cargado desde caché")

    if detect_only:
        return

    # --- Datos ---
    train_df   = pd.read_csv(DATA / "train_labels.csv")
    train_meta = pd.read_csv(DATA / "train_features.csv")[["id", "site"]]
    train_df   = train_df.merge(train_meta, on="id", how="left")
    sub_fmt    = pd.read_csv(DATA / "submission_format.csv")
    test_meta  = pd.read_csv(DATA / "test_features.csv")[["id"]]

    print(f"\nTrain: {len(train_df):,} | Test: {len(test_meta):,}")
    print(f"Sites train: {train_df['site'].nunique()} únicos (CV site-disjoint)")
    det_train = sum(1 for i in train_df["id"] if boxes.get(i) is not None)
    det_test  = sum(1 for i in test_meta["id"] if boxes.get(i) is not None)
    print(f"Detecciones: train {det_train:,}/{len(train_df):,} ({det_train/len(train_df):.1%}) | "
          f"test {det_test:,}/{len(test_meta):,} ({det_test/len(test_meta):.1%})")

    # Detecciones por clase (train)
    label_col = train_df[CLASSES].idxmax(axis=1)
    print("\nDetecciones por clase:")
    for cls in CLASSES:
        mask = label_col == cls
        n = mask.sum()
        n_det = sum(1 for i in train_df.loc[mask, "id"] if boxes.get(i) is not None)
        print(f"  {cls:<22} {n_det:>5}/{n:<5} ({n_det/n:.0%})")

    print("\n[Precarga en RAM]")
    train_cache = preload_images(train_df, TRAIN_DIR)
    test_cache  = preload_images(test_meta, TEST_DIR)

    y     = train_df[CLASSES].values.argmax(axis=1)
    sites = train_df["site"].values
    sgkf  = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    oof_probs  = np.zeros((len(train_df), len(CLASSES)))
    test_probs = np.zeros((len(test_meta), len(CLASSES)))
    criterion  = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    fold_scores = []
    t0 = time.time()

    # --- Resume desde checkpoint si existe ---
    start_fold = 0
    scores_path = CKPT_DIR / "fold_scores_v2.npy"
    if scores_path.exists():
        saved_scores = np.load(scores_path).tolist()
        start_fold = len(saved_scores)
        if start_fold > 0:
            oof_probs   = np.load(CKPT_DIR / "oof_probs_v2.npy")
            test_probs  = np.load(CKPT_DIR / "test_probs_v2.npy")
            fold_scores = saved_scores
            print(f"\nResumiendo desde fold {start_fold+1} "
                  f"({start_fold} fold(s) completado(s), "
                  f"scores: {[f'{s:.4f}' for s in fold_scores]})")

    for fold, (tr_idx, val_idx) in enumerate(sgkf.split(train_df, y, groups=sites)):
        if fold < start_fold:
            continue
        print(f"\n{'='*60}")
        print(f"FOLD {fold+1}/{N_FOLDS}")
        print(f"{'='*60}")

        tr_df  = train_df.iloc[tr_idx]
        val_df = train_df.iloc[val_idx]

        tr_ds  = WildlifeDataset(tr_df,  TRAIN_DIR, TRAIN_TRANSFORMS,
                                 cache=train_cache, boxes=boxes)
        val_ds = WildlifeDataset(val_df, TRAIN_DIR, VAL_TRANSFORMS,
                                 cache=train_cache, boxes=boxes)

        model = build_model().to(DEVICE)
        model, _ = train_fold(model, tr_ds, val_ds, criterion)

        val_loader = make_loader(val_ds, BATCH_VAL, shuffle=False)
        _, fold_ll, oof_fold = validate(model, val_loader, criterion, DEVICE)
        oof_probs[val_idx] = oof_fold
        fold_scores.append(fold_ll)
        print(f"  → Fold {fold+1} log-loss: {fold_ll:.4f}")

        # Test inference
        te_ds = WildlifeDataset(test_meta, TEST_DIR, VAL_TRANSFORMS,
                                is_test=True, cache=test_cache, boxes=boxes)
        te_loader = DataLoader(te_ds, batch_size=BATCH_VAL, shuffle=False,
                               num_workers=NUM_WORKERS, pin_memory=True,
                               persistent_workers=True)
        fold_test = []
        model.eval()
        with torch.no_grad():
            for imgs, _ in te_loader:
                with autocast():
                    logits = model(imgs.to(DEVICE))
                fold_test.append(torch.softmax(logits, dim=1).cpu().numpy())
        test_probs += np.concatenate(fold_test) / N_FOLDS

        # Checkpoint por si se interrumpe
        np.save(CKPT_DIR / "oof_probs_v2.npy", oof_probs)
        np.save(CKPT_DIR / "test_probs_v2.npy", test_probs)
        np.save(CKPT_DIR / "fold_scores_v2.npy", np.array(fold_scores))
        torch.save(model.state_dict(), CKPT_DIR / f"model_v2_fold{fold+1}.pth")
        print(f"  → Checkpoint → {CKPT_DIR}/model_v2_fold{fold+1}.pth")

    cv_ll = log_loss(y, oof_probs, labels=list(range(len(CLASSES))))
    elapsed = (time.time() - t0) / 60
    print(f"\n{'='*60}")
    print(f"CV log-loss: {cv_ll:.4f}  (folds: {[f'{s:.4f}' for s in fold_scores]})")
    print(f"Tiempo total entrenamiento: {elapsed:.1f} min")
    print(f"{'='*60}")

    # Submission
    test_probs_df = pd.DataFrame(test_probs, columns=CLASSES)
    test_probs_df.insert(0, "id", test_meta["id"].values)
    sub = sub_fmt[["id"]].merge(test_probs_df, on="id", how="left")[["id"] + CLASSES]
    sub_path = SUBS_DIR / "submission_v2_megadetector.csv"
    sub.to_csv(sub_path, index=False)
    print(f"\nSubmission → {sub_path}")

    _plot_cv_summary(fold_scores, cv_ll)


def _plot_cv_summary(fold_scores: list, cv_mean: float) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    x = [f"Fold {i+1}" for i in range(len(fold_scores))]
    bars = ax.bar(x, fold_scores, color="steelblue", edgecolor="white")
    ax.axhline(cv_mean, color="tomato", linestyle="--", linewidth=1.5,
               label=f"CV mean = {cv_mean:.4f}")
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    ax.set_ylabel("Log-loss (↓ mejor)")
    ax.set_title("CV por fold — ConvNeXt V2 Base + MegaDetector", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    path = HERE / "cv_summary_v2.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Plot → {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--detect-only", action="store_true",
                        help="Solo corre MegaDetector y guarda el caché de bboxes")
    parser.add_argument("--train-only", action="store_true",
                        help="Solo entrena — requiere caché de bboxes previo")
    args = parser.parse_args()
    main(detect_only=args.detect_only, train_only=args.train_only)
