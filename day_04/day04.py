"""
BirdCLEF+ 2026 — Día 4
Task:  Identify 234 species (birds/frogs/insects/mammals) in 5-sec soundscape windows
Metric: macro-averaged ROC-AUC
Pipeline:
  1. Phase 1 — EfficientNet-B0 on 29k individual clips  (teacher)
  2. Pseudo-label — teacher labels 10.5k unlabeled soundscapes
  3. Phase 2 — fine-tune on gold soundscapes (66) + pseudo-labeled (~10.5k)
  4. Export ONNX for CPU inference on Kaggle (90-min limit)
"""

import os, random, ast, warnings, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torchaudio
import torchaudio.transforms as T
import timm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────

class Cfg:
    data_dir    = Path("/workspace/data/birdclef-2026")
    out_dir     = Path("/workspace/10-days-of-ai/day_04")

    # Audio
    sr          = 32_000
    duration    = 5
    n_samples   = sr * duration         # 160 000 samples

    # Mel
    n_mels      = 128
    n_fft       = 1024
    hop_length  = 320
    fmin        = 50
    fmax        = 14_000

    # Model
    backbone    = "efficientnet_b0"
    img_size    = 224
    drop_rate   = 0.3

    # Phase 1 — clips
    epochs_clip = 25
    bs_clip     = 64
    lr_clip     = 1e-3
    wd          = 1e-4

    # Phase 2 — soundscapes
    epochs_sc   = 10
    bs_sc       = 32
    lr_sc       = 3e-4

    # Pseudo-label threshold: keep windows where max_prob > this
    pseudo_thr  = 0.25
    # Weight of pseudo-labeled windows relative to gold (1.0)
    pseudo_w    = 0.5

    # CV
    n_folds     = 5
    seed        = 42

    # Quality filter
    min_rating  = 3.0

    # SpecAugment
    freq_mask   = 20
    time_mask   = 50

    num_workers = 8
    device      = "cuda" if torch.cuda.is_available() else "cpu"

cfg = Cfg()
cfg.out_dir.mkdir(parents=True, exist_ok=True)

def seed_everything(s=cfg.seed):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

seed_everything()

# ── Species ───────────────────────────────────────────────────────────────────

sub      = pd.read_csv(cfg.data_dir / "sample_submission.csv")
SPECIES  = list(sub.columns[1:])
S2I      = {s: i for i, s in enumerate(SPECIES)}
N        = len(SPECIES)   # 234

# ── Audio helpers ─────────────────────────────────────────────────────────────

def load_clip(path, augment=False):
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1: wav = wav.mean(0, keepdim=True)
    if sr != cfg.sr: wav = torchaudio.functional.resample(wav, sr, cfg.sr)
    n = cfg.n_samples
    if wav.shape[1] < n:
        wav = F.pad(wav, (0, n - wav.shape[1]))
    else:
        start = random.randint(0, wav.shape[1] - n) if augment else (wav.shape[1] - n) // 2
        wav = wav[:, start:start + n]
    return wav  # [1, n_samples]

def load_window(path, start_sec):
    frame = int(start_sec * cfg.sr)
    wav, sr = torchaudio.load(str(path), frame_offset=frame, num_frames=cfg.n_samples)
    if wav.shape[0] > 1: wav = wav.mean(0, keepdim=True)
    if sr != cfg.sr: wav = torchaudio.functional.resample(wav, sr, cfg.sr)
    if wav.shape[1] < cfg.n_samples:
        wav = F.pad(wav, (0, cfg.n_samples - wav.shape[1]))
    return wav

# ── Mel transform ─────────────────────────────────────────────────────────────

class MelSpec(nn.Module):
    def __init__(self, augment=False):
        super().__init__()
        self.mel  = T.MelSpectrogram(cfg.sr, cfg.n_fft, hop_length=cfg.hop_length,
                                     n_mels=cfg.n_mels, f_min=cfg.fmin, f_max=cfg.fmax)
        self.db   = T.AmplitudeToDB(top_db=80)
        self.fm   = T.FrequencyMasking(cfg.freq_mask) if augment else None
        self.tm   = T.TimeMasking(cfg.time_mask)      if augment else None

    def forward(self, wav):          # wav: [1, n_samples]
        s = self.db(self.mel(wav))   # [1, n_mels, T]
        s = (s - s.min()) / (s.max() - s.min() + 1e-8)
        if self.fm is not None: s = self.fm(s)
        if self.tm is not None: s = self.tm(s)
        s = s.repeat(3, 1, 1)       # [3, n_mels, T]
        s = F.interpolate(s.unsqueeze(0), (cfg.img_size, cfg.img_size),
                          mode="bilinear", align_corners=False).squeeze(0)
        return s                     # [3, H, W]

# ── Datasets ──────────────────────────────────────────────────────────────────

def parse_sec(s):
    parts = str(s).split(":")
    return int(parts[-1]) + int(parts[-2]) * 60 if len(parts) >= 2 else int(parts[-1])

def parse_secondary(v):
    if pd.isna(v) or v in ("[]", ""): return []
    try:    return [str(x) for x in ast.literal_eval(v)]
    except: return []


class ClipDataset(Dataset):
    def __init__(self, df, augment=False):
        self.df    = df.reset_index(drop=True)
        self.mel   = MelSpec(augment=augment)
        self.aug   = augment
        self.adir  = cfg.data_dir / "train_audio"

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:    wav = load_clip(self.adir / row["filename"], self.aug)
        except: wav = torch.zeros(1, cfg.n_samples)
        spec = self.mel(wav)

        label = torch.zeros(N)
        p = str(row["primary_label"])
        if p in S2I: label[S2I[p]] = 1.0
        for s in parse_secondary(row.get("secondary_labels", "[]")):
            if s in S2I: label[S2I[s]] = 0.5

        r = float(row.get("rating", 0))
        w = 0.5 if 0 < r < cfg.min_rating else 1.0
        return spec, label, torch.tensor(w)


class SoundscapeDataset(Dataset):
    """Handles both gold-labeled and pseudo-labeled soundscape windows."""
    def __init__(self, rows, augment=False):
        # rows: list of dicts {filename, start_sec, label_vec (np [N]), weight}
        self.rows  = rows
        self.mel   = MelSpec(augment=augment)
        self.scdir = cfg.data_dir / "train_soundscapes"

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        try:    wav = load_window(self.scdir / r["filename"], r["start_sec"])
        except: wav = torch.zeros(1, cfg.n_samples)
        spec  = self.mel(wav)
        label = torch.tensor(r["label_vec"], dtype=torch.float32)
        w     = torch.tensor(r["weight"],    dtype=torch.float32)
        return spec, label, w


def build_gold_rows():
    """Parse train_soundscapes_labels.csv → list of row dicts."""
    df   = pd.read_csv(cfg.data_dir / "train_soundscapes_labels.csv")
    rows = []
    for _, r in df.iterrows():
        label_vec = np.zeros(N, dtype=np.float32)
        for s in str(r["primary_label"]).split(";"):
            s = s.strip()
            if s in S2I: label_vec[S2I[s]] = 1.0
        rows.append({"filename": r["filename"],
                     "start_sec": parse_sec(r["start"]),
                     "label_vec": label_vec,
                     "weight": 1.0})
    return rows

# ── Model ─────────────────────────────────────────────────────────────────────

class BirdModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(cfg.backbone, pretrained=True,
                                          num_classes=0, in_chans=3)
        nf = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Linear(nf, 512), nn.ReLU(), nn.Dropout(cfg.drop_rate),
            nn.Linear(512, N),
        )

    def forward(self, x): return self.head(self.backbone(x))

# ── Mixup ─────────────────────────────────────────────────────────────────────

def mixup(x, y, w, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    return (lam*x + (1-lam)*x[idx],
            lam*y + (1-lam)*y[idx],
            lam*w + (1-lam)*w[idx])

# ── Train / validate ──────────────────────────────────────────────────────────

criterion = nn.BCEWithLogitsLoss(reduction="none")

def train_epoch(model, loader, opt, scaler, use_mixup=True):
    model.train()
    losses = []
    for x, y, w in loader:
        x, y, w = x.to(cfg.device), y.to(cfg.device), w.to(cfg.device).view(-1, 1)
        if use_mixup: x, y, w = mixup(x, y, w)
        opt.zero_grad()
        with autocast():
            loss = (criterion(model(x), y) * w).mean()
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        losses.append(loss.item())
    return float(np.mean(losses))

@torch.no_grad()
def validate(model, loader):
    model.eval()
    preds, labs = [], []
    for x, y, _ in loader:
        with autocast(): logits = model(x.to(cfg.device))
        preds.append(torch.sigmoid(logits).cpu().float().numpy())
        labs.append(y.numpy())
    preds = np.vstack(preds); labs = np.vstack(labs)
    aucs  = []
    for i in range(N):
        if labs[:, i].sum() > 0:
            try: aucs.append(roc_auc_score(labs[:, i], preds[:, i]))
            except: pass
    return float(np.mean(aucs)) if aucs else 0.0

# ── Pseudo-label generation ───────────────────────────────────────────────────

@torch.no_grad()
def generate_pseudo_labels(model, labeled_files):
    """
    Run teacher model on all unlabeled soundscapes.
    Returns list of row dicts ready for SoundscapeDataset.
    Each 60-second soundscape → 12 windows of 5 seconds.
    Only keeps windows where max predicted prob > cfg.pseudo_thr.
    """
    model.eval()
    mel_fn   = MelSpec(augment=False).to(cfg.device)
    sc_dir   = cfg.data_dir / "train_soundscapes"
    all_sc   = sorted(sc_dir.glob("*.ogg"))
    unlabeled = [p for p in all_sc if p.name not in labeled_files]

    print(f"  Pseudo-labeling {len(unlabeled)} soundscapes …")
    rows = []
    for path in tqdm(unlabeled, desc="  pseudo-label"):
        try:
            wav_full, sr = torchaudio.load(str(path))
            if wav_full.shape[0] > 1: wav_full = wav_full.mean(0, keepdim=True)
            if sr != cfg.sr:
                wav_full = torchaudio.functional.resample(wav_full, sr, cfg.sr)
        except Exception:
            continue

        # Build batch of 12 × 5-sec windows
        windows, starts = [], []
        n = cfg.n_samples
        total = wav_full.shape[1]
        for start in range(0, 60 * cfg.sr, n):   # 0, 5s, 10s, … 55s
            if start + n > total:
                chunk = F.pad(wav_full[:, start:], (0, start + n - total))
            else:
                chunk = wav_full[:, start:start + n]
            windows.append(chunk)
            starts.append(start // cfg.sr)

        batch = torch.stack([mel_fn(w.to(cfg.device)) for w in windows])  # [12, 3, H, W]
        with autocast():
            probs = torch.sigmoid(model(batch)).cpu().float().numpy()      # [12, N]

        for prob_vec, start_sec in zip(probs, starts):
            if prob_vec.max() >= cfg.pseudo_thr:
                rows.append({"filename": path.name,
                             "start_sec": start_sec,
                             "label_vec": prob_vec.astype(np.float32),
                             "weight": cfg.pseudo_w})
    print(f"  → {len(rows)} pseudo-labeled windows (thr={cfg.pseudo_thr})")
    return rows

# ── Checkpointing ────────────────────────────────────────────────────────────

def save_ckpt(path, epoch, phase, model, opt, sched, best_auc, best_state, **extra):
    torch.save({
        "epoch":      epoch,
        "phase":      phase,
        "model":      model.state_dict(),
        "opt":        opt.state_dict(),
        "sched":      sched.state_dict(),
        "best_auc":   best_auc,
        "best_state": best_state,
        **extra,
    }, path)

def load_ckpt(path, model, opt, sched):
    ck = torch.load(path, map_location=cfg.device)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    sched.load_state_dict(ck["sched"])
    return ck["epoch"], ck["phase"], ck["best_auc"], ck["best_state"]

# ── ONNX export ───────────────────────────────────────────────────────────────

def export_onnx(model, path):
    model.eval()
    dummy = torch.randn(1, 3, cfg.img_size, cfg.img_size, device=cfg.device)
    torch.onnx.export(model, dummy, str(path),
                      input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=17)
    print(f"ONNX → {path}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    # ── Load clips ──
    train_df = pd.read_csv(cfg.data_dir / "train.csv")
    train_df["primary_label"] = train_df["primary_label"].astype(str)
    train_df = train_df[train_df["filename"].apply(
        lambda f: (cfg.data_dir / "train_audio" / f).exists()
    )].reset_index(drop=True)
    print(f"Clips: {len(train_df)} | Especies: {train_df['primary_label'].nunique()} | Clases: {N}", flush=True)

    # ── CV split ──
    skf = StratifiedKFold(cfg.n_folds, shuffle=True, random_state=cfg.seed)
    tr_idx, va_idx = list(skf.split(train_df, train_df["primary_label"]))[args.fold]
    tr_df, va_df   = train_df.iloc[tr_idx], train_df.iloc[va_idx]
    print(f"Fold {args.fold}: train={len(tr_df)} val={len(va_df)}", flush=True)

    tr_dl = DataLoader(ClipDataset(tr_df, augment=True),
                       batch_size=cfg.bs_clip, shuffle=True,
                       num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    va_dl = DataLoader(ClipDataset(va_df, augment=False),
                       batch_size=cfg.bs_clip*2, shuffle=False,
                       num_workers=cfg.num_workers, pin_memory=True)

    model   = BirdModel().to(cfg.device)
    scaler  = GradScaler()
    ckpt    = cfg.out_dir / f"model_fold{args.fold}.pt"      # Phase 1 best weights
    sc_ckpt = cfg.out_dir / f"model_fold{args.fold}_sc.pt"   # Phase 2 best weights
    resume  = cfg.out_dir / f"resume_fold{args.fold}.pt"     # Resumable state (overwritten each epoch)

    if args.eval_only:
        model.load_state_dict(torch.load(ckpt, map_location=cfg.device))
        print(f"Val AUC: {validate(model, va_dl):.4f}")
        return

    # ── Detect resume state ──
    cur_phase, cur_epoch = 1, 0
    if resume.exists():
        meta      = torch.load(resume, map_location="cpu")
        cur_phase = meta["phase"]
        cur_epoch = meta["epoch"]
        print(f"▶ Checkpoint detectado: phase={cur_phase}, epoch={cur_epoch}", flush=True)

    # ═══════════════════════════════════════════════════════════
    # PHASE 1 — clips (teacher training)
    # ═══════════════════════════════════════════════════════════
    best_auc, best_state = 0.0, None

    if cur_phase == 1:
        opt   = torch.optim.AdamW(model.parameters(), lr=cfg.lr_clip, weight_decay=cfg.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs_clip, eta_min=1e-6)

        if cur_epoch > 0:
            _, _, best_auc, best_state = load_ckpt(resume, model, opt, sched)
            print(f"  Resumiendo Phase 1 ep {cur_epoch+1}→{cfg.epochs_clip}  best_auc={best_auc:.4f}", flush=True)

        print(f"\n{'═'*60}")
        print(f"PHASE 1 — Clips  (ep {cur_epoch+1}→{cfg.epochs_clip}, bs={cfg.bs_clip})")
        print(f"{'═'*60}", flush=True)

        for ep in range(cur_epoch + 1, cfg.epochs_clip + 1):
            loss = train_epoch(model, tr_dl, opt, scaler, use_mixup=True)
            auc  = validate(model, va_dl)
            sched.step()
            mark = " ◀ best" if auc > best_auc else ""
            print(f"  ep {ep:02d}/{cfg.epochs_clip}  loss={loss:.4f}  val_auc={auc:.4f}"
                  f"  lr={opt.param_groups[0]['lr']:.1e}{mark}", flush=True)
            if auc > best_auc:
                best_auc   = auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            save_ckpt(resume, ep, 1, model, opt, sched, best_auc, best_state)

        model.load_state_dict({k: v.to(cfg.device) for k, v in best_state.items()})
        torch.save(best_state, ckpt)
        print(f"\nPhase 1 best AUC: {best_auc:.4f}  →  {ckpt}", flush=True)
        # Advance resume to phase 2 (carry phase1_auc forward for the final summary)
        save_ckpt(resume, 0, 2, model, opt, sched, 0.0, None, phase1_auc=best_auc)
        cur_phase, cur_epoch = 2, 0

    else:
        # Phase 1 already done — load its best weights and retrieve its AUC
        print(f"▶ Phase 1 ya completada — cargando {ckpt}", flush=True)
        model.load_state_dict(torch.load(ckpt, map_location=cfg.device))
        meta     = torch.load(resume, map_location="cpu")
        best_auc = meta.get("phase1_auc", meta["best_auc"])

    # ═══════════════════════════════════════════════════════════
    # PSEUDO-LABELING — teacher labels unlabeled soundscapes
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'═'*60}")
    print("PSEUDO-LABELING")
    print(f"{'═'*60}", flush=True)

    gold_rows   = build_gold_rows()
    labeled_set = {r["filename"] for r in gold_rows}

    pseudo_cache = cfg.out_dir / f"pseudo_fold{args.fold}.npz"
    if pseudo_cache.exists():
        print(f"  Cargando pseudo-labels de caché: {pseudo_cache}", flush=True)
        data        = np.load(pseudo_cache, allow_pickle=True)
        pseudo_rows = list(data["rows"])
    else:
        pseudo_rows = generate_pseudo_labels(model, labeled_set)
        np.savez(str(pseudo_cache), rows=np.array(pseudo_rows, dtype=object))
        print(f"  Guardado en {pseudo_cache}", flush=True)

    all_sc_rows = gold_rows + pseudo_rows
    print(f"  Total soundscape windows: {len(all_sc_rows)} "
          f"(gold={len(gold_rows)}, pseudo={len(pseudo_rows)})", flush=True)

    # ═══════════════════════════════════════════════════════════
    # PHASE 2 — fine-tune on gold + pseudo soundscapes
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'═'*60}")
    print(f"PHASE 2 — Soundscapes  ({cfg.epochs_sc} epochs, bs={cfg.bs_sc})")
    print(f"{'═'*60}", flush=True)

    rng        = np.random.RandomState(cfg.seed)
    gold_files = list({r["filename"] for r in gold_rows})
    val_files  = set(rng.choice(gold_files, size=max(1, len(gold_files)//5), replace=False))

    sc_tr_rows = [r for r in all_sc_rows if r["filename"] not in val_files]
    sc_va_rows = [r for r in gold_rows   if r["filename"] in val_files]

    sc_tr_dl = DataLoader(SoundscapeDataset(sc_tr_rows, augment=True),
                          batch_size=cfg.bs_sc, shuffle=True,
                          num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    sc_va_dl = DataLoader(SoundscapeDataset(sc_va_rows, augment=False),
                          batch_size=cfg.bs_sc*2, shuffle=False,
                          num_workers=cfg.num_workers, pin_memory=True)

    opt2   = torch.optim.AdamW(model.parameters(), lr=cfg.lr_sc, weight_decay=cfg.wd)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, cfg.epochs_sc, eta_min=1e-6)
    best_auc2, best_state2 = 0.0, None

    if cur_epoch > 0:
        _, _, best_auc2, best_state2 = load_ckpt(resume, model, opt2, sched2)
        print(f"  Resumiendo Phase 2 ep {cur_epoch+1}→{cfg.epochs_sc}  best_auc2={best_auc2:.4f}", flush=True)

    for ep in range(cur_epoch + 1, cfg.epochs_sc + 1):
        loss = train_epoch(model, sc_tr_dl, opt2, scaler, use_mixup=False)
        auc  = validate(model, sc_va_dl)
        sched2.step()
        mark = " ◀ best" if auc > best_auc2 else ""
        print(f"  ep {ep:02d}/{cfg.epochs_sc}  loss={loss:.4f}  sc_val_auc={auc:.4f}{mark}", flush=True)
        if auc > best_auc2:
            best_auc2   = auc
            best_state2 = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        save_ckpt(resume, ep, 2, model, opt2, sched2, best_auc2, best_state2, phase1_auc=best_auc)

    model.load_state_dict({k: v.to(cfg.device) for k, v in best_state2.items()})
    torch.save(best_state2, sc_ckpt)
    print(f"\nPhase 2 best AUC: {best_auc2:.4f}  →  {sc_ckpt}", flush=True)

    # ── ONNX export ──
    export_onnx(model, cfg.out_dir / f"model_fold{args.fold}.onnx")

    print(f"\n{'═'*60}")
    print(f"✓ Fold {args.fold} complete")
    print(f"  Phase 1 AUC : {best_auc:.4f}  (clips, {len(tr_df)} samples)")
    print(f"  Phase 2 AUC : {best_auc2:.4f}  (soundscapes, {len(sc_tr_rows)} windows)")
    print(f"{'═'*60}", flush=True)


if __name__ == "__main__":
    main()
