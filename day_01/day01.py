"""
Día 1 / 30 — Richter's Predictor
=================================

EDA mínimo + 2 baselines:
    1) Predecir siempre la clase mayoritaria (sanity check).
    2) LightGBM out-of-the-box con encoding categórico nativo.

Genera:
    - class_distribution.png
    - submissions/baseline_majority.csv
    - submissions/baseline_lgbm.csv
    - log de scores en consola
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

HERE = Path(__file__).parent
DATA = HERE.parent / "data"
SUBS = HERE / "submissions"
SUBS.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Cargar datos
# ---------------------------------------------------------------------------
X = pd.read_csv(DATA / "train_values.csv")
y = pd.read_csv(DATA / "train_labels.csv")["damage_grade"]
X_test = pd.read_csv(DATA / "test_values.csv")
sub_fmt = pd.read_csv(DATA / "submission_format.csv")

print(f"train_values: {X.shape}")
print(f"test_values:  {X_test.shape}")
print(f"target:       {y.shape}\n")

class_dist = y.value_counts(normalize=True).sort_index()
print("Distribución de damage_grade:")
for cls, pct in class_dist.items():
    print(f"  {cls}: {pct:.2%}")
print()

# ---------------------------------------------------------------------------
# Plot: distribución de clases
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
counts = y.value_counts().sort_index()
colors = ["#2ecc71", "#f39c12", "#e74c3c"]
labels = ["1\nDaño bajo", "2\nDaño medio", "3\nDestrucción\ncasi total"]
bars = ax.bar(labels, counts.values, color=colors,
              edgecolor="black", linewidth=0.6)

for bar, count, pct in zip(bars, counts.values, class_dist.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + counts.max() * 0.015,
        f"{count:,}\n({pct:.1%})",
        ha="center", va="bottom", fontsize=11, fontweight="bold",
    )

ax.set_ylabel("Número de edificios", fontsize=12)
ax.set_title(
    f"Distribución de daño · Terremoto de Gorkha (Nepal, 2015) · n = {len(y):,}",
    fontsize=13, pad=15,
)
ax.set_ylim(0, counts.max() * 1.18)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plot_path = HERE / "class_distribution.png"
plt.savefig(plot_path, dpi=140, bbox_inches="tight")
print(f"Plot guardado: {plot_path.name}\n")

# ---------------------------------------------------------------------------
# Baseline 1: predecir siempre la clase mayoritaria
# ---------------------------------------------------------------------------
print("=" * 55)
print("Baseline 1 — Predecir la clase mayoritaria")
print("=" * 55)
majority = int(y.value_counts().idxmax())
pred_majority = np.full(len(y), majority)
f1_majority = f1_score(y, pred_majority, average="micro")
print(f"Clase mayoritaria: {majority}")
print(f"F1 micro (sobre train completo): {f1_majority:.4f}\n")

sub_majority = sub_fmt.copy()
sub_majority["damage_grade"] = majority
sub_majority.to_csv(SUBS / "baseline_majority.csv", index=False)
print(f"Submission guardado: submissions/baseline_majority.csv\n")

# ---------------------------------------------------------------------------
# Baseline 2: LightGBM out-of-the-box, 5-fold StratifiedKFold
# ---------------------------------------------------------------------------
print("=" * 55)
print("Baseline 2 — LightGBM out-of-the-box (5-fold CV)")
print("=" * 55)

feature_cols = [c for c in X.columns if c != "building_id"]
cat_cols = X[feature_cols].select_dtypes(include="object").columns.tolist()
print(f"Features: {len(feature_cols)} ({len(cat_cols)} categóricas)\n")

X_lgb = X[feature_cols].copy()
X_test_lgb = X_test[feature_cols].copy()
for col in cat_cols:
    X_lgb[col] = X_lgb[col].astype("category")
    X_test_lgb[col] = X_test_lgb[col].astype("category")

# LightGBM espera labels 0-indexed para multiclase
y_lgb = y - 1

params = dict(
    n_estimators=300,
    learning_rate=0.05,
    objective="multiclass",
    num_class=3,
    random_state=42,
    verbose=-1,
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
for fold, (tr_idx, val_idx) in enumerate(skf.split(X_lgb, y_lgb), 1):
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_lgb.iloc[tr_idx], y_lgb.iloc[tr_idx],
        categorical_feature=cat_cols,
    )
    pred = model.predict(X_lgb.iloc[val_idx])
    f1 = f1_score(y_lgb.iloc[val_idx], pred, average="micro")
    print(f"  Fold {fold}: F1 micro = {f1:.4f}")
    cv_scores.append(f1)

cv_mean = float(np.mean(cv_scores))
cv_std = float(np.std(cv_scores))
print(f"\nCV F1 micro (mean ± std): {cv_mean:.4f} ± {cv_std:.4f}\n")

# Entrenar en train completo y predecir test
print("Entrenando en train completo y prediciendo test...")
model = lgb.LGBMClassifier(**params)
model.fit(X_lgb, y_lgb, categorical_feature=cat_cols)
test_pred = model.predict(X_test_lgb) + 1  # volver a 1-indexed

sub_lgbm = sub_fmt.copy()
sub_lgbm["damage_grade"] = test_pred
sub_lgbm.to_csv(SUBS / "baseline_lgbm.csv", index=False)
print(f"Submission guardado: submissions/baseline_lgbm.csv\n")

# ---------------------------------------------------------------------------
# Resumen final
# ---------------------------------------------------------------------------
print("=" * 55)
print("Resumen — Día 1")
print("=" * 55)
print(f"Baseline mayoritaria (train completo): F1 micro = {f1_majority:.4f}")
print(f"Baseline LightGBM (5-fold CV mean):    F1 micro = {cv_mean:.4f} ± {cv_std:.4f}")
print(f"Mejora vs trivial:                     +{cv_mean - f1_majority:.4f}")
