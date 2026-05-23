"""
Día 1 (iteración 2) — Forzando el score
========================================

Cadena de experimentos sobre el baseline (LightGBM, F1 micro = 0.7179):

    E1  parity check — mismos splits, mismos params que ayer (verificación)
    E2  + cast geo_level_{1,2,3}_id a category (el cambio más obvio: 11k aldeas
        no son ordinales)
    E3  + más árboles con early stopping (2000 max, paciencia 50)
    E4  + hyperparams algo más serios (num_leaves, min_data_in_leaf, feature_fraction)

Mismos folds (StratifiedKFold seed=42) en todos los experimentos → diffs comparables.
Submission generada para el ganador.
"""

import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

HERE = Path(__file__).parent
DATA = HERE.parent / "data"
SUBS = HERE / "submissions"
SUBS.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
X = pd.read_csv(DATA / "train_values.csv")
y = pd.read_csv(DATA / "train_labels.csv")["damage_grade"]
X_test = pd.read_csv(DATA / "test_values.csv")
sub_fmt = pd.read_csv(DATA / "submission_format.csv")

feature_cols = [c for c in X.columns if c != "building_id"]
text_cat_cols = X[feature_cols].select_dtypes(include="object").columns.tolist()
geo_cols = ["geo_level_1_id", "geo_level_2_id", "geo_level_3_id"]

y_lgb = y - 1  # LightGBM espera 0-indexed
N_SPLITS = 5
SEED = 42


def prepare(X_in, X_test_in, cast_geo=False):
    X_out = X_in[feature_cols].copy()
    X_test_out = X_test_in[feature_cols].copy()
    cat_cols = list(text_cat_cols)
    if cast_geo:
        cat_cols = cat_cols + geo_cols
    for col in cat_cols:
        X_out[col] = X_out[col].astype("category")
        X_test_out[col] = X_test_out[col].astype("category")
    return X_out, X_test_out, cat_cols


def run_cv(X_in, y_in, cat_cols, params, use_early_stopping=False, label=""):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    fold_scores = []
    best_iters = []
    t0 = time.time()
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_in, y_in), 1):
        model = lgb.LGBMClassifier(**params)
        if use_early_stopping:
            model.fit(
                X_in.iloc[tr_idx], y_in.iloc[tr_idx],
                eval_set=[(X_in.iloc[val_idx], y_in.iloc[val_idx])],
                eval_metric="multi_logloss",
                categorical_feature=cat_cols,
                callbacks=[lgb.early_stopping(50, verbose=False),
                           lgb.log_evaluation(0)],
            )
            best_iters.append(model.best_iteration_)
        else:
            model.fit(X_in.iloc[tr_idx], y_in.iloc[tr_idx],
                      categorical_feature=cat_cols)
        pred = model.predict(X_in.iloc[val_idx])
        f1 = f1_score(y_in.iloc[val_idx], pred, average="micro")
        fold_scores.append(f1)
    elapsed = time.time() - t0
    mean = float(np.mean(fold_scores))
    std = float(np.std(fold_scores))
    extra = ""
    if best_iters:
        extra = f"  best_iter mean={int(np.mean(best_iters))}"
    print(f"  {label:<60} F1={mean:.4f} ±{std:.4f}  ({elapsed:.1f}s){extra}")
    return mean, std, best_iters


results = []

# ---------------------------------------------------------------------------
# E1 — parity check (mismo setup que día 1)
# ---------------------------------------------------------------------------
print("=" * 72)
print("E1 — Parity check del Día 1")
print("=" * 72)
X_e1, X_test_e1, cats_e1 = prepare(X, X_test, cast_geo=False)
params_e1 = dict(n_estimators=300, learning_rate=0.05,
                 objective="multiclass", num_class=3,
                 random_state=SEED, verbose=-1)
m, s, _ = run_cv(X_e1, y_lgb, cats_e1, params_e1, label="baseline (8 cats, geos como int)")
results.append(("E1: baseline (geos como int)", m, s))

# ---------------------------------------------------------------------------
# E2 — cast geo_level_*_id a category
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("E2 — Cast geo_level_{1,2,3}_id a category (11 cats totales)")
print("=" * 72)
X_e2, X_test_e2, cats_e2 = prepare(X, X_test, cast_geo=True)
params_e2 = dict(params_e1)
m, s, _ = run_cv(X_e2, y_lgb, cats_e2, params_e2, label="+ geos categóricas, n_est=300")
results.append(("E2: + geos como category", m, s))

# ---------------------------------------------------------------------------
# E3 — más árboles con early stopping
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("E3 — Más árboles con early stopping (n_est=2000, paciencia=50)")
print("=" * 72)
params_e3 = dict(n_estimators=2000, learning_rate=0.05,
                 objective="multiclass", num_class=3,
                 random_state=SEED, verbose=-1)
m, s, biters_e3 = run_cv(X_e2, y_lgb, cats_e2, params_e3,
                         use_early_stopping=True,
                         label="+ early stopping (lr=0.05)")
results.append(("E3: + early stopping", m, s))

# ---------------------------------------------------------------------------
# E4 — hyperparams algo más serios
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("E4 — Hyperparams ajustados")
print("=" * 72)
params_e4 = dict(
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=127,
    min_data_in_leaf=40,
    feature_fraction=0.85,
    bagging_fraction=0.85,
    bagging_freq=5,
    objective="multiclass",
    num_class=3,
    random_state=SEED,
    verbose=-1,
)
m, s, biters_e4 = run_cv(X_e2, y_lgb, cats_e2, params_e4,
                         use_early_stopping=True,
                         label="+ num_leaves=127, lr=0.03, fractions=0.85")
results.append(("E4: + hyperparams", m, s))

# ---------------------------------------------------------------------------
# Submission para el ganador
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("Resumen")
print("=" * 72)
prev = None
for name, m, s in results:
    delta = "" if prev is None else f"  delta vs anterior = {m - prev:+.4f}"
    print(f"  {name:<45} F1 = {m:.4f} ± {s:.4f}{delta}")
    prev = m

best_idx = int(np.argmax([r[1] for r in results]))
print(f"\nMejor: {results[best_idx][0]} → F1 = {results[best_idx][1]:.4f}")

# Generar submission con el mejor setup (E4 si gana, si no el que gane)
print("\nEntrenando modelo final en train completo y prediciendo test...")
if best_idx == 3:
    final_params = dict(params_e4)
    final_params["n_estimators"] = int(np.mean(biters_e4))
    final_cats = cats_e2
    X_final, X_test_final = X_e2, X_test_e2
elif best_idx == 2:
    final_params = dict(params_e3)
    final_params["n_estimators"] = int(np.mean(biters_e3))
    final_cats = cats_e2
    X_final, X_test_final = X_e2, X_test_e2
elif best_idx == 1:
    final_params = dict(params_e2)
    final_cats = cats_e2
    X_final, X_test_final = X_e2, X_test_e2
else:
    final_params = dict(params_e1)
    final_cats = cats_e1
    X_final, X_test_final = X_e1, X_test_e1

final_model = lgb.LGBMClassifier(**final_params)
final_model.fit(X_final, y_lgb, categorical_feature=final_cats)
test_pred = final_model.predict(X_test_final) + 1

sub = sub_fmt.copy()
sub["damage_grade"] = test_pred
sub_path = SUBS / "best_iterate.csv"
sub.to_csv(sub_path, index=False)
print(f"Submission guardado: {sub_path.relative_to(HERE.parent)}")
