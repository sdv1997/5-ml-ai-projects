"""
Día 1 (iteración 3) — LightGBM vs CatBoost vs ensemble
========================================================

Compara sobre los mismos folds (StratifiedKFold seed=42):
    - LightGBM E4 (ganador de iterate.py): F1 CV = 0.7497
    - CatBoost out-of-the-box con cat_features nativo
    - Ensemble: media de probabilidades por fold

Genera submission del ensemble (entrena ambos modelos en train completo).
"""
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
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

feature_cols = [c for c in X.columns if c != "building_id"]
text_cat_cols = X[feature_cols].select_dtypes(include="object").columns.tolist()
geo_cols = ["geo_level_1_id", "geo_level_2_id", "geo_level_3_id"]
cat_cols = text_cat_cols + geo_cols
y_lgb = y - 1  # 0-indexed

# Para LightGBM: categoria nativa de pandas
X_lgb = X[feature_cols].copy()
X_test_lgb = X_test[feature_cols].copy()
for col in cat_cols:
    X_lgb[col] = X_lgb[col].astype("category")
    X_test_lgb[col] = X_test_lgb[col].astype("category")

# Para CatBoost: cast a string para que cat_features funcione siempre
X_cb = X[feature_cols].copy()
X_test_cb = X_test[feature_cols].copy()
for col in cat_cols:
    X_cb[col] = X_cb[col].astype(str)
    X_test_cb[col] = X_test_cb[col].astype(str)

# ---------------------------------------------------------------------------
# Params
# ---------------------------------------------------------------------------
lgb_params = dict(
    n_estimators=250,
    learning_rate=0.03,
    num_leaves=127,
    min_data_in_leaf=40,
    feature_fraction=0.85,
    bagging_fraction=0.85,
    bagging_freq=5,
    objective="multiclass",
    num_class=3,
    random_state=42,
    verbose=-1,
)

cb_params = dict(
    iterations=300,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3,
    random_seed=42,
    loss_function="MultiClass",
    eval_metric="TotalF1:average=Micro",
    verbose=100,           # imprime cada 100 iteraciones
    allow_writing_files=False,
)

# ---------------------------------------------------------------------------
# CV
# ---------------------------------------------------------------------------
SEED = 42
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

fold_scores_lgb, fold_scores_cb, fold_scores_ens = [], [], []
best_iters_cb = []

t_total = time.time()
for fold, (tr_idx, val_idx) in enumerate(skf.split(X_lgb, y_lgb), 1):
    print(f"\n--- Fold {fold} / {N_SPLITS} ---")
    y_tr, y_val = y_lgb.iloc[tr_idx], y_lgb.iloc[val_idx]

    # LightGBM
    t0 = time.time()
    m_lgb = lgb.LGBMClassifier(**lgb_params)
    m_lgb.fit(X_lgb.iloc[tr_idx], y_tr, categorical_feature=cat_cols)
    proba_lgb = m_lgb.predict_proba(X_lgb.iloc[val_idx])
    f1_lgb = f1_score(y_val, proba_lgb.argmax(axis=1), average="micro")
    fold_scores_lgb.append(f1_lgb)
    print(f"  LightGBM  F1={f1_lgb:.4f}  ({time.time()-t0:.0f}s)")

    # CatBoost
    t0 = time.time()
    m_cb = CatBoostClassifier(**cb_params, cat_features=cat_cols)
    m_cb.fit(X_cb.iloc[tr_idx], y_tr,
             eval_set=(X_cb.iloc[val_idx], y_val),
             early_stopping_rounds=50, use_best_model=True)
    proba_cb = m_cb.predict_proba(X_cb.iloc[val_idx])
    f1_cb = f1_score(y_val, proba_cb.argmax(axis=1), average="micro")
    fold_scores_cb.append(f1_cb)
    best_iters_cb.append(m_cb.get_best_iteration())
    print(f"  CatBoost  F1={f1_cb:.4f}  ({time.time()-t0:.0f}s)  best_iter={m_cb.get_best_iteration()}")

    # Ensemble: media de probabilidades
    proba_ens = (proba_lgb + proba_cb) / 2
    f1_ens = f1_score(y_val, proba_ens.argmax(axis=1), average="micro")
    fold_scores_ens.append(f1_ens)
    print(f"  Ensemble  F1={f1_ens:.4f}")

print(f"\n[total CV: {time.time()-t_total:.0f}s]")

# ---------------------------------------------------------------------------
# Resumen
# ---------------------------------------------------------------------------
print("\n" + "=" * 56)
print("Resumen CV (5-fold)")
print("=" * 56)
for name, scores in [("LightGBM", fold_scores_lgb),
                     ("CatBoost", fold_scores_cb),
                     ("Ensemble", fold_scores_ens)]:
    print(f"  {name:<10}: F1 = {np.mean(scores):.4f} +/- {np.std(scores):.4f}")

best_iter_mean = int(np.mean(best_iters_cb))
print(f"\nCatBoost best_iter medio en CV: {best_iter_mean}")

# ---------------------------------------------------------------------------
# Entrenar en train completo y generar submission del ensemble
# ---------------------------------------------------------------------------
print("\nEntrenando ambos modelos en train completo...")
t0 = time.time()
m_lgb_full = lgb.LGBMClassifier(**lgb_params)
m_lgb_full.fit(X_lgb, y_lgb, categorical_feature=cat_cols)
proba_test_lgb = m_lgb_full.predict_proba(X_test_lgb)
print(f"  LightGBM entrenado en {time.time()-t0:.0f}s")

# Para CatBoost final, fijo iteraciones = mean best_iter de los folds
cb_params_final = dict(cb_params)
cb_params_final["iterations"] = best_iter_mean
t0 = time.time()
m_cb_full = CatBoostClassifier(**cb_params_final, cat_features=cat_cols)
m_cb_full.fit(X_cb, y_lgb)
proba_test_cb = m_cb_full.predict_proba(X_test_cb)
print(f"  CatBoost entrenado en {time.time()-t0:.0f}s")

proba_test_ens = (proba_test_lgb + proba_test_cb) / 2
test_pred_ens = proba_test_ens.argmax(axis=1) + 1

sub = sub_fmt.copy()
sub["damage_grade"] = test_pred_ens
sub_path = SUBS / "ensemble_lgb_cb.csv"
sub.to_csv(sub_path, index=False)
print(f"\nSubmission ensemble guardado: {sub_path.relative_to(HERE.parent)}")
print(f"Distribucion predicciones:")
print(sub["damage_grade"].value_counts(normalize=True).sort_index())

# Tambien submission CatBoost solo, por curiosidad
sub_cb = sub_fmt.copy()
sub_cb["damage_grade"] = proba_test_cb.argmax(axis=1) + 1
sub_cb.to_csv(SUBS / "catboost_solo.csv", index=False)
print(f"Submission CatBoost solo: {(SUBS / 'catboost_solo.csv').relative_to(HERE.parent)}")
