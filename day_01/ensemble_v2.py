"""
Día 1 (iteración 6) — Ensemble v2: LightGBM (con feateng) + CatBoost
=====================================================================

Repite el ensemble de la iter. 3 pero con las features mejoradas de la iter. 5.

Esperado: ~0.752-0.753 CV (LightGBM solo subió de 0.7498 a 0.7510 con feateng;
el ensemble v1 ya daba +0.0020 sobre LightGBM solo, así que esperamos algo
similar o mejor).
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
# Cargar + Feature engineering (igual que feateng.py)
# ---------------------------------------------------------------------------
X = pd.read_csv(DATA / "train_values.csv")
y = pd.read_csv(DATA / "train_labels.csv")["damage_grade"]
X_test = pd.read_csv(DATA / "test_values.csv")
sub_fmt = pd.read_csv(DATA / "submission_format.csv")

combined = pd.concat([X, X_test], ignore_index=True)

agg3 = combined.groupby("geo_level_3_id").agg(
    geo3_count=("building_id", "size"),
    geo3_mean_age=("age", "mean"),
    geo3_mean_floors=("count_floors_pre_eq", "mean"),
    geo3_mean_area=("area_percentage", "mean"),
    geo3_mean_height=("height_percentage", "mean"),
    geo3_mean_families=("count_families", "mean"),
    geo3_pct_adobe_mud=("has_superstructure_adobe_mud", "mean"),
    geo3_pct_mud_stone=("has_superstructure_mud_mortar_stone", "mean"),
    geo3_pct_rc_eng=("has_superstructure_rc_engineered", "mean"),
    geo3_pct_timber=("has_superstructure_timber", "mean"),
).reset_index()

agg2 = combined.groupby("geo_level_2_id").agg(
    geo2_count=("building_id", "size"),
    geo2_mean_age=("age", "mean"),
    geo2_mean_floors=("count_floors_pre_eq", "mean"),
).reset_index()

X = X.merge(agg3, on="geo_level_3_id", how="left").merge(agg2, on="geo_level_2_id", how="left")
X_test = X_test.merge(agg3, on="geo_level_3_id", how="left").merge(agg2, on="geo_level_2_id", how="left")

def add_domain(df):
    superstructure_cols = [c for c in df.columns if c.startswith("has_superstructure_")]
    df["n_superstructure_types"] = df[superstructure_cols].sum(axis=1)
    df["volume_proxy"] = df["area_percentage"] * df["height_percentage"]
    df["age_per_floor"] = df["age"] / (df["count_floors_pre_eq"] + 1)
    df["age_minus_geo3_mean"] = df["age"] - df["geo3_mean_age"]
    df["floors_minus_geo3_mean"] = df["count_floors_pre_eq"] - df["geo3_mean_floors"]
    return df

X = add_domain(X)
X_test = add_domain(X_test)

feature_cols = [c for c in X.columns if c != "building_id"]
text_cat_cols = X[feature_cols].select_dtypes(include="object").columns.tolist()
cat_cols = text_cat_cols + ["geo_level_1_id", "geo_level_2_id", "geo_level_3_id"]
y_lgb = y - 1

# Prep LightGBM (categoria nativa)
X_lgb = X[feature_cols].copy()
X_test_lgb = X_test[feature_cols].copy()
for col in cat_cols:
    X_lgb[col] = X_lgb[col].astype("category")
    X_test_lgb[col] = X_test_lgb[col].astype("category")

# Prep CatBoost (cat features como string)
X_cb = X[feature_cols].copy()
X_test_cb = X_test[feature_cols].copy()
for col in cat_cols:
    X_cb[col] = X_cb[col].astype(str)
    X_test_cb[col] = X_test_cb[col].astype(str)

print(f"Features finales: {len(feature_cols)}")

# ---------------------------------------------------------------------------
# Params
# ---------------------------------------------------------------------------
lgb_params = dict(
    n_estimators=250, learning_rate=0.03, num_leaves=127, min_data_in_leaf=40,
    feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5,
    objective="multiclass", num_class=3, random_state=42, verbose=-1,
)
cb_params = dict(
    iterations=300, learning_rate=0.05, depth=6, l2_leaf_reg=3,
    random_seed=42, loss_function="MultiClass",
    eval_metric="TotalF1:average=Micro",
    verbose=100, allow_writing_files=False,
)

# ---------------------------------------------------------------------------
# CV
# ---------------------------------------------------------------------------
SEED = 42
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
scores_lgb, scores_cb, scores_ens = [], [], []
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
    scores_lgb.append(f1_lgb)
    print(f"  LightGBM  F1={f1_lgb:.4f}  ({time.time()-t0:.0f}s)")

    # CatBoost
    t0 = time.time()
    m_cb = CatBoostClassifier(**cb_params, cat_features=cat_cols)
    m_cb.fit(X_cb.iloc[tr_idx], y_tr,
             eval_set=(X_cb.iloc[val_idx], y_val),
             early_stopping_rounds=50, use_best_model=True)
    proba_cb = m_cb.predict_proba(X_cb.iloc[val_idx])
    f1_cb = f1_score(y_val, proba_cb.argmax(axis=1), average="micro")
    scores_cb.append(f1_cb)
    best_iters_cb.append(m_cb.get_best_iteration())
    print(f"  CatBoost  F1={f1_cb:.4f}  ({time.time()-t0:.0f}s)  best_iter={m_cb.get_best_iteration()}")

    proba_ens = (proba_lgb + proba_cb) / 2
    f1_ens = f1_score(y_val, proba_ens.argmax(axis=1), average="micro")
    scores_ens.append(f1_ens)
    print(f"  Ensemble  F1={f1_ens:.4f}")

print(f"\n[total CV: {time.time()-t_total:.0f}s]")
print("\n" + "=" * 56)
print("Resumen CV (5-fold)")
print("=" * 56)
for name, s in [("LightGBM", scores_lgb), ("CatBoost", scores_cb), ("Ensemble", scores_ens)]:
    print(f"  {name:<10}: F1 = {np.mean(s):.4f} +/- {np.std(s):.4f}")

# ---------------------------------------------------------------------------
# Submission ensemble
# ---------------------------------------------------------------------------
print("\nEntrenando modelos finales en train completo...")
t0 = time.time()
m_lgb_full = lgb.LGBMClassifier(**lgb_params)
m_lgb_full.fit(X_lgb, y_lgb, categorical_feature=cat_cols)
proba_test_lgb = m_lgb_full.predict_proba(X_test_lgb)
print(f"  LightGBM: {time.time()-t0:.0f}s")

cb_params_final = dict(cb_params)
cb_params_final["iterations"] = int(np.mean(best_iters_cb))
t0 = time.time()
m_cb_full = CatBoostClassifier(**cb_params_final, cat_features=cat_cols)
m_cb_full.fit(X_cb, y_lgb)
proba_test_cb = m_cb_full.predict_proba(X_test_cb)
print(f"  CatBoost: {time.time()-t0:.0f}s")

proba_test_ens = (proba_test_lgb + proba_test_cb) / 2
test_pred_ens = proba_test_ens.argmax(axis=1) + 1
sub = sub_fmt.copy()
sub["damage_grade"] = test_pred_ens
sub_path = SUBS / "ensemble_v2_feateng.csv"
sub.to_csv(sub_path, index=False)
print(f"\nSubmission guardado: {sub_path.relative_to(HERE.parent)}")
print("Distribucion predicciones:")
print(sub["damage_grade"].value_counts(normalize=True).sort_index())
