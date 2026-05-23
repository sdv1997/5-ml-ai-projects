"""
Día 1 (iteración 5) — Feature engineering con agregaciones sobre geo_*
=======================================================================

Idea: el árbol no puede calcular agregaciones (count, mean) sobre los OTROS
edificios de la misma aldea/distrito. Si le doy esas features explícitas,
puede usar señales tipo "edificios viejos en aldea con mucho mud-stone".

Sin riesgo de leakage porque las agregaciones NO usan la variable objetivo,
solo otras features. Se pueden calcular sobre train+test combinados.
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

X = pd.read_csv(DATA / "train_values.csv")
y = pd.read_csv(DATA / "train_labels.csv")["damage_grade"]
X_test = pd.read_csv(DATA / "test_values.csv")
sub_fmt = pd.read_csv(DATA / "submission_format.csv")

# ---------------------------------------------------------------------------
# Feature engineering — agregaciones sobre geo_3 y geo_2
# ---------------------------------------------------------------------------
combined = pd.concat([X, X_test], ignore_index=True)
print(f"Combined shape (train+test): {combined.shape}")

# Agregaciones sobre aldea (geo_3)
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

# Agregaciones sobre distrito (geo_2)
agg2 = combined.groupby("geo_level_2_id").agg(
    geo2_count=("building_id", "size"),
    geo2_mean_age=("age", "mean"),
    geo2_mean_floors=("count_floors_pre_eq", "mean"),
).reset_index()

X = X.merge(agg3, on="geo_level_3_id", how="left").merge(agg2, on="geo_level_2_id", how="left")
X_test = X_test.merge(agg3, on="geo_level_3_id", how="left").merge(agg2, on="geo_level_2_id", how="left")

# Domain features (sin agregación)
def add_domain(df):
    superstructure_cols = [c for c in df.columns if c.startswith("has_superstructure_")]
    df["n_superstructure_types"] = df[superstructure_cols].sum(axis=1)
    df["volume_proxy"] = df["area_percentage"] * df["height_percentage"]
    df["age_per_floor"] = df["age"] / (df["count_floors_pre_eq"] + 1)
    # Diferencia entre el edificio y la media de su aldea
    df["age_minus_geo3_mean"] = df["age"] - df["geo3_mean_age"]
    df["floors_minus_geo3_mean"] = df["count_floors_pre_eq"] - df["geo3_mean_floors"]
    return df

X = add_domain(X)
X_test = add_domain(X_test)

# ---------------------------------------------------------------------------
# Prep para LightGBM
# ---------------------------------------------------------------------------
feature_cols = [c for c in X.columns if c != "building_id"]
text_cat_cols = X[feature_cols].select_dtypes(include="object").columns.tolist()
cat_cols = text_cat_cols + ["geo_level_1_id", "geo_level_2_id", "geo_level_3_id"]

X_lgb = X[feature_cols].copy()
X_test_lgb = X_test[feature_cols].copy()
for col in cat_cols:
    X_lgb[col] = X_lgb[col].astype("category")
    X_test_lgb[col] = X_test_lgb[col].astype("category")

print(f"Features finales: {len(feature_cols)} (original: 38, nuevas: {len(feature_cols)-38})")
new_feats = [c for c in feature_cols if c not in {"building_id"} and c not in pd.read_csv(DATA / "train_values.csv").columns]
print(f"Nuevas: {new_feats}\n")

# ---------------------------------------------------------------------------
# Params E4
# ---------------------------------------------------------------------------
y_lgb = y - 1
lgb_params = dict(
    n_estimators=250, learning_rate=0.03, num_leaves=127, min_data_in_leaf=40,
    feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5,
    objective="multiclass", num_class=3, random_state=42, verbose=-1,
)

# ---------------------------------------------------------------------------
# CV
# ---------------------------------------------------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = []
t_total = time.time()
for fold, (tr_idx, val_idx) in enumerate(skf.split(X_lgb, y_lgb), 1):
    t0 = time.time()
    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(X_lgb.iloc[tr_idx], y_lgb.iloc[tr_idx], categorical_feature=cat_cols)
    pred = model.predict(X_lgb.iloc[val_idx])
    f1 = f1_score(y_lgb.iloc[val_idx], pred, average="micro")
    scores.append(f1)
    print(f"Fold {fold}: F1={f1:.4f}  ({time.time()-t0:.0f}s)")

cv_mean, cv_std = np.mean(scores), np.std(scores)
print(f"\n[total CV: {time.time()-t_total:.0f}s]")
print(f"F1 CV: {cv_mean:.4f} +/- {cv_std:.4f}")
print(f"delta vs E4 (0.7498): {cv_mean - 0.7498:+.4f}")

# ---------------------------------------------------------------------------
# Submission si mejora
# ---------------------------------------------------------------------------
if cv_mean > 0.7498:
    print("\nMEJORA. Reentrenando en train completo...")
    t0 = time.time()
    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(X_lgb, y_lgb, categorical_feature=cat_cols)
    print(f"Retrain: {time.time()-t0:.0f}s")
    test_pred = model.predict(X_test_lgb) + 1
    sub = sub_fmt.copy()
    sub["damage_grade"] = test_pred
    sub_path = SUBS / "lgbm_feateng.csv"
    sub.to_csv(sub_path, index=False)
    print(f"Submission guardado: {sub_path.relative_to(HERE.parent)}")
else:
    print("\nNo supera E4. No se genera submission.")
