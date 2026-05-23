"""
Día 1 (iteración 4) — Target encoding OOF de los geo_level_*_id
================================================================

Hipótesis (basada en el repo del top-28, score 0.7539):
    El cuello de botella es geo_level_3_id con 11.595 niveles.
    LightGBM con `astype("category")` agrupa OK pero cada nivel tiene
    pocos edificios → splits ruidosos.
    Reemplazar (o complementar) con la PROPORCIÓN DE DAÑO observada
    por aldea da una representación densa con mucho más información
    útil para los árboles.

Pipeline:
    - 5-fold StratifiedKFold (mismo seed=42)
    - En cada fold:
        * Calculo encoding en el train portion (smoothing bayesiano)
        * Aplico encoding al val portion → datos OOF, sin leakage
        * Entreno LightGBM E4 con las features originales + 9 nuevas
    - Para retraining final en train completo: encoding calculado en
      todo el train, aplicado al test.

9 nuevas features (3 niveles geo × 3 clases de daño):
    geo_level_1_te_c0, geo_level_1_te_c1, geo_level_1_te_c2
    geo_level_2_te_c0, geo_level_2_te_c1, geo_level_2_te_c2
    geo_level_3_te_c0, geo_level_3_te_c1, geo_level_3_te_c2
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
# Cargar
# ---------------------------------------------------------------------------
X_raw = pd.read_csv(DATA / "train_values.csv")
y = pd.read_csv(DATA / "train_labels.csv")["damage_grade"]
X_test_raw = pd.read_csv(DATA / "test_values.csv")
sub_fmt = pd.read_csv(DATA / "submission_format.csv")

feature_cols = [c for c in X_raw.columns if c != "building_id"]
text_cat_cols = X_raw[feature_cols].select_dtypes(include="object").columns.tolist()
geo_cols = ["geo_level_1_id", "geo_level_2_id", "geo_level_3_id"]

y_lgb = y - 1  # 0-indexed
SEED = 42
N_SPLITS = 5
SMOOTHING = 5   # m. Cuanto más alto, más se confía en el global mean para
                # categorías con pocas muestras. Con m=5 y 22 edif/aldea de
                # media en geo_3, el encoding es ~82% local.

# ---------------------------------------------------------------------------
# Función de target encoding con smoothing bayesiano
# ---------------------------------------------------------------------------
def fit_target_encoding(X_fit, y_fit, col, smoothing=SMOOTHING):
    """Devuelve (mapping_dict, global_means) para encodar 3 columnas (una por clase)."""
    global_means = np.array([(y_fit == c).mean() for c in [0, 1, 2]])
    df = pd.DataFrame({col: X_fit[col].values, "y": y_fit.values})
    grouped = df.groupby(col)["y"]
    counts = grouped.size()
    mapping = {}
    for c in [0, 1, 2]:
        local_means = grouped.apply(lambda v, c=c: (v == c).mean())
        smoothed = (counts * local_means + smoothing * global_means[c]) / (counts + smoothing)
        mapping[c] = smoothed.to_dict()
    return mapping, global_means


def apply_target_encoding(X_apply, col, mapping, global_means):
    """Devuelve un DataFrame con 3 columnas: {col}_te_c0, _c1, _c2."""
    out = pd.DataFrame(index=X_apply.index)
    for c in [0, 1, 2]:
        out[f"{col}_te_c{c}"] = (X_apply[col].map(mapping[c])
                                  .fillna(global_means[c])
                                  .astype(float))
    return out


# ---------------------------------------------------------------------------
# Prep base (sin las nuevas features TE — se añaden por fold)
# ---------------------------------------------------------------------------
X_base = X_raw[feature_cols].copy()
X_test_base = X_test_raw[feature_cols].copy()

# Conversión a category para LightGBM
cat_cols = text_cat_cols + geo_cols
for col in cat_cols:
    X_base[col] = X_base[col].astype("category")
    X_test_base[col] = X_test_base[col].astype("category")

# ---------------------------------------------------------------------------
# Params LightGBM E4
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
    random_state=SEED,
    verbose=-1,
)

# ---------------------------------------------------------------------------
# CV
# ---------------------------------------------------------------------------
print(f"Smoothing m = {SMOOTHING}")
print(f"Niveles únicos: geo_1={X_raw['geo_level_1_id'].nunique()}, "
      f"geo_2={X_raw['geo_level_2_id'].nunique()}, "
      f"geo_3={X_raw['geo_level_3_id'].nunique()}\n")

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
fold_scores = []

t_total = time.time()
for fold, (tr_idx, val_idx) in enumerate(skf.split(X_base, y_lgb), 1):
    print(f"--- Fold {fold} / {N_SPLITS} ---")
    t0 = time.time()

    X_tr = X_base.iloc[tr_idx].copy()
    X_val = X_base.iloc[val_idx].copy()
    y_tr = y_lgb.iloc[tr_idx]
    y_val = y_lgb.iloc[val_idx]

    # Fit + apply target encoding sobre las 3 geo cols
    for col in geo_cols:
        # Para el fit necesitamos el geo como int, no como category
        X_tr_int = X_raw.iloc[tr_idx][col]
        X_val_int = X_raw.iloc[val_idx][col]
        mapping, gmeans = fit_target_encoding(
            pd.DataFrame({col: X_tr_int.values}),
            pd.Series(y_tr.values), col, SMOOTHING)
        enc_tr = apply_target_encoding(
            pd.DataFrame({col: X_tr_int.values}, index=tr_idx), col, mapping, gmeans)
        enc_val = apply_target_encoding(
            pd.DataFrame({col: X_val_int.values}, index=val_idx), col, mapping, gmeans)
        for new_col in enc_tr.columns:
            X_tr[new_col] = enc_tr[new_col].values
            X_val[new_col] = enc_val[new_col].values

    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(X_tr, y_tr, categorical_feature=cat_cols)
    pred = model.predict(X_val)
    f1 = f1_score(y_val, pred, average="micro")
    fold_scores.append(f1)
    print(f"  F1 = {f1:.4f}  ({time.time()-t0:.0f}s, {X_tr.shape[1]} features)")

print(f"\n[total CV: {time.time()-t_total:.0f}s]")
print(f"\nF1 micro CV: {np.mean(fold_scores):.4f} +/- {np.std(fold_scores):.4f}")
print(f"delta vs E4 (0.7498): {np.mean(fold_scores) - 0.7498:+.4f}")

# ---------------------------------------------------------------------------
# Retrain en train completo + submission
# ---------------------------------------------------------------------------
print("\nEntrenando modelo final en train completo...")
X_full = X_base.copy()
X_test_full = X_test_base.copy()

for col in geo_cols:
    mapping, gmeans = fit_target_encoding(
        pd.DataFrame({col: X_raw[col].values}),
        pd.Series(y_lgb.values), col, SMOOTHING)
    enc_full = apply_target_encoding(
        pd.DataFrame({col: X_raw[col].values}, index=X_full.index), col, mapping, gmeans)
    enc_test = apply_target_encoding(
        pd.DataFrame({col: X_test_raw[col].values}, index=X_test_full.index), col, mapping, gmeans)
    for new_col in enc_full.columns:
        X_full[new_col] = enc_full[new_col].values
        X_test_full[new_col] = enc_test[new_col].values

t0 = time.time()
model_final = lgb.LGBMClassifier(**lgb_params)
model_final.fit(X_full, y_lgb, categorical_feature=cat_cols)
print(f"Entrenamiento final: {time.time()-t0:.0f}s")

test_pred = model_final.predict(X_test_full) + 1
sub = sub_fmt.copy()
sub["damage_grade"] = test_pred
sub_path = SUBS / "lgbm_target_encoding.csv"
sub.to_csv(sub_path, index=False)
print(f"\nSubmission guardado: {sub_path.relative_to(HERE.parent)}")
print("Distribucion predicciones:")
print(sub["damage_grade"].value_counts(normalize=True).sort_index())
