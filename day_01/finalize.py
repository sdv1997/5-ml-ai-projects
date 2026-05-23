"""
Entrena el modelo ganador (E4) en train completo y genera la submission.
Hyperparams elegidos por la CV en iterate.py: F1 micro = 0.7497 +/- 0.0012.
"""
import pandas as pd
import lightgbm as lgb
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE.parent / "data"
SUBS = HERE / "submissions"
SUBS.mkdir(exist_ok=True)

X = pd.read_csv(DATA / "train_values.csv")
y = pd.read_csv(DATA / "train_labels.csv")["damage_grade"]
X_test = pd.read_csv(DATA / "test_values.csv")
sub_fmt = pd.read_csv(DATA / "submission_format.csv")

feature_cols = [c for c in X.columns if c != "building_id"]
cat_cols = (X[feature_cols].select_dtypes(include="object").columns.tolist()
            + ["geo_level_1_id", "geo_level_2_id", "geo_level_3_id"])

X_tr = X[feature_cols].copy()
X_te = X_test[feature_cols].copy()
for col in cat_cols:
    X_tr[col] = X_tr[col].astype("category")
    X_te[col] = X_te[col].astype("category")

# E4 ganador. best_iter medio en CV = 203, uso 250 para tener un colchón.
params = dict(
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

print("Entrenando E4 en train completo...")
model = lgb.LGBMClassifier(**params)
model.fit(X_tr, y - 1, categorical_feature=cat_cols)

print("Prediciendo test...")
test_pred = model.predict(X_te) + 1

sub = sub_fmt.copy()
sub["damage_grade"] = test_pred
sub_path = SUBS / "best_iterate.csv"
sub.to_csv(sub_path, index=False)
print(f"Submission guardado: {sub_path.relative_to(HERE.parent)}")
print(f"Shape: {sub.shape}, distribucion predicciones:")
print(sub["damage_grade"].value_counts(normalize=True).sort_index())
