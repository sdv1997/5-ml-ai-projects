"""
Proyecto 4 — DengAI: compositor de submissions por ciudad
==========================================================

Genera una submission eligiendo el modelo de cada ciudad por separado. Sirve
para AISLAR efectos en el leaderboard cambiando UNA sola ciudad a la vez
(sin desglose por ciudad de DrivenData, decodificamos por aritmética del MAE
pooled = (260·MAE_sj + 156·MAE_iq) / 416).

Modelos disponibles (de compare_models): seasonal-naive, lightgbm-l1, negbin-glm, sarimax.

Uso:    python 04_dengai/make_submission.py          # usa CHOICE de abajo
        python 04_dengai/make_submission.py sj=lightgbm-l1 iq=seasonal-naive
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_models as cm  # reutiliza build_features, MODELS, refit_predict, paths

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Combinación por defecto (sub3: aísla sj vs sub2, e iq vs sub1).
CHOICE = {"sj": "lightgbm-l1", "iq": "seasonal-naive"}


def main():
    for arg in sys.argv[1:]:
        if "=" in arg:
            k, v = arg.split("=", 1)
            CHOICE[k.strip()] = v.strip()
    for c, m in CHOICE.items():
        assert m in cm.MODELS, f"modelo desconocido: {m} (opciones: {list(cm.MODELS)})"

    feat = pd.read_csv(cm.DATA_DIR / "dengue_features_train.csv", parse_dates=["week_start_date"])
    lab = pd.read_csv(cm.DATA_DIR / "dengue_labels_train.csv")
    test = pd.read_csv(cm.DATA_DIR / "dengue_features_test.csv", parse_dates=["week_start_date"])
    train = feat.merge(lab, on=["city", "year", "weekofyear"], how="left")

    print(f"Composición: {CHOICE}")
    subs = []
    for city in ["sj", "iq"]:
        trc = train[train.city == city].copy(); trc["_is_test"] = False
        tec = test[test.city == city].copy(); tec["_is_test"] = True
        full = cm.build_features(pd.concat([trc, tec], ignore_index=True, sort=False))
        lf, sf = full.attrs["lgb_feats"], full.attrs["stat_feats"]
        tr_f = full[~full._is_test].copy(); te_f = full[full._is_test].copy()
        pred = np.clip(np.round(cm.refit_predict(CHOICE[city], tr_f, te_f, lf, sf)), 0, None).astype(int)
        s = te_f[["city", "year", "weekofyear"]].copy(); s["total_cases"] = pred
        subs.append(s)
        print(f"  {city}: {CHOICE[city]:<16} pred media={pred.mean():.1f}  máx={pred.max()}")

    fmt = pd.read_csv(cm.DATA_DIR / "submission_format.csv")
    out = fmt[["city", "year", "weekofyear"]].merge(
        pd.concat(subs, ignore_index=True), on=["city", "year", "weekofyear"], how="left")
    assert out["total_cases"].notna().all()
    out["total_cases"] = out["total_cases"].astype(int)
    cm.SUB_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(cm.SUB_DIR / "submission.csv", index=False)
    print(f"\nSubmission escrita: {cm.SUB_DIR / 'submission.csv'}  ({len(out)} filas)")


if __name__ == "__main__":
    main()
