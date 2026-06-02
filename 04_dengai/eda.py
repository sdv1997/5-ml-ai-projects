"""
Proyecto 4 — DengAI: EDA
========================

Une features + labels de train, y explora lo que de verdad condiciona un
modelo de forecasting de conteos:
  - rangos temporales train vs test por ciudad (confirmar holdout futuro)
  - distribución del target total_cases (escala y asimetría por ciudad)
  - estacionalidad: casos medios por semana del año
  - missingness de las features (vienen con NaNs)
  - autocorrelación del target (¿cuánto pasado es útil?)

Salida: 04_dengai/eda.png + resumen por consola.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# La consola de Windows suele ser cp1252 y peta con flechas/guiones; forzamos UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "04_dengai"
OUT_DIR = Path(__file__).resolve().parent

FEATURES = [
    "ndvi_ne", "ndvi_nw", "ndvi_se", "ndvi_sw",
    "precipitation_amt_mm",
    "reanalysis_air_temp_k", "reanalysis_avg_temp_k", "reanalysis_dew_point_temp_k",
    "reanalysis_max_air_temp_k", "reanalysis_min_air_temp_k",
    "reanalysis_precip_amt_kg_per_m2", "reanalysis_relative_humidity_percent",
    "reanalysis_sat_precip_amt_mm", "reanalysis_specific_humidity_g_per_kg",
    "reanalysis_tdtr_k",
    "station_avg_temp_c", "station_diur_temp_rng_c",
    "station_max_temp_c", "station_min_temp_c", "station_precip_mm",
]


def load():
    feat = pd.read_csv(DATA_DIR / "dengue_features_train.csv", parse_dates=["week_start_date"])
    lab = pd.read_csv(DATA_DIR / "dengue_labels_train.csv")
    test = pd.read_csv(DATA_DIR / "dengue_features_test.csv", parse_dates=["week_start_date"])
    train = feat.merge(lab, on=["city", "year", "weekofyear"], how="left")
    return train, test


def summarize(train, test):
    print("=" * 60)
    print("DengAI — EDA")
    print("=" * 60)
    print(f"\nTrain: {len(train)} filas | Test: {len(test)} filas\n")

    print("Rango temporal por ciudad (train vs test):")
    for city in ["sj", "iq"]:
        tr = train[train.city == city]
        te = test[test.city == city]
        print(f"  {city}: train {tr.week_start_date.min().date()} → {tr.week_start_date.max().date()} "
              f"({len(tr)} sem) | test {te.week_start_date.min().date()} → {te.week_start_date.max().date()} "
              f"({len(te)} sem)")

    print("\nTarget total_cases por ciudad:")
    g = train.groupby("city")["total_cases"]
    desc = g.agg(["mean", "std", "min", "median", "max"]).round(2)
    print(desc.to_string())
    print("\n  Asimetría (skew):")
    for city in ["sj", "iq"]:
        s = train.loc[train.city == city, "total_cases"]
        print(f"    {city}: skew={s.skew():.2f}  (% semanas con 0 casos: {(s == 0).mean() * 100:.1f}%)")

    print("\nMissingness en features (train, top 8):")
    miss = train[FEATURES].isna().mean().sort_values(ascending=False) * 100
    print(miss.head(8).round(2).to_string())
    print(f"  Filas con algún NaN en features: {train[FEATURES].isna().any(axis=1).mean() * 100:.1f}%")

    print("\nAutocorrelación de total_cases (lag 1, 4, 8, 12, 52) por ciudad:")
    for city in ["sj", "iq"]:
        s = train.loc[train.city == city, "total_cases"].reset_index(drop=True)
        acs = {lag: s.autocorr(lag) for lag in [1, 4, 8, 12, 52]}
        print("  " + city + ": " + "  ".join(f"lag{l}={v:.2f}" for l, v in acs.items()))


def plot(train):
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    colors = {"sj": "tab:blue", "iq": "tab:orange"}

    # 1) Serie temporal del target por ciudad
    ax = axes[0, 0]
    for city in ["sj", "iq"]:
        d = train[train.city == city]
        ax.plot(d.week_start_date, d.total_cases, lw=0.8, color=colors[city], label=city)
    ax.set_title("total_cases en el tiempo (train)")
    ax.set_ylabel("casos semanales")
    ax.legend()

    # 2) Distribución del target (log1p) por ciudad
    ax = axes[0, 1]
    for city in ["sj", "iq"]:
        d = train.loc[train.city == city, "total_cases"]
        ax.hist(np.log1p(d), bins=40, alpha=0.6, color=colors[city], label=city)
    ax.set_title("Distribución de log1p(total_cases)")
    ax.set_xlabel("log1p(casos)")
    ax.legend()

    # 3) Estacionalidad: media por semana del año
    ax = axes[1, 0]
    for city in ["sj", "iq"]:
        d = train[train.city == city].groupby("weekofyear")["total_cases"].mean()
        ax.plot(d.index, d.values, color=colors[city], label=city)
    ax.set_title("Casos medios por semana del año (estacionalidad)")
    ax.set_xlabel("weekofyear")
    ax.set_ylabel("casos medios")
    ax.legend()

    # 4) Missingness por feature
    ax = axes[1, 1]
    miss = (train[FEATURES].isna().mean() * 100).sort_values()
    ax.barh(range(len(miss)), miss.values, color="tab:red", alpha=0.7)
    ax.set_yticks(range(len(miss)))
    ax.set_yticklabels(miss.index, fontsize=7)
    ax.set_title("% de NaNs por feature (train)")
    ax.set_xlabel("% missing")

    fig.tight_layout()
    out = OUT_DIR / "eda.png"
    fig.savefig(out, dpi=110)
    print(f"\nPlot guardado en {out}")


if __name__ == "__main__":
    train, test = load()
    summarize(train, test)
    plot(train)
