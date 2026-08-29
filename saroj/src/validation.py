import json

import pandas as pd

from src.config import (
    BMI_BOUNDS, BMI_TOLERANCE, CLEANED_CSV, CYCLE_BOUNDS,
    FEET_TO_METERS, HEIGHT_M_BOUNDS, MENARCHE_MIN_AGE,
    PERIOD_BOUNDS, RESULTS_DIR, WEIGHT_BOUNDS,
)

RULES = {
    "cycle_length_out_of_bounds": lambda d: (
        (d.filter(regex="cycle_length").lt(CYCLE_BOUNDS[0])
         | d.filter(regex="cycle_length").gt(CYCLE_BOUNDS[1])).any(axis=1)
    ),
    "period_length_out_of_bounds": lambda d: (
        (d.filter(regex="period_length").lt(PERIOD_BOUNDS[0])
         | d.filter(regex="period_length").gt(PERIOD_BOUNDS[1])).any(axis=1)
    ),
    "menarche_after_age": lambda d: d["age_at_menarche"] > d["age"],
    "menarche_below_min": lambda d: d["age_at_menarche"] < MENARCHE_MIN_AGE,
    "bmi_out_of_bounds": lambda d: ~d["bmi"].between(*BMI_BOUNDS),
    "weight_out_of_bounds": lambda d: ~d["weight"].between(*WEIGHT_BOUNDS),
    "height_m_out_of_bounds": lambda d: ~(
        d["height"] * FEET_TO_METERS
    ).between(*HEIGHT_M_BOUNDS),
    "bmi_inconsistent": lambda d: (
        (d["bmi"] - d["weight"] / (d["height"] * FEET_TO_METERS) ** 2).abs() > BMI_TOLERANCE
    ),
}

HARD_VIOLATIONS = ["menarche_after_age"]


def load_raw(path=None):
    return pd.read_csv(path or CLEANED_CSV.parent.parent / ".." / "calibrated_menstrual_cycle_ml_dataset.csv")


def validate(df):
    report = {}
    report["n_rows"] = int(len(df))
    report["n_cols"] = int(df.shape[1])
    report["missing_per_column"] = {
        k: int(v) for k, v in df.isna().sum().items() if v > 0
    }
    dup_mask = df.duplicated(keep=False)
    report["duplicate_rows"] = int(dup_mask.sum())
    for name, fn in RULES.items():
        mask = fn(df).fillna(False)
        report[name] = {"count": int(mask.sum()), "indices": df.index[mask].tolist()}
    return report


def clean(df, report):
    drop = set()
    for name in HARD_VIOLATIONS:
        drop.update(report[name]["indices"])
    drop.update(df.index[df.duplicated()].tolist())
    out = df.drop(index=drop).reset_index(drop=True)
    return out, len(drop)


def run_validation(raw_path=None):
    df = load_raw(raw_path)
    report = validate(df)
    cleaned, n_dropped = clean(df, report)
    report["rows_dropped"] = int(n_dropped)
    report["rows_kept"] = int(len(cleaned))
    cleaned.to_csv(CLEANED_CSV, index=False)
    with open(RESULTS_DIR / "validation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    return cleaned, report
