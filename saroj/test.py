"""
test.py - Interactive tester for the trained ``next_cycle_length`` model
=======================================================================

Loads ``models/final_next_cycle_length.joblib`` (an ``XGBRegressor`` plus a
frozen ``ColumnTransformer``), lets you enter one row of raw cycle/patient
inputs, then prints the predicted next cycle length (in days).

Run:
    python test.py

Self-contained: it re-implements the small feature-engineering step that the
training pipeline applied (mirroring ``src/features.engineer``), so it works
without importing anything from ``src/``.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "final_next_cycle_length.joblib"

# Order must match the trained model exactly (== src.config.FEATURES)
NUMERIC_RAW_COLS = [
    "age", "height", "weight", "bmi", "age_at_menarche", "sleep",
    "prev1_cycle_length", "prev2_cycle_length", "prev3_cycle_length",
    "prev1_period_length", "prev2_period_length", "prev3_period_length",
]
ENGINEERED_NUMERIC = [
    "cycle_mean_3", "cycle_std_3", "cycle_range_3", "cycle_slope_3",
    "period_mean_3", "period_std_3", "period_range_3", "period_slope_3",
    "irregular_count_3",
    "cramps_count_3", "fatigue_count_3", "mood_swings_count_3",
    "symptom_any_count_3",
    "bmi_x_stress", "sleep_x_exercise", "bmi_x_exercise",
]
ORDINAL_COLS = ["stress", "exercise"]
ONEHOT_COLS = ["medication_contraceptive"]
BOOL_COLS = [
    "prev1_irregular_flag", "prev2_irregular_flag", "prev3_irregular_flag",
    "prev1_symptom_cramps", "prev1_symptom_fatigue", "prev1_symptom_mood_swings",
    "prev2_symptom_cramps", "prev2_symptom_fatigue", "prev2_symptom_mood_swings",
    "prev3_symptom_cramps", "prev3_symptom_fatigue", "prev3_symptom_mood_swings",
]

FEATURES = (
    NUMERIC_RAW_COLS + ENGINEERED_NUMERIC
    + ORDINAL_COLS + ONEHOT_COLS + BOOL_COLS
)

RAW_INPUT_COLS = NUMERIC_RAW_COLS + ORDINAL_COLS + ONEHOT_COLS + BOOL_COLS

ORDINAL_ORDER = {
    "stress": ["Very Low", "Low", "Moderate", "High", "Very High"],
    "exercise": ["Never", "1-2 days/week", "3-4 days/week", "5+ days/week"],
}

CYCLE_COLS = ["prev1_cycle_length", "prev2_cycle_length", "prev3_cycle_length"]
PERIOD_COLS = ["prev1_period_length", "prev2_period_length", "prev3_period_length"]
SYMPTOM_TYPES = ("cramps", "fatigue", "mood_swings")


# --------------------------------------------------------------------------
# Feature engineering (mirrors src/features.engineer)
# --------------------------------------------------------------------------
def _cycle_slope(values):
    """Rolling slope over the 3 past cycles / periods (1,2,3 window)."""
    x = np.array([1.0, 2.0, 3.0])
    x_var = ((x - x.mean()) ** 2).sum()
    return (
        (x - x.mean()) * (values - values.mean(axis=1)[:, None])
    ).sum(axis=1) / x_var


def engineer(df):
    """Add derived/rolling + interaction features so the row matches training."""
    out = df.copy()
    cyc = out[CYCLE_COLS].to_numpy(dtype=float)
    per = out[PERIOD_COLS].to_numpy(dtype=float)

    out["cycle_mean_3"] = cyc.mean(axis=1)
    out["cycle_std_3"] = cyc.std(axis=1, ddof=0)
    out["cycle_range_3"] = np.ptp(cyc, axis=1)
    out["cycle_slope_3"] = _cycle_slope(cyc[:, ::-1])

    out["period_mean_3"] = per.mean(axis=1)
    out["period_std_3"] = per.std(axis=1, ddof=0)
    out["period_range_3"] = np.ptp(per, axis=1)
    out["period_slope_3"] = _cycle_slope(per[:, ::-1])

    irr = out[
        ["prev1_irregular_flag", "prev2_irregular_flag", "prev3_irregular_flag"]
    ].to_numpy()
    out["irregular_count_3"] = irr.sum(axis=1)

    for sym in SYMPTOM_TYPES:
        cols = [f"prev{i}_symptom_{sym}" for i in (1, 2, 3)]
        out[f"{sym}_count_3"] = out[cols].to_numpy().sum(axis=1)
    sym_cols = [c for c in out.columns if "_symptom_" in c]
    out["symptom_any_count_3"] = out[sym_cols].sum(axis=1)

    stress_code = out["stress"].map(
        {v: i for i, v in enumerate(ORDINAL_ORDER["stress"])}
    )
    exercise_code = out["exercise"].map(
        {v: i for i, v in enumerate(ORDINAL_ORDER["exercise"])}
    )
    out["bmi_x_stress"] = out["bmi"] * stress_code
    out["sleep_x_exercise"] = out["sleep"] * exercise_code
    out["bmi_x_exercise"] = out["bmi"] * exercise_code
    return out
# --------------------------------------------------------------------------
# Prediction core
# --------------------------------------------------------------------------
def predict_from_inputs(row_dict):
    """Predict next_cycle_length from a dict of RAW (non-engineered) inputs.

    ``row_dict`` must contain every ``RAW_INPUT_COLS`` key:
      age, height(feet), weight(kg), bmi, age_at_menarche, sleep,
      medication_contraceptive, stress, exercise,
      prev1/2/3_cycle_length, prev1/2/3_period_length,
      prev1/2/3_irregular_flag,
      prev1/2/3_symptom_{cramps,fatigue,mood_swings}
    """
    bundle = joblib.load(MODEL_PATH)
    prep = bundle["prep"]
    model = bundle["model"]

    missing = [c for c in RAW_INPUT_COLS if c not in row_dict]
    if missing:
        raise ValueError(f"Missing raw inputs: {missing}")

    df = pd.DataFrame([{c: row_dict[c] for c in RAW_INPUT_COLS}])
    engineered = engineer(df)
    reduced = engineered[FEATURES]
    missing_feat = [c for c in FEATURES if c not in reduced.columns]
    if missing_feat:
        raise ValueError(f"Engineering failed to produce: {missing_feat}")
    Xt = prep.transform(reduced)
    prediction = model.predict(Xt)[0]
    return float(prediction)


# --------------------------------------------------------------------------
# Interactive prompt helpers
# --------------------------------------------------------------------------
def _prompt_number(label, default_value):
    text = input(f"{label}  [{default_value}]: ").strip()
    if not text:
        return default_value
    try:
        return float(text)
    except ValueError:
        print(f"  !! not a number, using default {default_value}")
        return default_value


def _prompt_int(label, default_value):
    text = input(f"{label}  [{default_value}]: ").strip()
    if not text:
        return default_value
    try:
        return int(text)
    except ValueError:
        print(f"  !! not an integer, using default {default_value}")
        return default_value
def _prompt_choice(label, options, default_index=0):
    for i, o in enumerate(options, 1):
        marker = "  <-- default" if i - 1 == default_index else ""
        print(f"    {i}. {o}{marker}")
    text = input(f"{label} (enter 1-{len(options)}): ").strip()
    if not text:
        return options[default_index]
    try:
        idx = int(text) - 1
        return options[idx]
    except (ValueError, IndexError):
        print(f"  !! invalid, using default {options[default_index]!r}")
        return options[default_index]


def _prompt_bool(label, default_value):
    hint = "y" if default_value else "n"
    text = input(f"{label}? (y/n)  [{hint}]: ").strip().lower()
    if not text:
        return default_value
    if text in ("y", "yes", "true", "1"):
        return True
    if text in ("n", "no", "false", "0"):
        return False
    print(f"  !! answering y, using default {default_value}")
    return default_value
# --------------------------------------------------------------------------
# Interactive main
# --------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("  PREDICT NEXT CYCLE LENGTH  (models/final_next_cycle_length.joblib)")
    print("  Press Enter to accept the bracketed default value.")
    print("=" * 72)

    # Sensible defaults taken from row 0 of the raw dataset.
    defaults = {
        "age": 24.0, "height": 5.0, "weight": 37.8, "bmi": 16.3,
        "age_at_menarche": 12.0, "sleep": 7.4,
        "medication_contraceptive": "No",
        "stress": "Moderate", "exercise": "Never",
        "prev1_cycle_length": 31, "prev2_cycle_length": 31,
        "prev3_cycle_length": 32,
        "prev1_period_length": 5, "prev2_period_length": 4,
        "prev3_period_length": 5,
        "prev1_irregular_flag": False, "prev2_irregular_flag": False,
        "prev3_irregular_flag": False,
        "prev1_symptom_cramps": True, "prev1_symptom_fatigue": True,
        "prev1_symptom_mood_swings": False,
        "prev2_symptom_cramps": True, "prev2_symptom_fatigue": False,
        "prev2_symptom_mood_swings": True,
        "prev3_symptom_cramps": False, "prev3_symptom_fatigue": True,
        "prev3_symptom_mood_swings": True,
    }

    row = {}

    print("\n-- Numeric inputs --")
    row["age"] = _prompt_number("age (years)", defaults["age"])
    row["height"] = _prompt_number("height (feet)", defaults["height"])
    row["weight"] = _prompt_number("weight (kg)", defaults["weight"])
    row["bmi"] = _prompt_number("bmi", defaults["bmi"])
    row["age_at_menarche"] = _prompt_number(
        "age_at_menarche (years)", defaults["age_at_menarche"])
    row["sleep"] = _prompt_number("sleep (hrs/day)", defaults["sleep"])

    print("\n-- Categoricals --")
    row["medication_contraceptive"] = _prompt_choice(
        "medication_contraceptive", ["No", "Yes", "Not Sure"], 0)
    row["stress"] = _prompt_choice("stress", ORDINAL_ORDER["stress"], 2)
    row["exercise"] = _prompt_choice("exercise", ORDINAL_ORDER["exercise"], 0)

    print("\n-- Previous 3 cycle lengths (days) --")
    for col in CYCLE_COLS:
        row[col] = _prompt_int(col, defaults[col])
    print("\n-- Previous 3 period lengths (days) --")
    for col in PERIOD_COLS:
        row[col] = _prompt_int(col, defaults[col])

    print("\n-- Irregular flags (was each of the last 3 cycles irregular?) --")
    for col in [c for c in BOOL_COLS if c.endswith("_flag")]:
        row[col] = _prompt_bool(col, defaults[col])

    print("\n-- Symptoms present in each of the last 3 cycles --")
    for col in [c for c in BOOL_COLS if not c.endswith("_flag")]:
        row[col] = _prompt_bool(col, defaults[col])

    predicted = predict_from_inputs(row)
    print("\n" + "=" * 60)
    print(f"  Predicted next cycle length : {predicted:.2f} days")
    print("=" * 60)


def run_demo():
    """Run a quick non-interactive sanity check on the first dataset row."""
    demo = {
        "age": 24.0, "height": 5.0, "weight": 37.8, "bmi": 16.3,
        "age_at_menarche": 12.0, "sleep": 7.4,
        "medication_contraceptive": "No", "stress": "Moderate",
        "exercise": "Never",
        "prev1_cycle_length": 31, "prev2_cycle_length": 31,
        "prev3_cycle_length": 32,
        "prev1_period_length": 5, "prev2_period_length": 4,
        "prev3_period_length": 5,
        "prev1_irregular_flag": False, "prev2_irregular_flag": False,
        "prev3_irregular_flag": False,
        "prev1_symptom_cramps": True, "prev1_symptom_fatigue": True,
        "prev1_symptom_mood_swings": False,
        "prev2_symptom_cramps": True, "prev2_symptom_fatigue": False,
        "prev2_symptom_mood_swings": True,
        "prev3_symptom_cramps": False, "prev3_symptom_fatigue": True,
        "prev3_symptom_mood_swings": True,
    }
    pred = predict_from_inputs(demo)
    true_val = 31  # dataset target for that row
    print(f"demo next_cycle_length prediction = {pred:.2f} days "
          f"(dataset target was {true_val})")


# Order of the 27 raw inputs as they appear in the source CSV
# (the 3 target columns next_cycle_length / next_period_length /
# next_is_irregular are excluded).
CSV_ORDER_COLS = [
    "age", "height", "weight", "bmi", "age_at_menarche",
    "medication_contraceptive", "stress", "sleep", "exercise",
    "prev1_cycle_length", "prev2_cycle_length", "prev3_cycle_length",
    "prev1_period_length", "prev2_period_length", "prev3_period_length",
    "prev1_irregular_flag", "prev2_irregular_flag", "prev3_irregular_flag",
    "prev1_symptom_cramps", "prev1_symptom_fatigue", "prev1_symptom_mood_swings",
    "prev2_symptom_cramps", "prev2_symptom_fatigue", "prev2_symptom_mood_swings",
    "prev3_symptom_cramps", "prev3_symptom_fatigue", "prev3_symptom_mood_swings",
]

# Data types for parsing each CSV-order field
CSV_ORDER_TYPES = [
    "num", "num", "num", "num", "num",           # age..age_at_menarche
    "str", "str", "num", "str",                    # med, stress, sleep, exercise
    "int", "int", "int",                            # cycle lengths
    "int", "int", "int",                            # period lengths
    "bool", "bool", "bool",                        # irregular flags
    "bool", "bool", "bool",                        # prev1 symptoms
    "bool", "bool", "bool",                        # prev2 symptoms
    "bool", "bool", "bool",                        # prev3 symptoms
]


def parse_csv_row(csv_string):
    """Parse a comma-separated row (dataset column order) into a raw-input dict."""
    parts = [p.strip() for p in csv_string.split(",")]
    if len(parts) != len(CSV_ORDER_COLS):
        raise ValueError(
            f"Expected {len(CSV_ORDER_COLS)} comma-separated values, "
            f"got {len(parts)}"
        )
    row = {}
    for col, kind, val in zip(CSV_ORDER_COLS, CSV_ORDER_TYPES, parts):
        if kind == "num":
            row[col] = float(val)
        elif kind == "int":
            row[col] = int(float(val))
        elif kind == "bool":
            row[col] = val.strip().lower() in ("true", "1", "yes", "y")
        else:
            row[col] = val
    return row


def run_row(csv_string):
    """Parse a CSV-order row and print the prediction."""
    row = parse_csv_row(csv_string)
    pred = predict_from_inputs(row)
    print("Inputs (CSV column order):")
    for col in CSV_ORDER_COLS:
        print(f"  {col:34s} = {row[col]}")
    print("=" * 60)
    print(f"  Predicted next cycle length : {pred:.2f} days")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    if "--demo" in sys.argv:
        run_demo()
    elif "--row" in sys.argv:
        idx = sys.argv.index("--row")
        if idx + 1 >= len(sys.argv):
            print("usage: python test.py --row <comma-separated CSV inputs>")
            sys.exit(2)
        run_row(sys.argv[idx + 1])
    else:
        main()