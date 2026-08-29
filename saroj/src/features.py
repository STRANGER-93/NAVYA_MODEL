import numpy as np
import pandas as pd

from src.config import ENGINEERED_CSV, ORDINAL_ORDER

CYCLE_COLS = ["prev1_cycle_length", "prev2_cycle_length", "prev3_cycle_length"]
PERIOD_COLS = ["prev1_period_length", "prev2_period_length", "prev3_period_length"]
X = np.array([1.0, 2.0, 3.0])
X_VAR = ((X - X.mean()) ** 2).sum()


def _slope(values):
    return ((X - X.mean()) * (values - values.mean(axis=1)[:, None])).sum(axis=1) / X_VAR


def engineer(df):
    out = df.copy()
    cyc = out[CYCLE_COLS].to_numpy(dtype=float)
    per = out[PERIOD_COLS].to_numpy(dtype=float)

    out["cycle_mean_3"] = cyc.mean(axis=1)
    out["cycle_std_3"] = cyc.std(axis=1, ddof=0)
    out["cycle_range_3"] = np.ptp(cyc, axis=1)
    out["cycle_slope_3"] = _slope(cyc[:, ::-1])

    out["period_mean_3"] = per.mean(axis=1)
    out["period_std_3"] = per.std(axis=1, ddof=0)
    out["period_range_3"] = np.ptp(per, axis=1)
    out["period_slope_3"] = _slope(per[:, ::-1])

    irr = out[["prev1_irregular_flag", "prev2_irregular_flag", "prev3_irregular_flag"]].to_numpy()
    out["irregular_count_3"] = irr.sum(axis=1)

    for sym in ("cramps", "fatigue", "mood_swings"):
        cols = [f"prev{i}_symptom_{sym}" for i in (1, 2, 3)]
        out[f"{sym}_count_3"] = out[cols].to_numpy().sum(axis=1)
    sym_cols = [c for c in out.columns if "_symptom_" in c]
    out["symptom_any_count_3"] = out[sym_cols].sum(axis=1)

    stress_code = out["stress"].map({v: i for i, v in enumerate(ORDINAL_ORDER["stress"])})
    exercise_code = out["exercise"].map({v: i for i, v in enumerate(ORDINAL_ORDER["exercise"])})
    out["bmi_x_stress"] = out["bmi"] * stress_code
    out["sleep_x_exercise"] = out["sleep"] * exercise_code
    out["bmi_x_exercise"] = out["bmi"] * exercise_code
    return out


def run_features(cleaned_csv=None):
    df = pd.read_csv(cleaned_csv or ENGINEERED_CSV.parent / "cleaned.csv")
    eng = engineer(df)
    eng.to_csv(ENGINEERED_CSV, index=False)
    return eng
