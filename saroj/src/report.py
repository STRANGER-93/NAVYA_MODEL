import json
from pathlib import Path

import pandas as pd

from src.config import RESULTS_DIR


def save_table(df, name):
    p = Path(name)
    if not p.is_absolute():
        p = RESULTS_DIR / (name if str(name).endswith(".csv") else f"{name}.csv")
    df.to_csv(p, index=False)
    return str(p)


def save_json(obj, name):
    p = Path(name)
    if not p.is_absolute():
        p = RESULTS_DIR / (name if str(name).endswith(".json") else f"{name}.json")
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    return str(p)


def show_leaderboard(df, sort_by=None, ascending=True):
    if sort_by and sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=ascending)
    return df.reset_index(drop=True)
