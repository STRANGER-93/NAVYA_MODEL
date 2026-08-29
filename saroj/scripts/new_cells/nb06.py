SETUP = '''\
import sys
from pathlib import Path
root = Path.cwd()
while not (root / "src").exists() and root != root.parent:
    root = root.parent
sys.path.insert(0, str(root))
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")
from src.config import (SPLITS_DIR, RESULTS_DIR, FIGURES_DIR, FEATURES,
                        TARGETS_REG, CV_FOLDS, SEED)
from src.models import reg_models, REG_NAMES
from src.evaluate import cv_regression_folds
'''

CELLS = [
    '''\
train = pd.read_csv(SPLITS_DIR / "train.csv")
X = train[FEATURES]
factories = reg_models()
rows = []
for target in TARGETS_REG:
    y = train[target]
    for name in REG_NAMES:
        folds = cv_regression_folds(factories[name], X, y)
        for i, m in enumerate(folds, 1):
            rows.append({"model": name, "target": target, "fold": i,
                         "mae": m["mae"], "rmse": m["rmse"], "r2": m["r2"]})
        print(target, name, "fold MAEs:", np.round([m["mae"] for m in folds], 3), flush=True)
fold_df = pd.DataFrame(rows)
fold_df.to_csv(RESULTS_DIR / "reg_fold_scores.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, target in zip(axes, TARGETS_REG):
    sns.boxplot(data=fold_df[fold_df.target == target], x="model", y="mae", ax=ax,
                color="skyblue", showmeans=True,
                meanprops={"marker": "D", "markerfacecolor": "firebrick", "markeredgecolor": "firebrick"})
    ax.set_title(f"5-fold MAE spread - {target}")
    ax.set_xlabel(""); ax.tick_params(axis="x", rotation=35)
plt.suptitle("Regression CV boxplots (diamond = mean)", y=1.02)
plt.tight_layout(); plt.savefig(FIGURES_DIR / "cv_boxplots_regression.png", dpi=150, bbox_inches="tight"); plt.show()

spread = (fold_df.groupby(["target", "model"])["mae"].agg(["mean", "std"])
          .assign(cv_rel=lambda d: d["std"] / d["mean"]).round(4))
display(spread)
''',
]

HEADER = ("## Added diagnostics — per-fold CV boxplots\\n"
          "Each model re-scored with fold-level outputs so the spread across folds is visible "
          "(a good mean with wild fold swings is a red flag).")
