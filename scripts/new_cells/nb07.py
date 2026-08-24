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
                        TARGET_CLF, CV_FOLDS, SEED)
from src.models import clf_models, CLF_NAMES
from src.evaluate import cv_classification_folds
'''

CELLS = [
    '''\
train = pd.read_csv(SPLITS_DIR / "train.csv")
X = train[FEATURES]
y = train[TARGET_CLF].astype(int)
factories = clf_models((y == 1).sum(), (y == 0).sum())
rows = []
for name in CLF_NAMES:
    folds = cv_classification_folds(factories[name], X, y)
    for i, m in enumerate(folds, 1):
        rows.append({"model": name, "fold": i,
                     "pr_auc": m["pr_auc"], "f1": m["f1"], "roc_auc": m["roc_auc"]})
    print(name, "fold PR-AUCs:", np.round([m["pr_auc"] for m in folds], 3), flush=True)
fold_df = pd.DataFrame(rows)
fold_df.to_csv(RESULTS_DIR / "clf_fold_scores.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, metric in zip(axes, ["pr_auc", "f1"]):
    sns.boxplot(data=fold_df, x="model", y=metric, ax=ax, color="lightgreen",
                showmeans=True,
                meanprops={"marker": "D", "markerfacecolor": "firebrick", "markeredgecolor": "firebrick"})
    ax.set_title(f"5-fold {metric} spread"); ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=35)
plt.suptitle("Classification CV boxplots (diamond = mean)", y=1.02)
plt.tight_layout(); plt.savefig(FIGURES_DIR / "cv_boxplots_classification.png", dpi=150, bbox_inches="tight"); plt.show()

spread = fold_df.groupby("model")[["pr_auc", "f1", "roc_auc"]].agg(["mean", "std"]).round(4)
display(spread)
''',
]

HEADER = ("## Added diagnostics — per-fold CV boxplots\\n"
          "Fold-level PR-AUC / F1 distributions per model; variance here qualifies the mean "
          "reported in the leaderboard.")
