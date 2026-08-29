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
from src.config import (CLEANED_CSV, ENGINEERED_CSV, FIGURES_DIR,
                        TARGETS_REG, TARGET_CLF)
'''

CELLS = [
    '''\
df = pd.read_csv(CLEANED_CSV)
miss = df.isna().mean().sort_values(ascending=False)
plt.figure(figsize=(11, 4))
miss.plot(kind="bar", color="steelblue")
plt.ylabel("fraction missing")
plt.title("Missingness per column (raw check)")
plt.tight_layout(); plt.savefig(FIGURES_DIR / "eda_missingness.png", dpi=150); plt.show()
print("columns with any missing values:", int((miss > 0).sum()), "| total NaN:", int(df.isna().sum().sum()))
''',
    '''\
static_cols = ["age", "height", "weight", "bmi", "age_at_menarche",
               "medication_contraceptive", "stress", "sleep", "exercise"]
grp = df.groupby(static_cols).size()
dup_rows = int(grp[grp >= 2].sum())
print(f"rows sharing an identical static profile: {dup_rows} / {len(df)} "
      f"({dup_rows / len(df):.1%}) across {len(grp[grp >= 2])} groups | max group size: {int(grp.max())}")
plt.figure(figsize=(7, 4))
grp.value_counts().sort_index().plot(kind="bar", color="indianred")
plt.xlabel("rows per identical static profile"); plt.ylabel("number of profiles")
plt.title("Near-duplicate static profiles (i.i.d. sanity check)")
plt.tight_layout(); plt.savefig(FIGURES_DIR / "eda_near_duplicates.png", dpi=150); plt.show()
''',
    '''\
eng = pd.read_csv(ENGINEERED_CSV)
num_cols = eng.select_dtypes(include=[np.number, bool]).columns
corr = eng[num_cols].corr()
plt.figure(figsize=(16, 13))
sns.heatmap(corr, cmap="coolwarm", center=0, annot=False,
            xticklabels=True, yticklabels=True)
plt.xticks(fontsize=7); plt.yticks(fontsize=7)
plt.title("Full correlation matrix: raw + engineered features + targets")
plt.tight_layout(); plt.savefig(FIGURES_DIR / "eda_correlation_full.png", dpi=150); plt.show()
for t in TARGETS_REG + [TARGET_CLF]:
    s = corr[t].drop(index=[t]).sort_values(key=np.abs, ascending=False)[:5]
    print(f"top-5 |corr| with {t}:")
    print(s.round(3).to_string(), "\\n")
''',
]

HEADER = "## Added diagnostics — missingness, near-duplicate profiles, full correlation heatmap"
