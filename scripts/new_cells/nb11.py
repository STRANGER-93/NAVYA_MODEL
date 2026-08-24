SETUP = '''\
import sys, json
from pathlib import Path
root = Path.cwd()
while not (root / "src").exists() and root != root.parent:
    root = root.parent
sys.path.insert(0, str(root))
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from src.config import (SPLITS_DIR, RESULTS_DIR, OPTUNA_DIR, FIGURES_DIR,
                        FEATURES, TARGETS_REG, TARGET_CLF)
from src.tune import make_est
from src.preprocessing import build_preprocessor

trainval = pd.concat([pd.read_csv(SPLITS_DIR / "train.csv"),
                      pd.read_csv(SPLITS_DIR / "val.csv")], ignore_index=True)
winners = json.load(open(RESULTS_DIR / "winners.json"))
shap_dir = RESULTS_DIR / "shap_vals"
shap_dir.mkdir(exist_ok=True)

def winner_name(kind, target):
    if kind == "clf":
        return winners["classification"]["model"]
    return next(w["model"] for w in winners["regression"] if w["target"] == target)

def compute_shap(target):
    kind = "clf" if target == TARGET_CLF else "reg"
    wn = winner_name(kind, target)
    base = wn.replace("_tuned", "")
    pfile = OPTUNA_DIR / f"{kind}_{base}_{target}_best.json"
    q = json.load(open(pfile))["best_params"] if wn.endswith("_tuned") and pfile.exists() else {}
    y = trainval[target]
    spw = 1.0
    if kind == "clf":
        yi = y.astype(int)
        spw = float((yi == 0).sum()) / max((yi == 1).sum(), 1)
        y = yi
    est = make_est(kind, base, q, spw=spw)
    prep = build_preprocessor()
    Xt = prep.fit_transform(trainval[FEATURES])
    est.fit(Xt, y)
    sample = shap.sample(Xt, 120, random_state=42)
    ex = shap.TreeExplainer(est)
    sv = ex.shap_values(sample, check_additivity=False)
    vals = np.asarray(sv[1] if isinstance(sv, list) else sv)
    if vals.ndim == 3:
        vals = vals[:, :, -1]
    names = [str(n).split("__")[-1] for n in prep.get_feature_names_out()]
    np.savez_compressed(shap_dir / f"shap_{target}.npz", vals=vals,
                        sample=sample, names=np.array(names))
    return vals, sample, names
'''

CELLS = [
    '''\
for target in TARGETS_REG + [TARGET_CLF]:
    print("computing SHAP for", target, flush=True)
    vals, sample, names = compute_shap(target)
    means = np.abs(vals).mean(axis=0)
    order = np.argsort(means)[::-1]
    top_idx = order[:15]
    plt.figure(figsize=(8, 6))
    plt.barh(range(len(top_idx)), means[top_idx][::-1], color="teal")
    plt.yticks(range(len(top_idx)), [names[i] for i in top_idx][::-1])
    plt.xlabel("mean |SHAP value|")
    plt.title(f"Feature importance - {target}")
    plt.tight_layout(); plt.savefig(FIGURES_DIR / f"shap_bar_{target}.png", dpi=150); plt.show()

    top3 = [names[i] for i in order[:3]]
    print(target, "top-3:", top3, flush=True)
    for feat in top3:
        fi = names.index(feat)
        plt.figure(figsize=(7.5, 5))
        shap.dependence_plot(fi, vals, sample, feature_names=names, show=False)
        plt.gcf().suptitle(f"SHAP dependence - {target} - {feat}", y=1.02)
        plt.tight_layout()
        safe = feat.replace("/", "_")
        plt.savefig(FIGURES_DIR / f"shap_dep_{target}_{safe}.png", dpi=150, bbox_inches="tight")
        plt.show(); plt.close("all")
''',
]

HEADER = ("## Added diagnostics — mean |SHAP| bar plots & dependence plots\\n"
          "Bar-ranked importance per target plus dependence plots for the top-3 features "
          "(linearity vs nonlinear kinks). SHAP values persisted to reports/metrics/shap_vals/.")
