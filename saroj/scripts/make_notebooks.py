import nbformat as nbf


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(src):
    return nbf.v4.new_code_cell(src)


SETUP = '''
import sys, os
from pathlib import Path
root = Path.cwd()
while not (root / "src").exists() and root != root.parent:
    root = root.parent
sys.path.insert(0, str(root))
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")'''

NOTEBOOKS = {}

NOTEBOOKS["01_data_loading_and_validation"] = [
    md("# 01 â€” Data Loading & Validation\nLoads raw CSV, runs rule-based validation, saves `data/processed/cleaned.csv`."),
    code('''\
from src.validation import run_validation
import pandas as pd

cleaned, report = run_validation()
rows = [{"rule": k, "flagged": v["count"]} for k, v in report.items() if isinstance(v, dict) and "count" in v]
pd.DataFrame(rows)
'''),
    md("## Summary"),
    code('''\
print(f"raw rows: {report['n_rows']}, cols: {report['n_cols']}")
print(f"duplicate rows: {report['duplicate_rows']}")
print(f"missing values: {report['missing_per_column'] or 'none'}")
print(f"dropped (hard violations/dups): {report['rows_dropped']} -> kept {report['rows_kept']}")
cleaned.describe().T.round(3)
'''),
]

NOTEBOOKS["02_exploratory_analysis"] = [
    md("# 02 â€” Exploratory Analysis\nTarget distributions, correlations of prevN_* features, class balance."),
    code('''\
import pandas as pd
from src.config import CLEANED_CSV, FIGURES_DIR, TARGETS_REG, TARGET_CLF

df = pd.read_csv(CLEANED_CSV)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, t in zip(axes, TARGETS_REG):
    sns.histplot(df[t], bins=30, ax=ax, kde=True)
    ax.set_title(t)
plt.tight_layout(); plt.savefig(FIGURES_DIR / "eda_target_distributions.png", dpi=150); plt.show()

print(df[TARGETS_REG].describe().round(3))
'''),
    md("## Class balance of `next_is_irregular`"),
    code('''\
counts = df[TARGET_CLF].value_counts()
print(counts)
print("positive rate:", round(counts[True] / len(df), 4))
ax = sns.countplot(x=df[TARGET_CLF]); ax.set_title("next_is_irregular class balance")
plt.savefig(FIGURES_DIR / "eda_class_balance.png", dpi=150); plt.show()
'''),
    md("## Correlation matrix (prevN_* and targets)"),
    code('''\
cols = [c for c in df.columns if c.startswith(("prev1_", "prev2_", "prev3_", "next_"))
        or c in ("age", "bmi", "sleep", "age_at_menarche")]
corr = df[cols].corr(numeric_only=True)
plt.figure(figsize=(14, 11))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, annot_kws={"size": 7})
plt.title("Feature correlation matrix")
plt.tight_layout(); plt.savefig(FIGURES_DIR / "eda_correlation_matrix.png", dpi=150); plt.show()
'''),
    md("## Categoricals vs targets"),
    code('''\
for cat in ["stress", "exercise", "medication_contraceptive"]:
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.5))
    for ax, t in zip(axes, TARGETS_REG + [TARGET_CLF]):
        if t == TARGET_CLF:
            df.groupby(cat)[t].mean().sort_index().plot(kind="bar", ax=ax, color="salmon")
            ax.set_ylabel("irregular rate")
        else:
            sns.boxplot(data=df, x=cat, y=t, ax=ax)
        ax.set_title(f"{t} by {cat}"); ax.tick_params(axis="x", rotation=30)
    plt.tight_layout(); plt.savefig(FIGURES_DIR / f"eda_{cat}_vs_targets.png", dpi=150); plt.show()
'''),
]

NOTEBOOKS["03_feature_engineering"] = [
    md("# 03 â€” Feature Engineering\nRolling stats over prev1-3, irregular/symptom counts, ordinal interactions. Target-free -> no leakage."),
    code('''\
from src.features import run_features
from src.config import ENGINEERED_CSV, ENGINEERED_NUMERIC

eng = run_features()
eng[ENGINEERED_NUMERIC].describe().T.round(3)
'''),
    md("## New-feature correlations with targets"),
    code('''\
from src.config import FIGURES_DIR

targets = ["next_cycle_length", "next_period_length", "next_is_irregular"]
corr = eng[ENGINEERED_NUMERIC + targets].corr(numeric_only=True)[targets].drop(index=targets)
display(corr.round(3))
plt.figure(figsize=(5, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Engineered features vs targets")
plt.tight_layout(); plt.savefig(FIGURES_DIR / "feat_engineered_correlations.png", dpi=150); plt.show()
print("engineered shape:", eng.shape)
print("saved ->", ENGINEERED_CSV)
'''),
]

NOTEBOOKS["04_preprocessing_and_split"] = [
    md("# 04 â€” Preprocessing & Split\nColumnTransformer (fold-safe): median+scale numeric, ordinal encode stress/exercise, one-hot contraceptive. 70/15/15 stratified on `next_is_irregular`.\n\n**i.i.d. assumption**: no user id/timestamp column exists; rows treated independent. If rows were repeated measures, mild leakage could inflate CV slightly (noted in final report)."),
    code('''\
import pandas as pd
from src.split import run_split
from src.config import SPLITS_DIR

train, val, test = run_split()
for name, d in [("train", train), ("val", val), ("test", test)]:
    print(name, d.shape, "irregular rate:", round(d["next_is_irregular"].mean(), 4))
'''),
    md("## Preprocessor preview"),
    code('''\
from src.preprocessing import build_preprocessor
from src.config import FEATURES

prep = build_preprocessor()
Xt = prep.fit_transform(train[FEATURES])
names = prep.get_feature_names_out()
print(len(FEATURES), "input features ->", len(names), "transformed columns")
print(list(names[:8]), "...", list(names[-4:]))
print("splits saved ->", SPLITS_DIR)
'''),
]

NOTEBOOKS["05_baselines"] = [
    md("# 05 â€” Baselines\nRegression: global mean + persistence (prev1 as next). Classification: majority class + persistence (prev1 flag). Same 5-fold CV protocol."),
    code('''\
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from src.config import SPLITS_DIR, RESULTS_DIR, TARGETS_REG, CV_FOLDS, SEED
from src.evaluate import reg_metrics

train = pd.read_csv(SPLITS_DIR / "train.csv")
kf = KFold(CV_FOLDS, shuffle=True, random_state=SEED)
reg_rows = []
for target in TARGETS_REG:
    y = train[target].reset_index(drop=True)
    mean_scores, pers_scores = [], []
    for tr, va in kf.split(y):
        mean_scores.append(reg_metrics(y.iloc[va], np.full(len(va), y.iloc[tr].mean())))
        pers_scores.append(reg_metrics(y.iloc[va], train[target.replace("next_", "prev1_")].iloc[va]))
    row = {"target": target}
    for n, sc in [("mean", mean_scores), ("persistence", pers_scores)]:
        for metric in ("mae", "rmse", "r2"):
            row[f"{metric}_mean_{n}"] = round(float(np.mean([s[metric] for s in sc])), 4)
    reg_rows.append(row)
reg_base = pd.DataFrame(reg_rows)
display(reg_base)
reg_base.to_csv(RESULTS_DIR / "baselines_reg_cv.csv", index=False)
'''),
    md("## Classification baselines"),
    code('''\
from sklearn.model_selection import StratifiedKFold
from src.config import TARGET_CLF
from src.evaluate import clf_metrics

skf = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED)
y = train[TARGET_CLF].astype(int).reset_index(drop=True)
prev1 = train["prev1_irregular_flag"].astype(int).reset_index(drop=True)
maj, per = [], []
for tr, va in skf.split(train, y):
    maj.append(clf_metrics(y.iloc[va], np.zeros(len(va))))
    per.append(clf_metrics(y.iloc[va], prev1.iloc[va]))
clf_base = pd.DataFrame({
    "majority_class": {k: round(float(np.mean([m[k] for m in maj])), 4) for k in maj[0]},
    "persistence_prev1_flag": {k: round(float(np.mean([m[k] for m in per])), 4) for k in per[0]},
}).T.reset_index().rename(columns={"index": "model"}).assign(target=TARGET_CLF)
display(clf_base)
clf_base.to_csv(RESULTS_DIR / "baselines_clf_cv.csv", index=False)
'''),
]

NOTEBOOKS["06_model_comparison_regression"] = [
    md("# 06 â€” Regression Model Comparison\nRidge Â· RandomForest Â· XGBoost Â· LightGBM Â· SVR Â· KNN with default hyperparams, 5-fold CV on train split. Primary metric: MAE."),
    code('''\
import pandas as pd
from src.config import SPLITS_DIR, RESULTS_DIR, FEATURES, TARGETS_REG
from src.models import reg_models, REG_NAMES
from src.evaluate import evaluate_target, leaderboard_row

train = pd.read_csv(SPLITS_DIR / "train.csv")
X = train[FEATURES]
factories = reg_models()
rows = []
for target in TARGETS_REG:
    y = train[target]
    for name in REG_NAMES:
        scores = evaluate_target("reg", factories[name], X, y)
        row = leaderboard_row(name, target, "reg", scores)
        rows.append(row)
        print(target, name, "->", row["primary_metric"], row["primary_value"], flush=True)
reg_cv = pd.DataFrame(rows)
reg_cv.to_csv(RESULTS_DIR / "reg_baseline_cv.csv", index=False)
reg_cv.sort_values(["target", "mae_mean"])
'''),
]

NOTEBOOKS["07_model_comparison_classification"] = [
    md("# 07 â€” Classification Model Comparison\nLogReg(balanced) Â· RF Â· XGB(scale_pos_weight) Â· LGBM(is_unbalance) Â· SVC Â· KNN, 5-fold CV. Primary metric: PR-AUC."),
    code('''\
import pandas as pd
from src.config import SPLITS_DIR, RESULTS_DIR, FEATURES, TARGET_CLF
from src.models import clf_models, CLF_NAMES
from src.evaluate import evaluate_target, leaderboard_row

train = pd.read_csv(SPLITS_DIR / "train.csv")
X = train[FEATURES]
y = train[TARGET_CLF].astype(int)
factories = clf_models((y == 1).sum(), (y == 0).sum())
rows = []
for name in CLF_NAMES:
    scores = evaluate_target("clf", factories[name], X, y)
    row = leaderboard_row(name, TARGET_CLF, "clf", scores)
    rows.append(row)
    print(name, "-> pr_auc", row["pr_auc_mean"], "f1", row["f1_mean"], flush=True)
clf_cv = pd.DataFrame(rows)
clf_cv.to_csv(RESULTS_DIR / "clf_baseline_cv.csv", index=False)
clf_cv.sort_values("pr_auc_mean", ascending=False)
'''),
]

TUNING_NOTE = ("\nEach Optuna trial = one hyperparameter combo scored by 5-fold CV "
               "(MAE for regression / PR-AUC for classification). Fold-level scores feed "
               "`MedianPruner`; XGBoost/LightGBM use early stopping on an inner validation slice "
               "of each training fold. SVM models are tuned on a fixed 5000-row subsample for tractability.")

NOTEBOOKS["08_optuna_tuning_regression"] = [
    md("# 08 â€” Optuna Tuning: Regression" + TUNING_NOTE),
    code('''\
import json
import pandas as pd
from src.config import (SPLITS_DIR, RESULTS_DIR, OPTUNA_DIR, FEATURES,
                        TARGETS_REG, N_TRIALS_DEFAULT, N_TRIALS_SVM)
from src.tune import tune_model, make_est
from src.evaluate import evaluate_target, leaderboard_row
from src.models import REG_NAMES

train = pd.read_csv(SPLITS_DIR / "train.csv")
X = train[FEATURES]
N_TRIALS = {"svr": N_TRIALS_SVM}
summaries = {}
for target in TARGETS_REG:
    y = train[target]
    for name in REG_NAMES:
        n_trials = N_TRIALS.get(name, N_TRIALS_DEFAULT)
        print(f"tuning {name} | {target} | trials={n_trials}", flush=True)
        study, best = tune_model("reg", f"{name}_{target}", X, y, n_trials)
        q = dict(best["best_params"])
        scores = evaluate_target("reg", lambda q=q, name=name: make_est("reg", name, q), X, y)
        row = leaderboard_row(f"{name}_tuned", target, "reg", scores, extra={
            "optuna_best_value": best["best_value"],
            "best_params": json.dumps(best["best_params"])})
        summaries.setdefault(target, []).append(row)
        print("  best CV", best["metric"], round(best["best_value"], 4),
              "| re-scored mae", row["mae_mean"], flush=True)
for target, rows in summaries.items():
    pd.DataFrame(rows).to_csv(RESULTS_DIR / f"reg_tuned_cv_{target}.csv", index=False)
print("done")
'''),
]

NOTEBOOKS["09_optuna_tuning_classification"] = [
    md("# 09 â€” Optuna Tuning: Classification" + TUNING_NOTE),
    code('''\
import json
import pandas as pd
from src.config import (SPLITS_DIR, RESULTS_DIR, OPTUNA_DIR, FEATURES,
                        TARGET_CLF, N_TRIALS_DEFAULT, N_TRIALS_SVM)
from src.tune import tune_model, make_est
from src.evaluate import evaluate_target, leaderboard_row
from src.models import CLF_NAMES

train = pd.read_csv(SPLITS_DIR / "train.csv")
X = train[FEATURES]
y = train[TARGET_CLF].astype(int)
spw_full = float((y == 0).sum()) / max((y == 1).sum(), 1)
N_TRIALS = {"svc": N_TRIALS_SVM}
rows = []
for name in CLF_NAMES:
    n_trials = N_TRIALS.get(name, N_TRIALS_DEFAULT)
    print(f"tuning {name} | trials={n_trials}", flush=True)
    study, best = tune_model("clf", f"{name}_{TARGET_CLF}", X, y, n_trials)
    q = dict(best["best_params"])
    scores = evaluate_target(
        "clf", lambda q=q, name=name: make_est("clf", name, q, spw=spw_full), X, y)
    row = leaderboard_row(f"{name}_tuned", TARGET_CLF, "clf", scores, extra={
        "optuna_best_value": best["best_value"],
        "best_params": json.dumps(best["best_params"])})
    rows.append(row)
    print("  best CV", best["metric"], round(best["best_value"], 4),
          "| re-scored pr_auc", row["pr_auc_mean"], flush=True)
pd.DataFrame(rows).to_csv(RESULTS_DIR / f"clf_tuned_cv_{TARGET_CLF}.csv", index=False)
print("done")
'''),
]

NOTEBOOKS["10_leaderboard"] = [
    md("# 10 â€” Leaderboard\nAssembles baselines + default-hyperparam CV + tuned CV. Winner per target by primary metric (MAE min / PR-AUC max)."),
    code('''\
import json
import numpy as np
import pandas as pd
from src.config import RESULTS_DIR, TARGET_CLF

REG_COLS = ["model", "target", "mae_mean", "rmse_mean", "r2_mean"]
CLF_COLS = ["model", "target", "pr_auc_mean", "pr_auc_std", "f1_mean", "roc_auc_mean"]

base_reg = pd.read_csv(RESULTS_DIR / "baselines_reg_cv.csv")
rows = []
for _, r in base_reg.iterrows():
    rows.append({"model": "baseline_mean", "target": r["target"],
                 "mae_mean": r["mae_mean_mean"], "rmse_mean": r["rmse_mean_mean"], "r2_mean": r["r2_mean_mean"]})
    rows.append({"model": "baseline_persistence", "target": r["target"],
                 "mae_mean": r["mae_mean_persistence"], "rmse_mean": r["rmse_mean_persistence"], "r2_mean": r["r2_mean_persistence"]})
frames = [pd.DataFrame(rows)]
frames.append(pd.read_csv(RESULTS_DIR / "reg_baseline_cv.csv")[REG_COLS])
tuned_paths = sorted(RESULTS_DIR.glob("reg_tuned_cv_*.csv"))
if tuned_paths:
    frames.append(pd.concat([pd.read_csv(p)[REG_COLS] for p in tuned_paths], ignore_index=True))
leaderboard_reg = pd.concat(frames, ignore_index=True)
leaderboard_reg.to_csv(RESULTS_DIR / "leaderboard_regression.csv", index=False)
display(leaderboard_reg.sort_values(["target", "mae_mean"]))
'''),
    code('''\
winners_reg = (leaderboard_reg.sort_values("mae_mean").groupby("target").first()[["model", "mae_mean"]])
print("REGRESSION WINNERS")
display(winners_reg)
'''),
    md("## Classification leaderboard"),
    code('''\
base_clf = pd.read_csv(RESULTS_DIR / "baselines_clf_cv.csv")
base_clf = base_clf.rename(columns={c: f"{c}_mean" for c in ["pr_auc", "f1", "roc_auc"]})
base_clf["target"] = TARGET_CLF
frames_c = [base_clf.reindex(columns=CLF_COLS)]
frames_c.append(pd.read_csv(RESULTS_DIR / "clf_baseline_cv.csv")[CLF_COLS])
tuned_path = RESULTS_DIR / f"clf_tuned_cv_{TARGET_CLF}.csv"
if tuned_path.exists():
    frames_c.append(pd.read_csv(tuned_path)[CLF_COLS])
leaderboard_clf = pd.concat(frames_c, ignore_index=True)
leaderboard_clf.to_csv(RESULTS_DIR / "leaderboard_classification.csv", index=False)
display(leaderboard_clf.sort_values("pr_auc_mean", ascending=False))

winner_clf = leaderboard_clf.sort_values("pr_auc_mean", ascending=False).iloc[0]
print("CLASSIFICATION WINNER:", winner_clf["model"], "pr_auc:", winner_clf["pr_auc_mean"])
with open(RESULTS_DIR / "winners.json", "w") as f:
    json.dump({
        "regression": winners_reg.reset_index().to_dict(orient="records"),
        "classification": {k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in winner_clf.items()},
    }, f, indent=2)
'''),
]

NOTEBOOKS["11_shap_analysis"] = [
    md("# 11 â€” SHAP Analysis\nTreeSHAP on the winning tree-based model per target (refit on train+val). Sanity checks: cycle history should dominate `next_cycle_length`; irregular history should dominate `next_is_irregular`. Linear/kernel/instance-based winners fall back to skipped SHAP with a note."),
    code('''\
import json
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
from src.config import (SPLITS_DIR, RESULTS_DIR, FIGURES_DIR, MODELS_DIR,
                        FEATURES, TARGET_CLF)
from src.tune import make_est
from src.preprocessing import build_preprocessor

trainval = pd.concat([pd.read_csv(SPLITS_DIR / "train.csv"), pd.read_csv(SPLITS_DIR / "val.csv")], ignore_index=True)
winners = json.load(open(RESULTS_DIR / "winners.json"))

def default_est(kind, name, spw=1.0):
    if kind == "reg":
        table = {
            "ridge": lambda: Ridge(random_state=42),
            "random_forest": lambda: RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
            "xgboost": lambda: XGBRegressor(n_estimators=500, random_state=42, n_jobs=-1, tree_method="hist", eval_metric="mae", verbosity=0),
            "lightgbm": lambda: LGBMRegressor(n_estimators=500, random_state=42, n_jobs=-1, verbose=-1),
            "svr": lambda: SVR(),
            "knn": lambda: KNeighborsRegressor(),
        }
    else:
        table = {
            "logistic_regression": lambda: LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42),
            "random_forest": lambda: RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1),
            "xgboost": lambda: XGBClassifier(n_estimators=500, scale_pos_weight=spw, random_state=42, n_jobs=-1, tree_method="hist", eval_metric="aucpr", verbosity=0),
            "lightgbm": lambda: LGBMClassifier(n_estimators=500, is_unbalance=True, random_state=42, n_jobs=-1, verbose=-1),
            "svc": lambda: SVC(class_weight="balanced", probability=True, random_state=42),
            "knn": lambda: KNeighborsClassifier(),
        }
    return table[name]()

TREE_MODELS = {"xgboost", "lightgbm", "random_forest"}

def fit_and_shap(kind, winner_name, target):
    base = winner_name.replace("_tuned", "")
    spw = 1.0
    y = trainval[target]
    if kind == "clf":
        yi = y.astype(int)
        spw = float((yi == 0).sum()) / max((yi == 1).sum(), 1)
    params = {}
    if winner_name.endswith("_tuned"):
        tuned_file = RESULTS_DIR / "optuna" / f"{kind}_{base}_{target}_best.json"
        if tuned_file.exists():
            params = json.load(open(tuned_file))["best_params"]
    est = make_est(kind, base, params, spw=spw)
    prep = build_preprocessor()
    Xt = prep.fit_transform(trainval[FEATURES])
    est.fit(Xt, yi if kind == "clf" else y)
    sample = shap.sample(Xt, 2000, random_state=42)
    explainer = shap.TreeExplainer(est)
    sv = explainer.shap_values(sample)
    vals = sv[1] if isinstance(sv, list) else sv
    names = [n.split("__")[-1] for n in prep.get_feature_names_out()]
    plt.figure(figsize=(9, 7))
    shap.summary_plot(vals, sample, feature_names=names, show=False)
    plt.title(f"SHAP - {target}")
    plt.tight_layout(); plt.savefig(FIGURES_DIR / f"shap_{target}.png", dpi=150); plt.show()
    plt.close("all")
    top = sorted(zip(names, np.abs(vals).mean(axis=0)), key=lambda t: -t[1])[:10]
    joblib.dump({"prep": prep, "model": est}, MODELS_DIR / f"final_{target}.joblib")
    return [{"feature": f, "mean_abs_shap": round(float(v), 5)} for f, v in top]
'''),
    md("## Compute SHAP per winning model"),
    code('''\
import joblib

shap_top = {}
skipped = {}
for w in winners["regression"]:
    target, winner_name = w["target"], w["model"]
    if winner_name.replace("_tuned", "") in TREE_MODELS:
        shap_top[target] = fit_and_shap("reg", winner_name, target)
        print(target, "top:", shap_top[target][:3], flush=True)
    else:
        skipped[target] = f"{winner_name} is not tree-based; SHAP summary skipped"
        print(skipped[target])
winner_name = winners["classification"]["model"]
if winner_name.replace("_tuned", "") in TREE_MODELS:
    shap_top[TARGET_CLF] = fit_and_shap("clf", winner_name, TARGET_CLF)
    print(TARGET_CLF, "top:", shap_top[TARGET_CLF][:3], flush=True)
else:
    skipped[TARGET_CLF] = f"{winner_name} is not tree-based; SHAP summary skipped"
json.dump({"top_features": shap_top, "skipped": skipped}, open(RESULTS_DIR / "shap_top_features.json", "w"), indent=2)
'''),
    md("## Sanity checks"),
    code('''\
checks = []
def add_check(target, label, keyword):
    if target in shap_top:
        tops = [d["feature"] for d in shap_top[target]][:5]
        checks.append({"check": label, "pass": any(keyword in t for t in tops), "top5": tops})

add_check("next_cycle_length", "cycle-length history dominates next_cycle_length", "cycle")
add_check("next_period_length", "period history dominates next_period_length", "period")
add_check(TARGET_CLF, "irregular history drives irregular classifier", "irregular")
pd.DataFrame(checks)
'''),
]

NOTEBOOKS["12_final_test_evaluation"] = [
    md("# 12 â€” Final Test Evaluation\nRetrain each winner on train+val, evaluate ONCE on the untouched test set. No tuning after this point."),
    code('''\
import json
import joblib
import pandas as pd
from src.config import SPLITS_DIR, RESULTS_DIR, MODELS_DIR, FEATURES, TARGETS_REG, TARGET_CLF
from src.tune import make_est
from src.preprocessing import build_preprocessor
from src.evaluate import reg_metrics, clf_metrics, confusion

trainval = pd.concat([pd.read_csv(SPLITS_DIR / "train.csv"), pd.read_csv(SPLITS_DIR / "val.csv")], ignore_index=True)
test = pd.read_csv(SPLITS_DIR / "test.csv")
winners = json.load(open(RESULTS_DIR / "winners.json"))

def fit_eval(kind, winner_name, target):
    base = winner_name.replace("_tuned", "")
    y_tr = trainval[target]
    spw = 1.0
    if kind == "clf":
        y_tr = y_tr.astype(int)
        spw = float((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1)
    params = {}
    if winner_name.endswith("_tuned"):
        tuned_file = RESULTS_DIR / "optuna" / f"{kind}_{base}_{target}_best.json"
        if tuned_file.exists():
            params = json.load(open(tuned_file))["best_params"]
    est = make_est(kind, base, params, spw=spw)
    prep = build_preprocessor()
    Xt = prep.fit_transform(trainval[FEATURES])
    est.fit(Xt, y_tr)
    joblib.dump({"prep": prep, "model": est}, MODELS_DIR / f"final_{target}.joblib")
    Xte = prep.transform(test[FEATURES])
    if kind == "reg":
        return reg_metrics(test[target], est.predict(Xte))
    proba = est.predict_proba(Xte)[:, 1] if hasattr(est, "predict_proba") else est.decision_function(Xte)
    m = clf_metrics(test[target].astype(int), proba)
    m["confusion_matrix"] = confusion(test[target], proba)
    return m

final = {}
for w in winners["regression"]:
    final[w["target"]] = fit_eval("reg", w["model"], w["target"])
    print(w["target"], w["model"], "->", {k: round(v, 4) for k, v in final[w["target"]].items()}, flush=True)

final[TARGET_CLF] = fit_eval("clf", winners["classification"]["model"], TARGET_CLF)
print(TARGET_CLF, winners["classification"]["model"], "->",
      {k: round(v, 4) for k, v in final[TARGET_CLF].items() if k != "confusion_matrix"},
      "| CM:", final[TARGET_CLF]["confusion_matrix"])

flat = {f"{t}.{k}": v for t, m in final.items() for k, v in m.items()}
json.dump(flat, open(RESULTS_DIR / "final_test_results.json", "w"), indent=2)
rows = [{"target": t, **{k: (round(v, 4) if isinstance(v, float) else str(v)) for k, v in m.items()}} for t, m in final.items()]
pd.DataFrame(rows).to_csv(RESULTS_DIR / "final_test_results.csv", index=False)
pd.DataFrame(rows)
'''),
]

NOTEBOOKS["13_summary_and_report"] = [
    md("# 13 â€” Summary & Consolidated Pipeline\nWrites `reports/results_summary.md`; consolidated runnable pipeline lives in `src/pipeline.py`."),
    code('''\
import json
from pathlib import Path
import pandas as pd
from src.config import RESULTS_DIR

val = json.load(open(RESULTS_DIR / "validation_report.json"))
final = json.load(open(RESULTS_DIR / "final_test_results.json"))
lb_r = pd.read_csv(RESULTS_DIR / "leaderboard_regression.csv")
lb_c = pd.read_csv(RESULTS_DIR / "leaderboard_classification.csv")

lines = ["# Results Summary", "",
         f"- Rows kept after validation: {val['rows_kept']} (dropped {val['rows_dropped']})",
         "- i.i.d. assumption: no user id/timestamp column exists; if rows are repeated measures, mild leakage could inflate CV slightly.",
         "", "## Regression leaderboard (CV MAE, lower is better)", "",
         lb_r.sort_values(["target", "mae_mean"]).to_markdown(index=False), "",
         "## Classification leaderboard (CV PR-AUC, higher is better)", "",
         lb_c.sort_values("pr_auc_mean", ascending=False).to_markdown(index=False), "",
         "## Final held-out test results", ""]
for t, m in final.items():
    pretty = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}
    lines += [f"### {t}", "", "```json", json.dumps(pretty, indent=2), "```", ""]
lines += ["## Notes", "",
          "- Test set evaluated exactly once (notebook 12).",
          "- SVM tuning used a fixed 5000-row subsample for tractability.",
          "- Class imbalance handled via class weights / scale_pos_weight / is_unbalance; PR-AUC is the primary classification metric."]
summary = "\\n".join(lines)
Path("..", "reports", "results_summary.md").write_text(summary, encoding="utf-8")
print(summary[:1500])
'''),
    md("## Verify consolidated pipeline (`src/pipeline.py`)"),
    code('''\
from pathlib import Path
p = Path("..") / "src" / "pipeline.py"
print(p.resolve(), "exists:", p.exists())
print(p.read_text(encoding="utf-8")[:600] if p.exists() else "pipeline.py will be created next")
'''),
]

KERNEL_META = {
    "kernelspec": {"display_name": "Python 3.13 (mc-cycle)", "language": "python", "name": "mc-cycle"},
    "language_info": {"name": "python"},
}

for name, cells in NOTEBOOKS.items():
    nbk = nbf.v4.new_notebook(cells=[cells[0], code(SETUP)] + cells[1:], metadata=dict(KERNEL_META))
    path = f"E:/NAVYA_MODEL/notebooks/{name}.ipynb"
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nbk, f)
    print("wrote", path)
