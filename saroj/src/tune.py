import json

import numpy as np
import optuna
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

from src.config import (
    CV_FOLDS, OPTUNA_DIR, SEED, TARGET_CLF,
)
from src.evaluate import clf_metrics, reg_metrics
from src.preprocessing import build_preprocessor

EARLY_STOP_MODELS = {"xgboost", "lightgbm"}
SVM_MODELS = {"svr", "svc"}
SVM_TUNING_SUBSAMPLE = 5000


def make_est(kind, name, q, spw=1.0, es=False):
    es_rounds = 50 if es else None
    if kind == "reg":
        table = {
            "ridge": lambda: Ridge(alpha=q["alpha"], random_state=SEED),
            "random_forest": lambda: RandomForestRegressor(
                n_estimators=q["n_estimators"], max_depth=q["max_depth"],
                min_samples_split=q["min_samples_split"],
                min_samples_leaf=q["min_samples_leaf"],
                max_features=q["max_features"], random_state=SEED, n_jobs=-1),
            "xgboost": lambda: XGBRegressor(
                n_estimators=q["n_estimators"], learning_rate=q["learning_rate"],
                max_depth=q["max_depth"], subsample=q["subsample"],
                colsample_bytree=q["colsample_bytree"],
                min_child_weight=q["min_child_weight"],
                reg_lambda=q["reg_lambda"], early_stopping_rounds=es_rounds,
                random_state=SEED, n_jobs=-1, tree_method="hist",
                eval_metric="mae", verbosity=0),
            "lightgbm": lambda: LGBMRegressor(
                n_estimators=q["n_estimators"], learning_rate=q["learning_rate"],
                num_leaves=q["num_leaves"], subsample=q["subsample"],
                colsample_bytree=q["colsample_bytree"],
                min_child_samples=q["min_child_samples"],
                reg_lambda=q["reg_lambda"],
                random_state=SEED, n_jobs=-1, verbose=-1),
            "svr": lambda: SVR(C=q["C"], gamma=q["gamma"]),
            "knn": lambda: KNeighborsRegressor(
                n_neighbors=q["n_neighbors"], weights=q["weights"], p=q["p"]),
        }
    else:
        table = {
            "logistic_regression": lambda: LogisticRegression(
                C=q["C"], class_weight="balanced", max_iter=2000, random_state=SEED),
            "random_forest": lambda: RandomForestClassifier(
                n_estimators=q["n_estimators"], max_depth=q["max_depth"],
                min_samples_split=q["min_samples_split"],
                min_samples_leaf=q["min_samples_leaf"],
                max_features=q["max_features"], class_weight="balanced",
                random_state=SEED, n_jobs=-1),
            "xgboost": lambda: XGBClassifier(
                n_estimators=q["n_estimators"], learning_rate=q["learning_rate"],
                max_depth=q["max_depth"], subsample=q["subsample"],
                colsample_bytree=q["colsample_bytree"],
                min_child_weight=q["min_child_weight"],
                reg_lambda=q["reg_lambda"], scale_pos_weight=spw,
                early_stopping_rounds=es_rounds,
                random_state=SEED, n_jobs=-1, tree_method="hist",
                eval_metric="aucpr", verbosity=0),
            "lightgbm": lambda: LGBMClassifier(
                n_estimators=q["n_estimators"], learning_rate=q["learning_rate"],
                num_leaves=q["num_leaves"], subsample=q["subsample"],
                colsample_bytree=q["colsample_bytree"],
                min_child_samples=q["min_child_samples"],
                reg_lambda=q["reg_lambda"], is_unbalance=True,
                random_state=SEED, n_jobs=-1, verbose=-1),
            "svc": lambda: SVC(C=q["C"], gamma=q["gamma"], class_weight="balanced",
                               random_state=SEED),
            "knn": lambda: KNeighborsClassifier(
                n_neighbors=q["n_neighbors"], weights=q["weights"], p=q["p"]),
        }
    return table[name]()


def suggest(trial, name):
    q = {}
    if name == "ridge":
        q["alpha"] = trial.suggest_float("alpha", 1e-3, 1e3, log=True)
    elif name in ("random_forest",):
        q["n_estimators"] = trial.suggest_int("n_estimators", 150, 450, step=50)
        q["max_depth"] = trial.suggest_categorical(
            "max_depth", [None] + list(range(4, 31, 2)))
        q["min_samples_split"] = trial.suggest_int("min_samples_split", 2, 20)
        q["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 1, 10)
        q["max_features"] = trial.suggest_categorical(
            "max_features", ["sqrt", "log2", 0.3, 0.5, 1.0])
    elif name in ("xgboost", "lightgbm"):
        q["n_estimators"] = trial.suggest_int("n_estimators", 100, 900, step=100)
        q["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        if name == "xgboost":
            q["max_depth"] = trial.suggest_int("max_depth", 3, 10)
            q["min_child_weight"] = trial.suggest_int("min_child_weight", 1, 10)
        else:
            q["num_leaves"] = trial.suggest_int("num_leaves", 15, 127)
            q["min_child_samples"] = trial.suggest_int("min_child_samples", 5, 60)
        q["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)
        q["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.6, 1.0)
        q["reg_lambda"] = trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True)
    elif name in ("svr", "svc"):
        q["C"] = trial.suggest_float("C", 1e-2, 100.0, log=True)
        q["gamma"] = trial.suggest_categorical("gamma", ["scale", "auto"])
    elif name == "logistic_regression":
        q["C"] = trial.suggest_float("C", 1e-3, 100.0, log=True)
    elif name == "knn":
        q["n_neighbors"] = trial.suggest_int("n_neighbors", 3, 60)
        q["weights"] = trial.suggest_categorical("weights", ["uniform", "distance"])
        q["p"] = trial.suggest_int("p", 1, 2)
    return q


def _scores(kind, y_true, out):
    if kind == "reg":
        m = reg_metrics(y_true, out)
        return m, m["mae"]
    m = clf_metrics(y_true, out)
    return m, m["pr_auc"]


def _run_fold(kind, name, q, spw, X, y, tr_idx, va_idx):
    prep = build_preprocessor()
    est = make_est(kind, name, q, spw, es=(name in EARLY_STOP_MODELS))
    X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
    if name in EARLY_STOP_MODELS:
        strat = y_tr.astype(int) if kind == "clf" else None
        itr, iev = train_test_split(
            np.arange(len(X_tr)), test_size=0.15, random_state=SEED, stratify=strat)
        Xtr_in, ytr_in = X_tr.iloc[itr], y_tr.iloc[itr]
        Xev, yev = X_tr.iloc[iev], y_tr.iloc[iev]
        Xt = prep.fit_transform(Xtr_in)
        fit_kwargs = {"eval_set": [(prep.transform(Xev), yev)]}
        if name == "lightgbm":
            from lightgbm import early_stopping as lgb_es
            fit_kwargs["callbacks"] = [lgb_es(50, verbose=False)]
        est.fit(Xt, ytr_in, **fit_kwargs)
    else:
        Xt = prep.fit_transform(X_tr)
        est.fit(Xt, y_tr)
    Xv = prep.transform(X.iloc[va_idx])
    if kind == "clf":
        out = est.predict_proba(Xv)[:, 1] if hasattr(est, "predict_proba") \
            else est.decision_function(Xv)
    else:
        out = est.predict(Xv)
    y_true = y.iloc[va_idx]
    return _scores(kind, y_true.astype(int) if kind == "clf" else y_true, out)


def objective_fn(kind, name, X, y, spw=1.0):
    def objective(trial):
        q = suggest(trial, name)
        if kind == "reg":
            splitter = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
            splits = list(splitter.split(X))
        else:
            splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
            splits = list(splitter.split(X, y))
        metrics_all = []
        for step, (tr, va) in enumerate(splits):
            metrics, prune_val = _run_fold(kind, name, q, spw, X, y, tr, va)
            metrics_all.append(metrics)
            trial.report(prune_val, step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        mean_metrics = {
            k: float(np.mean([m[k] for m in metrics_all])) for k in metrics_all[0]
        }
        return mean_metrics["mae"] if kind == "reg" else mean_metrics["pr_auc"]
    return objective


def tune_model(kind, name, X, y, n_trials, seed=SEED, study_name=None):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    direction = "minimize" if kind == "reg" else "maximize"
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=2)
    sname = study_name or name
    storage = f"sqlite:///{OPTUNA_DIR / f'{kind}_{sname}.db'}"
    study = optuna.create_study(
        study_name=sname, storage=storage, direction=direction,
        sampler=sampler, pruner=pruner, load_if_exists=True,
    )
    if kind == "clf":
        yi = y.astype(int)
        spw = float((yi == 0).sum()) / max((yi == 1).sum(), 1)
    else:
        spw = 1.0
    if name in SVM_MODELS and len(X) > SVM_TUNING_SUBSAMPLE:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X), size=SVM_TUNING_SUBSAMPLE, replace=False)
        X_t, y_t = X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)
    else:
        X_t, y_t = X, y
    study.optimize(objective_fn(kind, name, X_t, y_t, spw), n_trials=n_trials)
    best = {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "direction": direction,
        "n_trials": len(study.trials),
        "metric": "mae" if kind == "reg" else "pr_auc",
    }
    with open(OPTUNA_DIR / f"{kind}_{sname}_best.json", "w") as f:
        json.dump(best, f, indent=2, default=str)
    return study, best
