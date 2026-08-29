import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    mean_absolute_error, mean_squared_error, r2_score, roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.config import CV_FOLDS, SEED, TARGET_CLF
from src.preprocessing import build_preprocessor


def reg_metrics(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def clf_metrics(y_true, y_proba, threshold=0.5):
    y_hat = (np.asarray(y_proba) >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_hat, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def make_pipeline(estimator):
    return Pipeline([("prep", build_preprocessor()), ("model", estimator)])


def _agg(rows):
    df = pd.DataFrame(rows)
    return {c: {"mean": float(df[c].mean()), "std": float(df[c].std())} for c in df.columns}


def cv_regression(estimator_factory, X, y, n_splits=CV_FOLDS):
    return _agg(cv_regression_folds(estimator_factory, X, y, n_splits))


def cv_regression_folds(estimator_factory, X, y, n_splits=CV_FOLDS):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    scores = []
    for tr, va in kf.split(X):
        pipe = make_pipeline(estimator_factory())
        pipe.fit(X.iloc[tr], y.iloc[tr])
        pred = pipe.predict(X.iloc[va])
        scores.append(reg_metrics(y.iloc[va], pred))
    return scores


def cv_classification(estimator_factory, X, y, n_splits=CV_FOLDS):
    return _agg(cv_classification_folds(estimator_factory, X, y, n_splits))


def _proba(estimator, X_va):
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X_va)[:, 1]
    return estimator.decision_function(X_va)


def cv_classification_folds(estimator_factory, X, y, n_splits=CV_FOLDS):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    scores = []
    for tr, va in skf.split(X, y):
        pipe = make_pipeline(estimator_factory())
        pipe.fit(X.iloc[tr], y.iloc[tr].astype(int))
        proba = _proba(pipe, X.iloc[va])
        scores.append(clf_metrics(y.iloc[va].astype(int), proba))
    return scores


def evaluate_target(kind, estimator_factory, X, y):
    if kind == "reg":
        return cv_regression(estimator_factory, X, y)
    return cv_classification(estimator_factory, X, y)


def leaderboard_row(name, target, kind, cv_scores, extra=None):
    row = {"model": name, "target": target}
    for metric, agg in cv_scores.items():
        row[f"{metric}_mean"] = round(agg["mean"], 5)
        row[f"{metric}_std"] = round(agg["std"], 5)
    if extra:
        row.update(extra)
    if kind == "clf":
        row["primary_metric"] = "pr_auc"
        row["primary_value"] = row.get("pr_auc_mean")
    else:
        row["primary_metric"] = "mae"
        row["primary_value"] = row.get("mae_mean")
    return row


def confusion(y_true, y_proba, threshold=0.5):
    return confusion_matrix(
        np.asarray(y_true).astype(int),
        (np.asarray(y_proba) >= threshold).astype(int),
    ).tolist()


PRIMARY = {
    "reg": ("mae", "min"),
    "clf": ("pr_auc", "max"),
}


def better(kind, new, old):
    metric, direction = PRIMARY[kind]
    if old is None:
        return True
    return new < old if direction == "min" else new > old
