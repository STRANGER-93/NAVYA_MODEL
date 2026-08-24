import json

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    FEATURES, MODELS_DIR, RESULTS_DIR, SEED, SPLITS_DIR,
    TARGET_CLF, TARGETS_REG, TEST_SIZE, VAL_SIZE,
)
from src.evaluate import clf_metrics, confusion, reg_metrics
from src.features import engineer
from src.preprocessing import build_preprocessor
from src.split import make_splits
from src.tune import make_est
from src.validation import clean, load_raw, validate


def load_and_prepare():
    raw = load_raw()
    report = validate(raw)
    cleaned, _ = clean(raw, report)
    engineered = engineer(cleaned)
    return engineered, report


def get_splits(engineered=None):
    if engineered is None:
        engineered, _ = load_and_prepare()
    return make_splits(engineered)


def _load_params(kind, base, target):
    path = RESULTS_DIR / "optuna" / f"{kind}_{base}_{target}_best.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)["best_params"]
    return {}


def fit_final_models(trainval=None):
    if trainval is None:
        trainval = pd.concat(
            [pd.read_csv(SPLITS_DIR / "train.csv"), pd.read_csv(SPLITS_DIR / "val.csv")],
            ignore_index=True,
        )
    fitted = {}
    for target in TARGETS_REG:
        est = make_est("reg", "xgboost", _load_params("reg", "xgboost", target))
        fitted[target] = _fit_one(est, trainval, target, "reg")
    yi = trainval[TARGET_CLF].astype(int)
    spw = float((yi == 0).sum()) / max((yi == 1).sum(), 1)
    est = make_est("clf", "lightgbm", _load_params("clf", "lightgbm", TARGET_CLF), spw=spw)
    fitted[TARGET_CLF] = _fit_one(est, trainval, TARGET_CLF, "clf")
    return fitted


def _fit_one(est, data, target, kind):
    prep = build_preprocessor()
    y = data[target]
    if kind == "clf":
        y = y.astype(int)
    Xt = prep.fit_transform(data[FEATURES])
    est.fit(Xt, y)
    return {"prep": prep, "model": est}


def predict(fitted, target, df):
    X = fitted[target]["prep"].transform(df[FEATURES])
    model = fitted[target]["model"]
    if target == TARGET_CLF:
        return model.predict_proba(X)[:, 1]
    return model.predict(X)


def evaluate_on_test(fitted, test):
    results = {}
    for target in TARGETS_REG:
        results[target] = reg_metrics(test[target], predict(fitted, target, test))
    proba = predict(fitted, TARGET_CLF, test)
    m = clf_metrics(test[TARGET_CLF].astype(int), proba)
    m["confusion_matrix"] = confusion(test[TARGET_CLF], proba)
    results[TARGET_CLF] = m
    return results


def save_models(fitted):
    for target, bundle in fitted.items():
        joblib.dump(bundle, MODELS_DIR / f"final_{target}.joblib")


def main():
    engineered, report = load_and_prepare()
    train, val, test = make_splits(engineered)
    trainval = pd.concat([train, val], ignore_index=True)
    fitted = fit_final_models(trainval)
    results = evaluate_on_test(fitted, test)
    save_models(fitted)
    print(json.dumps({t: {k: round(v, 4) if isinstance(v, float) else v
                          for k, v in r.items()} for t, r in results.items()}, indent=2))
    return results


if __name__ == "__main__":
    main()
