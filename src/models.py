from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

from src.config import SEED


def reg_models():
    return {
        "ridge": lambda: Ridge(random_state=SEED),
        "random_forest": lambda: RandomForestRegressor(
            n_estimators=300, random_state=SEED, n_jobs=-1),
        "xgboost": lambda: XGBRegressor(
            n_estimators=500,
            random_state=SEED, n_jobs=-1, tree_method="hist", eval_metric="mae", verbosity=0),
        "lightgbm": lambda: LGBMRegressor(
            n_estimators=500, random_state=SEED, n_jobs=-1, verbose=-1),
        "svr": lambda: SVR(),
        "knn": lambda: KNeighborsRegressor(),
    }


def clf_models(n_pos, n_neg):
    spw = n_neg / max(n_pos, 1)
    return {
        "logistic_regression": lambda: LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=SEED),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=300, class_weight="balanced",
            random_state=SEED, n_jobs=-1),
        "xgboost": lambda: XGBClassifier(
            n_estimators=500, scale_pos_weight=spw,
            random_state=SEED, n_jobs=-1, tree_method="hist", eval_metric="aucpr", verbosity=0),
        "lightgbm": lambda: LGBMClassifier(
            n_estimators=500, is_unbalance=True,
            random_state=SEED, n_jobs=-1, verbose=-1),
        "svc": lambda: SVC(class_weight="balanced", probability=True, random_state=SEED),
        "knn": lambda: KNeighborsClassifier(),
    }


REG_NAMES = ["ridge", "random_forest", "xgboost", "lightgbm", "svr", "knn"]
CLF_NAMES = [
    "logistic_regression", "random_forest", "xgboost", "lightgbm", "svc", "knn",
]
