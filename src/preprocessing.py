import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.config import BOOL_COLS, FEATURES, ONEHOT_COLS, ORDINAL_COLS, ORDINAL_ORDER

ORDINAL_FEATURES = [c for c in FEATURES if c in ORDINAL_COLS]
ONEHOT_FEATURES = [c for c in FEATURES if c in ONEHOT_COLS]
NUMERIC_FEATURES = [
    c for c in FEATURES
    if c not in ORDINAL_FEATURES + ONEHOT_FEATURES + BOOL_COLS
]


def build_preprocessor():
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    ordinal = OrdinalEncoder(
        categories=[ORDINAL_ORDER[c] for c in ORDINAL_FEATURES],
        handle_unknown="use_encoded_value", unknown_value=-1,
    )
    onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("ord", ordinal, ORDINAL_FEATURES),
            ("cat", onehot, ONEHOT_FEATURES),
            ("bool", "passthrough", BOOL_COLS),
        ],
        remainder="drop",
    )
