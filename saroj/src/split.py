from sklearn.model_selection import train_test_split

from src.config import SEED, SPLITS_DIR, TARGET_CLF, TEST_SIZE, VAL_SIZE


def make_splits(df):
    y_strat = df[TARGET_CLF].astype(int)
    train_val, test = train_test_split(
        df, test_size=TEST_SIZE, random_state=SEED, stratify=y_strat
    )
    val_rel = VAL_SIZE / (1.0 - TEST_SIZE)
    train, val = train_test_split(
        train_val, test_size=val_rel, random_state=SEED,
        stratify=train_val[TARGET_CLF].astype(int),
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def run_split(engineered_csv=None):
    import pandas as pd
    df = pd.read_csv(engineered_csv or SPLITS_DIR.parent / "engineered.csv")
    train, val, test = make_splits(df)
    train.to_csv(SPLITS_DIR / "train.csv", index=False)
    val.to_csv(SPLITS_DIR / "val.csv", index=False)
    test.to_csv(SPLITS_DIR / "test.csv", index=False)
    return train, val, test
