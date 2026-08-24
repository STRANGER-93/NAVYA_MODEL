# Menstrual Cycle Outcome Prediction

Production-quality ML pipeline predicting three targets from one CSV:

| Target | Type | Primary metric |
|---|---|---|
| `next_cycle_length` | regression | MAE |
| `next_period_length` | regression | MAE |
| `next_is_irregular` | binary classification | PR-AUC |

## Setup

```
python -m venv venv
venv\Scripts\python -m pip install -r requirements.txt
venv\Scripts\python -m ipykernel install --user --name mc-cycle
```

## Run

Execute notebooks `notebooks/01` … `notebooks/13` in order with the `mc-cycle`
kernel (or `jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=mc-cycle notebooks/<nb>.ipynb`).
Dependency chain:

```
01 -> 02 -> 03 -> 04 -> (05, 06, 07) -> (08, 09) -> 10 -> 11 -> 12 -> 13
```

One-shot end-to-end alternative (data prep + final models + test metrics):

```
venv\Scripts\python -m src.pipeline
```

## Layout

- `src/` — reusable modules (`config`, `validation`, `features`, `preprocessing`,
  `split`, `models`, `evaluate`, `tune`, `pipeline`)
- `notebooks/` — the 13-step analysis
- `data/processed/` — cleaned data, engineered features, train/val/test splits
- `reports/metrics/` — validation report, CV leaderboards, Optuna studies, final results
- `reports/figures/` — EDA + SHAP plots
- `models/` — fitted final model bundles (`joblib`)

## Protocol

- 70/15/15 stratified split (seed 42); rows treated i.i.d. (no user id/timestamp exists)
- Model selection by 5-fold CV on the train split only; winner retrained on
  train+val; test set evaluated exactly once (notebook 12)
- Optuna TPE tuning with median pruning; XGBoost/LightGBM use early stopping on an
  inner slice of each training fold
- Class imbalance handled via class weights / `scale_pos_weight` / `is_unbalance`;
  plain accuracy is never used for the classifier
