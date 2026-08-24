# ML Pipeline Plan: Menstrual Cycle Outcome Prediction

## 1. Goal

Build a complete, production-quality ML pipeline that predicts three targets from a single CSV:

| Target | Type | Metric |
|---|---|---|
| `next_cycle_length` | regression | MAE / RMSE / R² |
| `next_period_length` | regression | MAE / RMSE / R² |
| `next_is_irregular` | binary classification | F1 / PR-AUC / confusion matrix (never plain accuracy) |

## 2. Environment (done)

- venv created at `/workspace/.venv` (Python 3.11)
- Jupyter kernel registered: `mc-cycle`
- Installed: pandas, numpy, scikit-learn, xgboost, lightgbm, optuna, shap, matplotlib, seaborn, scipy, jupyter, ipykernel

## 3. Dataset facts (verified)

- 20,000 rows x 30 cols, **0 missing values**, **0 duplicate rows** (validation step still built in)
- `height` is in **feet** (range 4.4–6.0); `weight` in kg; `bmi` consistent with `weight/(height_m)²`
- `cycle_length` range 15–50 (plausible bounds OK)
- `period_length` range 1–9
- Categoricals: `medication_contraceptive` (No/Yes/Not Sure), `stress` (Very Low..Very High, ordinal), `exercise` (Never..5+ days/week, ordinal)

## 4. Key assumptions (stated, not asked)

1. **No user ID / timestamp column exists** → rows are treated as **i.i.d.**. Cannot use GroupKFold or time-based split. Risk: if rows were repeated measures of the same person, mild leakage could inflate CV scores slightly; we accept this and note it in the report.
2. **Split**: 70% train / 15% validation / 15% test, fixed `random_state=42`, stratified on `next_is_irregular` for the classification target (train/test splits are identical across targets so models are comparable).
3. **Feature engineering is target-free** (mean/std/slope/range of prevN_* + interactions) → computed once, no target leakage. Only imputation/scaling/encoding are fit inside CV folds.
4. **Model selection protocol** (per user's spec):
   - Tune + compare using **5-fold CV on the 70% train split only**
   - Pick winner by mean CV metric
   - Retrain final model on **train + validation**
   - Evaluate **once** on the held-out test set. No tuning after.

## 5. Validation rules (notebook 01)

Flag (report count + rows) any row where:
- `prev*/next_cycle_length` < 10 or > 90 (outside plausible physiology)
- `prev*/next_period_length` < 1 or > 15
- `age_at_menarche` > `age` (physically impossible)
- `age_at_menarche` < 8 (precocious menarche cutoff)
- `bmi` < 10 or > 60; `weight` < 30 or > 200 kg; `height` outside 1.3–2.2 m (converted from feet)
- `bmi` vs `weight/(height_m²)` inconsistent by > 3 units
- Duplicate rows
- Report missingness per column (none expected, but handled)

Decision policy: flag rows are **counted and reported**, then dropped only if they violate hard physical constraints (menarche > age). Everything is configurable in `src/validation.py`.

## 6. Project structure

```
/workspace
├── plan.md                        ← this file
├── calibrated_menstrual_cycle_ml_dataset.csv
├── .venv/
├── data/                          ← intermediate artifacts (cleaned, engineered, splits, results)
│   ├── cleaned.csv
│   ├── engineered.parquet
│   ├── splits/                    (train/val/test csvs)
│   ├── results/                   (CV leaderboards, optuna studies, test results, shap)
│   └── figures/                   (all plots saved)
├── src/                           ← modular reusable code (imported by notebooks)
│   ├── __init__.py
│   ├── config.py                  (paths, validation bounds, column lists, seeds)
│   ├── validation.py              (load + validate + flag)
│   ├── features.py                (feature engineering)
│   ├── preprocessing.py           (ColumnTransformer builders, fold-safe)
│   ├── split.py                   (train/val/test split)
│   ├── models.py                  (model factories for reg/clf)
│   ├── evaluate.py                (metrics, cross-validate helpers, leaderboard)
│   ├── tune.py                    (optuna objectives for each model+target)
│   └── report.py                  (pretty tables, saving utilities)
├── notebooks/
│   ├── 01_data_loading_and_validation.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_preprocessing_and_split.ipynb
│   ├── 05_baselines.ipynb
│   ├── 06_model_comparison_regression.ipynb
│   ├── 07_model_comparison_classification.ipynb
│   ├── 08_optuna_tuning_regression.ipynb
│   ├── 09_optuna_tuning_classification.ipynb
│   ├── 10_leaderboard.ipynb
│   ├── 11_shap_analysis.ipynb
│   ├── 12_final_test_evaluation.ipynb
│   └── 13_summary_and_report.ipynb
└── README.md
```

## 7. Notebook-by-notebook plan (maps to the 13 requested steps)

| # | Notebook | Steps covered | Output |
|---|---|---|---|
| 01 | `01_data_loading_and_validation.ipynb` | (1) Load + validate | `data/cleaned.csv`, validation report |
| 02 | `02_exploratory_analysis.ipynb` | (2) EDA: target distributions, prevN_* correlation matrix, class balance of `next_is_irregular` | `data/figures/eda_*.png`, saved stats |
| 03 | `03_feature_engineering.ipynb` | (3) mean/std/slope(range)/range of prev1-3 cycle & period length; irregular_count_in_last_3; symptom aggregates; interactions `bmi*stress`, `sleep*exercise`, `bmi*exercise` | `data/engineered.parquet` |
| 04 | `04_preprocessing_and_split.ipynb` | (4)+(5) ColumnTransformer (median impute + StandardScaler for numeric; OrdinalEncoder for ordinal stress/exercise; OneHot for `medication_contraceptive`); 70/15/15 stratified split; explain i.i.d. assumption | `data/splits/*.csv`, pipeline builders in `src/` |
| 05 | `05_baselines.ipynb` | (6) Regression: mean + persistence (prev1 as next). Classification: majority-class. Scored on same CV protocol | baseline leaderboard |
| 06 | `06_model_comparison_regression.ipynb` | (7) Ridge, RandomForest, XGBoost, LightGBM, SVR, KNN — 5-fold CV on train, default hyperparams | `data/results/reg_baseline_cv.csv` |
| 07 | `07_model_comparison_classification.ipynb` | (7) LogisticRegression (class_weight), RandomForest, XGBoost (scale_pos_weight), LightGBM, SVC, KNN — 5-fold CV | `data/results/clf_baseline_cv.csv` |
| 08 | `08_optuna_tuning_regression.ipynb` | (8) Optuna per model+target, 5-fold CV objective, early stopping / pruning for XGB & LGBM | `data/results/optuna/reg_*.db`, best params JSON |
| 09 | `09_optuna_tuning_classification.ipynb` | (8) Same for classification; objective = PR-AUC, class weighting tuned | `data/results/optuna/clf_*.db`, best params JSON |
| 10 | `10_leaderboard.ipynb` | (10) Assemble mean±std leaderboards (baselines + default + tuned), pick winner per target | `data/results/leaderboard_*.csv` |
| 11 | `11_shap_analysis.ipynb` | (11) SHAP on winning model per target (fitted on train+val); sanity-check physiological sense | `data/figures/shap_*.png`, notes |
| 12 | `12_final_test_evaluation.ipynb` | (12) Retrain winner on train+val, evaluate once on test, report MAE/RMSE/R² / F1/PR-AUC/CM | `data/results/final_test_*.csv`, metrics JSON |
| 13 | `13_summary_and_report.ipynb` | (13) Consolidate `src/` modules into one clean runnable pipeline; written summary of model choice + results | `src/pipeline.py`, `docs/results_summary.md` |

## 8. Models per target

- **Regression** (`next_cycle_length`, `next_period_length`): Ridge · RandomForest · XGBoost · LightGBM · SVR · KNN → tuned via Optuna
- **Classification** (`next_is_irregular`): LogisticRegression (class_weight=balanced) · RandomForest · XGBoost · LightGBM · SVC · KNN → tuned via Optuna (imbalance handled via class_weight / scale_pos_weight / is_unbalance)

## 9. Metrics (never plain accuracy for the imbalanced classifier)

- Regression: MAE (primary), RMSE, R² — reported as mean ± std across 5 folds
- Classification: PR-AUC (primary, imbalanced), F1, ROC-AUC, confusion matrix, class balance report

## 10. Tuning protocol (per user's method, notebook 08/09)

1. One "trial" = one hyperparameter combo → fit on 4 folds → eval on 1 → score → discard model
2. ~50–150 trials per model, 5-fold CV objective, random seed fixed per study
3. XGB/LGBM get `early_stopping` on a validation slice inside the fold + Optuna pruning callbacks
4. Winner = best mean CV score; then retrain on train+val; then one test evaluation

## 11. Execution order & dependency chain

```
01 → 02 → 03 → 04 → (05, 06, 07) → (08, 09) → 10 → 11 → 12 → 13
```
Each notebook loads its inputs from `data/` (already produced by earlier notebooks) — re-runnable in isolation.

## 12. Acceptance criteria

- All 13 notebooks run end-to-end with the `mc-cycle` kernel
- No preprocessing step is ever fit on the full dataset or on test data
- Test set is touched exactly once, only in notebook 12
- SHAP sanity checks pass (e.g., prev1_cycle_length should dominate `next_cycle_length`; irregular flags should dominate `next_is_irregular`)
- Modular, reusable code in `src/` + one consolidated `src/pipeline.py`