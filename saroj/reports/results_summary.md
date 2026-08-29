# Results Summary

- Rows kept after validation: 20000 (dropped 0)
- i.i.d. assumption: no user id/timestamp column exists; if rows are repeated measures, mild leakage could inflate CV slightly.

## Regression leaderboard (CV MAE, lower is better)

| model                | target             |   mae_mean |   rmse_mean |   r2_mean |
|:---------------------|:-------------------|-----------:|------------:|----------:|
| random_forest_tuned  | next_cycle_length  |    1.1916  |     1.73416 |   0.79573 |
| xgboost_tuned        | next_cycle_length  |    1.20509 |     1.75389 |   0.79109 |
| lightgbm_tuned       | next_cycle_length  |    1.21017 |     1.76233 |   0.78908 |
| random_forest        | next_cycle_length  |    1.21759 |     1.77032 |   0.78722 |
| ridge_tuned          | next_cycle_length  |    1.22316 |     1.78756 |   0.78301 |
| ridge                | next_cycle_length  |    1.22345 |     1.78799 |   0.7829  |
| svr_tuned            | next_cycle_length  |    1.23922 |     1.78624 |   0.78329 |
| svr                  | next_cycle_length  |    1.24341 |     1.79129 |   0.78208 |
| lightgbm             | next_cycle_length  |    1.26114 |     1.83171 |   0.77207 |
| knn_tuned            | next_cycle_length  |    1.2776  |     1.82624 |   0.77347 |
| baseline_persistence | next_cycle_length  |    1.3023  |     1.987   |   0.7319  |
| xgboost              | next_cycle_length  |    1.33991 |     1.90969 |   0.75246 |
| knn                  | next_cycle_length  |    1.3645  |     1.933   |   0.7463  |
| baseline_mean        | next_cycle_length  |    2.7447  |     3.8407  |  -0.0008  |
| random_forest_tuned  | next_period_length |    0.57132 |     0.72032 |   0.68268 |
| ridge_tuned          | next_period_length |    0.57432 |     0.72156 |   0.68156 |
| ridge                | next_period_length |    0.5746  |     0.7221  |   0.68107 |
| lightgbm_tuned       | next_period_length |    0.57675 |     0.72572 |   0.67789 |
| xgboost_tuned        | next_period_length |    0.57774 |     0.72672 |   0.677   |
| random_forest        | next_period_length |    0.57951 |     0.72699 |   0.67675 |
| knn_tuned            | next_period_length |    0.58447 |     0.73368 |   0.67082 |
| svr_tuned            | next_period_length |    0.58566 |     0.73868 |   0.66631 |
| svr                  | next_period_length |    0.59112 |     0.74571 |   0.65994 |
| lightgbm             | next_period_length |    0.60056 |     0.75362 |   0.65265 |
| knn                  | next_period_length |    0.61701 |     0.77956 |   0.62833 |
| baseline_persistence | next_period_length |    0.6332  |     0.8913  |   0.5142  |
| xgboost              | next_period_length |    0.64317 |     0.80593 |   0.60269 |
| baseline_mean        | next_period_length |    1.0302  |     1.2791  |  -0.0004  |

## Classification leaderboard (CV PR-AUC, higher is better)

| model                     | target            |   pr_auc_mean |   pr_auc_std |   f1_mean |   roc_auc_mean |
|:--------------------------|:------------------|--------------:|-------------:|----------:|---------------:|
| random_forest_tuned       | next_is_irregular |       0.87767 |      0.02137 |   0.75029 |        0.9706  |
| random_forest             | next_is_irregular |       0.87266 |      0.02337 |   0.79219 |        0.96897 |
| knn_tuned                 | next_is_irregular |       0.86822 |      0.02924 |   0.78058 |        0.96122 |
| lightgbm_tuned            | next_is_irregular |       0.86652 |      0.02383 |   0.78725 |        0.9633  |
| xgboost_tuned             | next_is_irregular |       0.86018 |      0.02993 |   0.71938 |        0.96489 |
| lightgbm                  | next_is_irregular |       0.85117 |      0.02303 |   0.78153 |        0.95723 |
| xgboost                   | next_is_irregular |       0.84228 |      0.0239  |   0.77739 |        0.95292 |
| svc_tuned                 | next_is_irregular |       0.83361 |      0.01966 |   0.78167 |        0.95928 |
| svc                       | next_is_irregular |       0.81211 |      0.04107 |   0.77999 |        0.9579  |
| knn                       | next_is_irregular |       0.78186 |      0.04307 |   0.78033 |        0.91431 |
| logistic_regression_tuned | next_is_irregular |       0.77692 |      0.01809 |   0.71788 |        0.94322 |
| logistic_regression       | next_is_irregular |       0.77658 |      0.018   |   0.7183  |        0.94317 |
| persistence_prev1_flag    | next_is_irregular |       0.616   |    nan       |   0.7708  |        0.8708  |
| majority_class            | next_is_irregular |       0.0909  |    nan       |   0       |        0.5     |

## Final held-out test results

### next_cycle_length

```json
{
  "mae": 1.1982,
  "rmse": 1.7608,
  "r2": 0.7984
}
```

### next_period_length

```json
{
  "mae": 0.5856,
  "rmse": 0.7355,
  "r2": 0.6715
}
```

### next_is_irregular

```json
{
  "pr_auc": 0.8736,
  "f1": 0.7508,
  "roc_auc": 0.974,
  "confusion_matrix": [
    [
      2599,
      128
    ],
    [
      32,
      241
    ]
  ]
}
```

## Notes

- Test set evaluated exactly once (notebook 12).
- SVM tuning used a fixed 5000-row subsample for tractability.
- Class imbalance handled via class weights / scale_pos_weight / is_unbalance; PR-AUC is the primary classification metric.

## Added diagnostics (follow-up)

- **Near-duplicate check**: 37.1% of rows (7,417/20,000) share an identical static profile
  (3,235 groups, max group size 7). This strengthens the i.i.d. caveat: rows may be repeated
  measures of the same individuals, which could mildly inflate CV scores.
- **Per-fold CV boxplots**: fold-level MAE / PR-AUC distributions saved to
  reports/metrics/reg_fold_scores.csv and clf_fold_scores.csv (plots in reports/figures/cv_boxplots_*.png).
- **Optuna convergence + hyperparameter importances**: interactive plots for every study in
  reports/figures/optuna_<study>_{history,importances}.html.
- **Training curves**: RandomForest validation curves vs n_estimators and true boosting loss
  curves (train vs val, early-stopping point marked) - reports/figures/final_rf_val_curves.png,
  final_boosting_loss_curves.png.
- **Regression diagnostics**: OOF predicted-vs-actual, residuals-vs-predicted, residual histogram
  (final_reg_diagnostics.png).
- **Classification diagnostics**: raw + normalized confusion matrices, PR curve (OOF AP=0.874),
  ROC, calibration, and threshold sweep on out-of-fold predictions. Default 0.5 gives
  F1=0.750 (R=0.854/P=0.668); best-F1 threshold is 0.74 -> F1=0.795 (R=0.757/P=0.837).
  If false negatives are costlier, choose a threshold below 0.5 deliberately.
- **SHAP bar + dependence plots** for top features per target; SHAP values persisted to
  reports/metrics/shap_vals/*.npz.
