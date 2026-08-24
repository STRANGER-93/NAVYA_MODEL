from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW_CSV = ROOT / "calibrated_menstrual_cycle_ml_dataset.csv"

DATA_DIR = ROOT / "data" / "processed"
SPLITS_DIR = DATA_DIR / "splits"
RESULTS_DIR = ROOT / "reports" / "metrics"
OPTUNA_DIR = RESULTS_DIR / "optuna"
FIGURES_DIR = ROOT / "reports" / "figures"
MODELS_DIR = ROOT / "models"

CLEANED_CSV = DATA_DIR / "cleaned.csv"
ENGINEERED_CSV = DATA_DIR / "engineered.csv"

for _d in (DATA_DIR, SPLITS_DIR, RESULTS_DIR, OPTUNA_DIR, FIGURES_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42
CV_FOLDS = 5
TEST_SIZE = 0.15
VAL_SIZE = 0.15
N_TRIALS_DEFAULT = 50
N_TRIALS_SVM = 30

TARGETS_REG = ["next_cycle_length", "next_period_length"]
TARGET_CLF = "next_is_irregular"
ALL_TARGETS = TARGETS_REG + [TARGET_CLF]

ORDINAL_COLS = ["stress", "exercise"]
ONEHOT_COLS = ["medication_contraceptive"]
BOOL_COLS = [
    "prev1_irregular_flag", "prev2_irregular_flag", "prev3_irregular_flag",
    "prev1_symptom_cramps", "prev1_symptom_fatigue", "prev1_symptom_mood_swings",
    "prev2_symptom_cramps", "prev2_symptom_fatigue", "prev2_symptom_mood_swings",
    "prev3_symptom_cramps", "prev3_symptom_fatigue", "prev3_symptom_mood_swings",
]
NUMERIC_RAW_COLS = [
    "age", "height", "weight", "bmi", "age_at_menarche", "sleep",
    "prev1_cycle_length", "prev2_cycle_length", "prev3_cycle_length",
    "prev1_period_length", "prev2_period_length", "prev3_period_length",
]
RAW_FEATURES = NUMERIC_RAW_COLS + ORDINAL_COLS + ONEHOT_COLS + BOOL_COLS

ENGINEERED_NUMERIC = [
    "cycle_mean_3", "cycle_std_3", "cycle_range_3", "cycle_slope_3",
    "period_mean_3", "period_std_3", "period_range_3", "period_slope_3",
    "irregular_count_3",
    "cramps_count_3", "fatigue_count_3", "mood_swings_count_3", "symptom_any_count_3",
    "bmi_x_stress", "sleep_x_exercise", "bmi_x_exercise",
]

FEATURES = NUMERIC_RAW_COLS + ENGINEERED_NUMERIC + ORDINAL_COLS + ONEHOT_COLS + BOOL_COLS

ORDINAL_ORDER = {
    "stress": ["Very Low", "Low", "Moderate", "High", "Very High"],
    "exercise": ["Never", "1-2 days/week", "3-4 days/week", "5+ days/week"],
}

FEET_TO_METERS = 0.3048
CYCLE_BOUNDS = (10, 90)
PERIOD_BOUNDS = (1, 15)
MENARCHE_MIN_AGE = 8
BMI_BOUNDS = (10, 60)
WEIGHT_BOUNDS = (30, 200)
HEIGHT_M_BOUNDS = (1.3, 2.2)
BMI_TOLERANCE = 3.0
