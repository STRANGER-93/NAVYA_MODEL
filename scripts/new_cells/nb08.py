SETUP = '''\
import sys
from pathlib import Path
root = Path.cwd()
while not (root / "src").exists() and root != root.parent:
    root = root.parent
sys.path.insert(0, str(root))
%matplotlib inline
import glob
import optuna
from IPython.display import display
from optuna.visualization import plot_optimization_history, plot_param_importances
optuna.logging.set_verbosity(optuna.logging.WARNING)
from src.config import OPTUNA_DIR, FIGURES_DIR
'''

CELLS = [
    '''\
PREFIX = "reg"
for db in sorted(glob.glob(str(OPTUNA_DIR / f"{PREFIX}_*.db"))):
    storage = f"sqlite:///{db}"
    for sname in optuna.get_all_study_names(storage):
        study = optuna.load_study(study_name=sname, storage=storage)
        print(f"{sname}: {len(study.trials)} trials | best {study.best_value:.4f}", flush=True)
        hist = plot_optimization_history(study)
        hist.update_layout(title=f"{sname} - optimization history", height=360)
        try:
            imp = plot_param_importances(study)
            imp.update_layout(title=f"{sname} - hyperparameter importances", height=360)
            imp.write_html(FIGURES_DIR / f"optuna_{sname}_importances.html")
        except Exception as e:
            imp = None
            print("  importance plot skipped:", e)
        hist.write_html(FIGURES_DIR / f"optuna_{sname}_history.html")
        display(hist)
        if imp is not None:
            display(imp)
''',
]

HEADER = ("## Added diagnostics — Optuna convergence & hyperparameter importances\\n"
          "Loaded from the persisted study databases (no re-tuning). Optimization history shows "
          "whether the search converged; importances show which hyperparameters mattered.")
