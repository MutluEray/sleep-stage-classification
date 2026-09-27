"""Central configuration: paths and constants shared across the pipeline.

Edit DATA_DIR to point at your already-downloaded Sleep-EDF Expanded folder.
Everything else (features/, results/) is created relative to the repo root.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# Point this at your local copy of the dataset:
# https://physionet.org/content/sleep-edfx/1.0.0/
DATA_DIR = REPO_ROOT / "data" / "sleep-edf-database-expanded-1.0.0"

FEATURES_DIR = REPO_ROOT / "features"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"

FEATURE_PARQUET = FEATURES_DIR / "sleep-edf__telemetry_features_ALL__90s.parquet"
BEST_PARAMS_PATH = RESULTS_DIR / "best_hyperparams.json"
METRICS_PATH = RESULTS_DIR / "metrics.json"
CONFUSION_MATRIX_PATH = FIGURES_DIR / "confusion_matrix.png"
LEARNING_CURVE_PATH = FIGURES_DIR / "learning_curve.png"

# "sleep-telemetry" = the 22-patient ST subset used throughout the notebook.
# Switch to "sleep-cassette" for the larger ~78-patient SC cohort.
SUBFOLDER = "sleep-telemetry"

COMMON_SIGNALS = ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal", "EMG submental"]

SKIP_COLS = ["psg_file", "label", "patient_id"]
VALID_LABELS = [
    "Sleep stage W",
    "Sleep stage 1",
    "Sleep stage 2",
    "Sleep stage 3",
    "Sleep stage R",
]
RANDOM_STATE = 0

FEATURES_DIR.mkdir(exist_ok=True, parents=True)
RESULTS_DIR.mkdir(exist_ok=True, parents=True)
FIGURES_DIR.mkdir(exist_ok=True, parents=True)
