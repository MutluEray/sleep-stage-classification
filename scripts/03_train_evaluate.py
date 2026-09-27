"""Full 22-fold nested evaluation using the tuned config from script 02.

Writes metrics.json and confusion_matrix.png to results/ — these are what
feed the website showcase later.

    python scripts/03_train_evaluate.py
"""
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import pandas as pd

import config
from src.evaluation import run_nested_cv
from src.visualization import plot_confusion_matrix


def main():
    df_feats = pd.read_parquet(config.FEATURE_PARQUET)
    df_feats = df_feats[df_feats["label"].isin(config.VALID_LABELS)].reset_index(drop=True)
    feat_cols = [c for c in df_feats.columns if c not in config.SKIP_COLS]

    if config.BEST_PARAMS_PATH.exists():
        with open(config.BEST_PARAMS_PATH) as f:
            best_params = json.load(f)
        print(f"Using tuned params from {config.BEST_PARAMS_PATH}: {best_params}")
    else:
        best_params = {"depth": 6, "l2_leaf_reg": 3}
        print(f"No tuned params found — run 02_hyperparameter_search.py first. "
              f"Using default: {best_params}")

    results = run_nested_cv(df_feats, feat_cols, best_params, n_splits=22)

    with open(config.METRICS_PATH, "w") as f:
        json.dump(results["metrics"], f, indent=2)
    print(f"\nSaved metrics to {config.METRICS_PATH}")

    fig, ax = plt.subplots(figsize=(8, 7))
    plot_confusion_matrix(results["trues"], results["preds"], classes=config.VALID_LABELS, normalize=True, ax=ax)
    fig.tight_layout()
    fig.savefig(config.CONFUSION_MATRIX_PATH, dpi=150)
    print(f"Saved confusion matrix to {config.CONFUSION_MATRIX_PATH}")

    # A raw (non-normalized) version is also useful for the website writeup
    fig2, ax2 = plt.subplots(figsize=(8, 7))
    plot_confusion_matrix(results["trues"], results["preds"], classes=config.VALID_LABELS, normalize=False, ax=ax2)
    fig2.tight_layout()
    fig2.savefig(config.FIGURES_DIR / "confusion_matrix_raw.png", dpi=150)
    print(f"Saved raw confusion matrix to {config.FIGURES_DIR / 'confusion_matrix_raw.png'}")


if __name__ == "__main__":
    main()
