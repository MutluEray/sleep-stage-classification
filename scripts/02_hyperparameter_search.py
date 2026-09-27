"""Quick grouped hyperparameter search over (depth, l2_leaf_reg).

Reads the feature parquet built by 01_build_feature_dataset.py, writes the
best config to results/best_hyperparams.json for script 03 to consume.

    python scripts/02_hyperparameter_search.py
"""
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

import config
from src.evaluation import quick_hparam_search


def main():
    df_feats = pd.read_parquet(config.FEATURE_PARQUET)
    df_feats = df_feats[df_feats["label"].isin(config.VALID_LABELS)].reset_index(drop=True)
    feat_cols = [c for c in df_feats.columns if c not in config.SKIP_COLS]

    print(f"Loaded {len(df_feats)} rows, {len(feat_cols)} raw features, "
          f"{df_feats.patient_id.nunique()} patients\n")

    best_params = quick_hparam_search(df_feats, feat_cols, n_splits=5)

    with open(config.BEST_PARAMS_PATH, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"\nSaved best params to {config.BEST_PARAMS_PATH}")


if __name__ == "__main__":
    main()
