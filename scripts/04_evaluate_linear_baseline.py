"""Linear SGD baseline through the same 22-fold GroupKFold structure as
CatBoost, for a direct apples-to-apples comparison on the rebuilt feature
set. No early stopping (not applicable to SGD) and no feature selection
(matches the original notebook's linear pipeline, and this model wasn't the
one overfitting) — so this is quick to run, no PowerShap overhead.

    python scripts/04_evaluate_linear_baseline.py
"""
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

import config
from src.evaluation import run_linear_baseline_cv


def main():
    df_feats = pd.read_parquet(config.FEATURE_PARQUET)
    df_feats = df_feats[df_feats["label"].isin(config.VALID_LABELS)].reset_index(drop=True)
    feat_cols = [c for c in df_feats.columns if c not in config.SKIP_COLS]

    print(f"Loaded {len(df_feats)} rows, {len(feat_cols)} raw features, "
          f"{df_feats.patient_id.nunique()} patients\n")

    results = run_linear_baseline_cv(df_feats, feat_cols, n_splits=22)

    out_path = config.RESULTS_DIR / "metrics_linear_baseline.json"
    with open(out_path, "w") as f:
        json.dump(results["metrics"], f, indent=2)
    print(f"\nSaved metrics to {out_path}")

    # Side-by-side comparison against the CatBoost run, if it's already been done
    if config.METRICS_PATH.exists():
        with open(config.METRICS_PATH) as f:
            catboost_metrics = json.load(f)
        print("\n=== Comparison: Linear baseline vs. tuned CatBoost ===")
        print(f"{'':<20} {'Linear':>10} {'CatBoost':>10}")
        for k in ["macro_f1", "balanced_accuracy", "accuracy"]:
            lin_v = results["metrics"]["test"][k]
            cb_v = catboost_metrics["test"][k]
            print(f"{k:<20} {lin_v:>10.4f} {cb_v:>10.4f}")


if __name__ == "__main__":
    main()
