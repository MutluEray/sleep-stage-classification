"""Raw EDF files -> feature parquet.

This is the expensive step (loads + filters + extracts features for every
patient recording). Run it once; downstream scripts just read the parquet
it produces.

Usage:
    python scripts/01_build_feature_dataset.py                # full run, all patients
    python scripts/01_build_feature_dataset.py --n-patients 2  # smoke test on first 2 patients
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from tqdm.auto import tqdm

import config
from src.data_loading import build_file_index, load_annotations, load_signals, annotation_to_30s_labels
from src.feature_engineering import add_band_ratio_features, add_shift_context_features, clean_labels
from src.feature_extraction import build_feature_collection
from src.preprocessing import build_processing_pipeline


def main(n_patients: int = None):
    df_files = build_file_index(str(config.DATA_DIR), config.SUBFOLDER)
    print(f"Found {len(df_files)} recordings across {df_files.patient_id.nunique()} patients")

    if n_patients is not None:
        keep_patients = df_files.patient_id.unique()[:n_patients]
        df_files = df_files[df_files.patient_id.isin(keep_patients)].reset_index(drop=True)
        print(f"SMOKE TEST: restricted to {n_patients} patient(s), {len(df_files)} recording(s): "
              f"{list(keep_patients)}")

    process_pipe = build_processing_pipeline()
    feature_collection = build_feature_collection()

    df_feats_list = []
    for sub_folder, psg_file, hypnogram_file in tqdm(
        zip(df_files.subfolder, df_files.psg_file, df_files.label_file), total=len(df_files)
    ):
        file_folder = config.DATA_DIR / sub_folder
        data = load_signals(
            str(file_folder / psg_file), retrieve_signals=config.COMMON_SIGNALS
        )
        data_processed = process_pipe.process(data)
        df_feat = feature_collection.calculate(
            data_processed, return_df=True, window_idx="begin"
        ).astype("float32")

        annotations = load_annotations(str(file_folder / hypnogram_file), str(file_folder / psg_file))
        annotations = annotation_to_30s_labels(annotations)
        df_feat = df_feat.merge(annotations, left_index=True, right_index=True)

        df_feat["psg_file"] = psg_file
        df_feat["patient_id"] = psg_file[:5]
        df_feats_list.append(df_feat)

    df_feats = pd.concat(df_feats_list)
    df_feats.rename(columns={"description": "label"}, inplace=True)

    print("Adding band-ratio features...")
    df_feats = add_band_ratio_features(df_feats)

    print("Adding shift-context features...")
    df_feats = add_shift_context_features(df_feats)

    print("Cleaning labels...")
    df_feats = clean_labels(df_feats)

    print(f"Final shape: {df_feats.shape}")
    print(f"Patients: {df_feats.patient_id.nunique()}")
    print(f"Label distribution:\n{df_feats['label'].value_counts()}")

    out_path = config.FEATURE_PARQUET
    if n_patients is not None:
        out_path = config.FEATURES_DIR / f"smoke_test_{n_patients}patients.parquet"

    df_feats.to_parquet(out_path)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-patients", type=int, default=None,
                         help="Restrict to the first N patients (smoke test). Omit for a full run.")
    args = parser.parse_args()
    main(n_patients=args.n_patients)
