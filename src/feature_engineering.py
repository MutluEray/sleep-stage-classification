"""Post-processing of the raw feature table: band ratios, temporal-context
(shift) features, and label cleanup.

Ported from cells 45, 47, 49 — these produced the 1051-column table that was
then fed unfiltered into CatBoost. That's fixed downstream in
feature_selection.py, not here; this module's job is unchanged.
"""
import pandas as pd


def add_band_ratio_features(df_feats: pd.DataFrame) -> pd.DataFrame:
    """Adds classic sleep-EEG band-power ratios (alpha/theta, delta/beta,
    etc.) for each EEG channel and window size. Ported from cell 45."""
    eeg_signals = [c.split("__")[0] for c in df_feats.columns if c.startswith("EEG")]
    eeg_signals = sorted(set(eeg_signals))
    bands = ["alpha", "beta", "sdelta", "fdelta", "sigma", "theta"]

    for eeg_sig in eeg_signals:
        eeg_bands = [c for c in df_feats.columns if c.startswith(eeg_sig) and c.split("__")[1] in bands]
        windows = sorted(set(b.split("__")[-1] for b in eeg_bands))
        for window in windows:
            delta = df_feats["__".join([eeg_sig, "sdelta", window])] + df_feats["__".join([eeg_sig, "fdelta", window])]
            fdelta_theta = df_feats["__".join([eeg_sig, "fdelta", window])] + df_feats["__".join([eeg_sig, "theta", window])]
            alpha = df_feats["__".join([eeg_sig, "alpha", window])]
            beta = df_feats["__".join([eeg_sig, "beta", window])]
            theta = df_feats["__".join([eeg_sig, "theta", window])]
            sigma = df_feats["__".join([eeg_sig, "sigma", window])]

            df_feats["__".join([eeg_sig, "fdelta+theta", window])] = fdelta_theta.astype("float32")
            df_feats["__".join([eeg_sig, "alpha/theta", window])] = (alpha / theta).astype("float32")
            df_feats["__".join([eeg_sig, "delta/beta", window])] = (delta / beta).astype("float32")
            df_feats["__".join([eeg_sig, "delta/sigma", window])] = (delta / sigma).astype("float32")
            df_feats["__".join([eeg_sig, "delta/theta", window])] = (delta / theta).astype("float32")
    return df_feats


def add_shift_context_features(df_feats: pd.DataFrame) -> pd.DataFrame:
    """Adds features from the adjacent epoch(s) before/after each 30s epoch,
    per patient recording. Ported from cell 47.

    NOTE: this is applied to the full dataset before the group-wise CV split,
    same as the original — that's safe because shifting is computed
    per-psg_file (i.e., only ever looks at a single patient's own adjacent
    epochs), so it introduces no leakage across patients/folds.
    """
    feats_30s = [f for f in df_feats.columns if "w=30s" in f]
    feats_60s = [f for f in df_feats.columns if "w=1m_" in f]
    feats_90s = [f for f in df_feats.columns if "w=1m30s" in f]

    dfs = []
    for psg_file in df_feats.psg_file.unique():
        sub_df = df_feats[df_feats.psg_file == psg_file]

        sub_df = sub_df.merge(sub_df[feats_90s].shift(1).add_suffix("_shift=30s"), left_index=True, right_index=True)
        sub_df = sub_df.drop(columns=feats_90s)

        sub_df = sub_df.merge(sub_df[feats_60s].shift(1).add_suffix("_shift=30s"), left_index=True, right_index=True)

        sub_df = sub_df.merge(sub_df[feats_30s].shift(2).add_suffix("_shift=1m"), left_index=True, right_index=True)
        sub_df = sub_df.merge(sub_df[feats_30s].shift(1).add_suffix("_shift=30s"), left_index=True, right_index=True)
        sub_df = sub_df.merge(sub_df[feats_30s].shift(-1).add_suffix("_shift=-30s"), left_index=True, right_index=True)
        sub_df = sub_df.merge(sub_df[feats_30s].shift(-2).add_suffix("_shift=-1m"), left_index=True, right_index=True)
        dfs.append(sub_df)

    return pd.concat(dfs)


def clean_labels(df_feats: pd.DataFrame) -> pd.DataFrame:
    """Merges Stage 4 into Stage 3 (standard practice for this dataset) and
    drops Movement/unscored epochs. Ported from cell 49."""
    df_feats = df_feats.copy()
    df_feats.loc[df_feats["label"] == "Sleep stage 4", "label"] = "Sleep stage 3"
    df_feats = df_feats[df_feats["label"] != "Movement time"]
    df_feats = df_feats[df_feats["label"] != "Sleep stage ?"]
    df_feats = df_feats[~df_feats["label"].isna()]
    return df_feats
