"""tsflex/tsfresh/antropy/yasa feature extraction.

Ported from cells 33-38. The feature set itself (time-domain stats, entropy
measures, band powers at 30s/60s/90s windows) was not the source of the
overfitting problem — the problem was that all ~1000 resulting columns went
into CatBoost unfiltered. Feature *selection* now happens downstream in
feature_selection.py, not here.
"""
import logging

import antropy as ant
import numpy as np
import scipy.stats as ss
from tsflex.features import FeatureCollection, FuncWrapper, MultipleFeatureDescriptors
from tsflex.features.integrations import tsfresh_settings_wrapper
from tsflex.features.logger import logger as tsflex_feat_logger
from tsflex.processing.logger import logger as tsflex_proc_logger
from yasa import bandpower

tsflex_feat_logger.setLevel(level=logging.ERROR)
tsflex_proc_logger.setLevel(level=logging.ERROR)

TSFRESH_SETTINGS = {
    "fft_aggregated": [
        {"aggtype": "centroid"},
        {"aggtype": "variance"},
        {"aggtype": "skew"},
        {"aggtype": "kurtosis"},
    ],
    "fourier_entropy": [{"bins": b} for b in [2, 3, 5, 10, 30, 60, 100]],
    "binned_entropy": [{"max_bins": b} for b in [5, 10, 30, 60]],
}

BANDS = [
    (0.4, 1, "sdelta"),
    (1, 4, "fdelta"),
    (4, 8, "theta"),
    (8, 12, "alpha"),
    (12, 16, "sigma"),
    (16, 30, "beta"),
]
BANDPOWER_OUTPUTS = [b[2] for b in BANDS] + ["TotalAbsPow"]


def _wrapped_higuchi_fd(x):
    return ant.higuchi_fd(np.array(x, dtype="float64"))


def _wrapped_bandpowers(x, sf, bands):
    return bandpower(x, sf=sf, bands=bands).values[0][:-2]


def build_feature_collection() -> FeatureCollection:
    """Returns the tsflex FeatureCollection used to build df_feats."""
    time_funcs = [
        np.std,
        ss.iqr,
        ss.skew,
        ss.kurtosis,
        ant.num_zerocross,
        FuncWrapper(ant.hjorth_params, output_names=["hjorth_mobility", "hjorth_complexity"]),
        _wrapped_higuchi_fd,
        ant.petrosian_fd,
        ant.perm_entropy,
    ] + tsfresh_settings_wrapper(TSFRESH_SETTINGS)

    freq_funcs = [
        FuncWrapper(_wrapped_bandpowers, sf=100, bands=BANDS, output_names=BANDPOWER_OUTPUTS)
    ]

    time_feats = MultipleFeatureDescriptors(
        time_funcs,
        ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal", "EMG submental"],
        windows=["30s", "60s", "90s"],
        strides="30s",
    )
    freq_feats = MultipleFeatureDescriptors(
        freq_funcs,
        ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal"],
        windows=["30s", "60s", "90s"],
        strides="30s",
    )
    return FeatureCollection([time_feats, freq_feats])
