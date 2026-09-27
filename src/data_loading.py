"""EDF/annotation loading and patient-file indexing.

Ported directly from the original notebook (cells 10-19) — this logic was
already correct and untouched by the modeling fixes, so it's moved as-is.
"""
import itertools
from typing import List, Tuple, Union

import mne
import pandas as pd
import pyedflib


def load_signals(
    file_path: str, only_info: bool = False, retrieve_signals: List[str] = None
) -> Union[List, Tuple[List[str], List[float]]]:
    """Load the EDF signals for the given filepath.

    Parameters
    ----------
    file_path: str
        The path to the EDF file containing the PSG signals.
    only_info: bool, optional
        If True, return only (signal names, frequencies) instead of the data.
    retrieve_signals: List[str], optional
        The list of signals to load. If None, all signals are loaded.

    Returns
    -------
    Union[List, Tuple[List[str], List[float]]]
        The info (signal names and frequencies) or a list of loaded signals.
    """
    edf = pyedflib.EdfReader(file_path)
    start = edf.getStartdatetime()
    signals, frequencies = edf.getSignalLabels(), edf.getSampleFrequencies()
    if only_info:
        edf.close()
        return signals, frequencies
    data = []

    for ch_idx, sig_name, freq in zip(range(len(signals)), signals, frequencies):
        if retrieve_signals is not None and sig_name not in retrieve_signals:
            continue
        sig = edf.readSignal(chn=ch_idx)
        idx = pd.date_range(
            start=start, periods=len(sig), freq=pd.Timedelta(1 / freq, unit="s")
        )
        data += [pd.Series(sig, index=idx, name=sig_name)]
    edf.close()
    return data


def load_annotations(annotation_file_path: str, psg_file_path: str) -> pd.DataFrame:
    """Load the EDF annotations (hypnogram) for the given filepath."""
    annotations = mne.read_annotations(annotation_file_path)
    # Some hypnogram files error out when getting the start time directly —
    # fall back to reading it from the paired PSG file.
    start_time = pyedflib.EdfReader(psg_file_path).getStartdatetime()
    df = pd.DataFrame()

    df["onset"] = annotations.onset
    df["onset"] = start_time + pd.to_timedelta(df["onset"], unit="s")
    df = df.rename(columns={"onset": "start"})
    assert df["start"].is_unique
    df.set_index("start", inplace=True)

    df["duration"] = annotations.duration
    df["end"] = df.index + pd.to_timedelta(df["duration"], unit="s")
    df.drop(columns="duration", inplace=True)

    df["description"] = annotations.description
    return df


def annotation_to_30s_labels(annotations: pd.DataFrame) -> pd.DataFrame:
    """Convert variable-length annotation spans to fixed 30s-epoch labels."""
    if not (annotations.index[1:] == annotations.end[:-1]).all():
        mask = (annotations.index[1:] == annotations.end[:-1]).to_numpy()
        diffs = (annotations.index[1:] - annotations.end[:-1]).dt.seconds.to_numpy()
        gaps = diffs[diffs != 0]
        gap_starts = annotations[:-1][~mask]
        gap_ends = annotations[1:][~mask]
        for idx, gap in enumerate(gaps):
            assert gap > 0
            gap_start_label = gap_starts["description"].values[idx]
            gap_end_label = gap_ends["description"].values[idx]
            if gap_start_label == gap_end_label:
                annotations.loc[
                    annotations.index == gap_starts.index[idx], "end"
                ] += pd.Timedelta(gap, unit="s")
            else:
                print("Cannot fix gap")

    index = pd.date_range(
        start=annotations.index[0], end=annotations.end.iloc[-1], freq=pd.Timedelta("30s")
    )
    duration = (annotations.end - annotations.index).dt.seconds.values // 30
    labels = itertools.chain.from_iterable(
        [[l] * d for (l, d) in zip(annotations["description"], duration)]
    )
    df = pd.DataFrame({"label": labels}, index=index[:-1])
    df.index.name = "start"
    return df


def build_file_index(data_dir: str, subfolder: str) -> pd.DataFrame:
    """Build the (psg_file, label_file, patient_id) index for one subfolder.

    Mirrors cell 17: PSG and Hypnogram files alternate in the sorted listing,
    and the first 5 characters of the PSG filename identify the patient
    (each ST patient has 2 consecutive-night recordings).
    """
    import os

    folder = os.path.join(data_dir, subfolder)
    sorted_files = sorted(os.listdir(folder))
    psg_hypnogram_files = [(p, h) for p, h in zip(sorted_files[::2], sorted_files[1::2])]
    df_files = pd.DataFrame(psg_hypnogram_files, columns=["psg_file", "label_file"])
    df_files["subfolder"] = subfolder
    df_files["patient_id"] = df_files.psg_file.apply(lambda f: f[:5])
    return df_files
