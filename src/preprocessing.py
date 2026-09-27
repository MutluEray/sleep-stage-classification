"""Signal preprocessing: bandpass filtering via a tsflex SeriesPipeline.

Ported from cell 29 — bandpass ranges (0.4-30 Hz for EEG/EOG, 0.5-10 Hz for
EMG) were already sensible choices for sleep-staging literature and were not
part of the overfitting problem, so left unchanged.
"""
from scipy.signal import butter, lfilter
from tsflex.processing import SeriesPipeline, SeriesProcessor


def butter_bandpass_filter(sig, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return lfilter(b, a, sig)


def build_processing_pipeline() -> SeriesPipeline:
    """Returns the bandpass-filter pipeline applied to raw PSG signals."""
    eeg_bandpass = SeriesProcessor(
        function=butter_bandpass_filter,
        series_names=["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal"],
        lowcut=0.4,
        highcut=30,
        fs=100,
    )
    emg_bandpass = SeriesProcessor(
        function=butter_bandpass_filter,
        series_names=["EMG submental"],
        lowcut=0.5,
        highcut=10,
        fs=100,
    )
    return SeriesPipeline([eeg_bandpass, emg_bandpass])
