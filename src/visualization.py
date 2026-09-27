"""Plotting helpers.

NOTE: your original notebook imported `plot_confusion_matrix`,
`plot_learning_curve`, and `plot_linear_classification_coefs` from a
`visualizations.py` module that wasn't part of the uploaded files, so these
are clean reimplementations covering the same two plots actually used in
the ML section (confusion matrix + learning curve). If your original module
had a specific style/layout you want to match exactly, share it and I'll
port it in place of this.
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve

STAGE_ORDER = {
    "Sleep stage W": 4,
    "Sleep stage R": 3,
    "Sleep stage 1": 2,
    "Sleep stage 2": 1,
    "Sleep stage 3": 0,
}
STAGE_SHORT = {
    "Sleep stage W": "W",
    "Sleep stage R": "REM",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
}


def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, ax=None, cmap="Blues"):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True label",
        xlabel="Predicted label",
        title="Normalized confusion matrix" if normalize else "Confusion matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    return ax


def plot_learning_curve(estimator, X, y, cv, scoring="f1_macro", n_jobs=1,
                         train_sizes=None, title="Learning curve", ax=None):
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)

    train_sizes_abs, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs, train_sizes=train_sizes,
    )

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6))

    train_mean, train_std = train_scores.mean(axis=1), train_scores.std(axis=1)
    test_mean, test_std = test_scores.mean(axis=1), test_scores.std(axis=1)

    ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.15, color="tab:blue")
    ax.fill_between(train_sizes_abs, test_mean - test_std, test_mean + test_std, alpha=0.15, color="tab:orange")
    ax.plot(train_sizes_abs, train_mean, "o-", color="tab:blue", label="Training score")
    ax.plot(train_sizes_abs, test_mean, "o-", color="tab:orange", label="Cross-validation score")
    ax.set(title=title, xlabel="Training examples", ylabel=scoring)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    return ax


def plot_hypnogram(true_labels, pred_labels=None, epoch_seconds=30, ax=None,
                    true_color="#0f172a", pred_color="#0d9488", show_disagreement=False,
                    mismatch_color="#fecaca"):
    """Classic clinical hypnogram: sleep stage over time, W/REM at top,
    N3 (deepest) at bottom. If pred_labels is given, overlays the predicted
    stage as a dashed line. The solid-vs-dashed offset already makes
    disagreement visually obvious; set show_disagreement=True to also shade
    mismatched epochs if you want it more explicit.

    true_labels / pred_labels: array-like of strings like "Sleep stage W",
    in chronological order for a single night.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(13, 3.6))

    n = len(true_labels)
    hours = np.arange(n) * epoch_seconds / 3600
    bin_width = epoch_seconds / 3600
    true_y = np.array([STAGE_ORDER[l] for l in true_labels])

    if pred_labels is not None:
        pred_y = np.array([STAGE_ORDER[l] for l in pred_labels])
        if show_disagreement:
            mismatch = true_y != pred_y
            ax.fill_between(hours, -0.5, 4.5, where=mismatch, step="post",
                              color=mismatch_color, alpha=0.55, linewidth=0, zorder=1,
                              label="Disagreement")
        ax.step(hours, pred_y, where="post", color=pred_color, linewidth=1.3,
                 linestyle="--", alpha=0.9, label="Predicted stage", zorder=3)

    ax.step(hours, true_y, where="post", color=true_color, linewidth=1.6,
             label="True stage", zorder=4)

    ax.set_yticks(sorted(STAGE_ORDER.values()))
    ax.set_yticklabels([STAGE_SHORT[k] for k, v in sorted(STAGE_ORDER.items(), key=lambda kv: kv[1])])
    ax.set_ylim(-0.5, 4.5)
    ax.set_xlabel("Time (hours into recording)")
    ax.set_ylabel("Sleep stage")
    ax.set_xlim(0, hours.max() + bin_width)
    ax.grid(axis="y", alpha=0.2, linewidth=0.6)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95, ncol=3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return ax


def plot_transition_spectrograms(windows, sf=100, epoch_seconds=30, pre_epochs=2, post_epochs=2,
                                  fmax=30, cmap="viridis", ncols=None, figsize_per_panel=(3.6, 2.8)):
    """Small-multiples spectrograms (time-frequency energy) around each
    stage transition — shows the actual band-power characteristics
    (delta/sigma/alpha) changing at the boundary, which raw waveform
    amplitude alone doesn't make visible to the eye.

    windows: list of dicts with keys 'signal', 'from_label', 'to_label',
    and optionally 'story' (short caption shown above the panel).
    """
    from scipy.signal import spectrogram

    n = len(windows)
    if ncols is None:
        ncols = 3 if n > 4 else 2
    ncols = min(n, ncols)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows))
    axes = np.atleast_1d(axes).ravel()

    for i, w in enumerate(windows):
        ax = axes[i]
        sig = w["signal"]
        f, t, Sxx = spectrogram(sig, fs=sf, nperseg=int(sf * 2), noverlap=int(sf * 1.5))
        t_rel = t - pre_epochs * epoch_seconds
        mask = f <= fmax
        Sxx_db = 10 * np.log10(Sxx[mask] + 1e-12)

        ax.pcolormesh(t_rel, f[mask], Sxx_db, shading="gouraud", cmap=cmap)
        ax.axvline(0, color="white", linewidth=1.2, linestyle="--", alpha=0.85)
        ax.set_ylim(0, fmax)
        ax.set_xlabel("Time relative to transition (s)", fontsize=8.5)
        ax.set_ylabel("Freq (Hz)", fontsize=8.5)
        ax.tick_params(labelsize=8)

        ax.text(t_rel.min() * 0.55, fmax * 0.9, STAGE_SHORT.get(w["from_label"], w["from_label"]),
                color="white", fontsize=10, fontweight="bold", ha="center")
        ax.text(t_rel.max() * 0.55, fmax * 0.9, STAGE_SHORT.get(w["to_label"], w["to_label"]),
                color="white", fontsize=10, fontweight="bold", ha="center")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    return fig, axes


def plot_signal_snippet(signal, sf, epoch_labels=None, epoch_seconds=30, ax=None, color="#0f766e"):
    """Raw signal strip (e.g. a few minutes of EEG) with epoch boundaries
    and true-stage labels annotated, to accompany the hypnogram.

    signal: 1D array of raw (or filtered) signal values.
    sf: sampling frequency in Hz.
    epoch_labels: list of stage strings, one per epoch_seconds-length segment
        covered by `signal`.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 2.2))

    t = np.arange(len(signal)) / sf
    ax.plot(t, signal, color=color, linewidth=0.6)
    ax.set_xlim(0, t.max())
    ax.set_xlabel("Time (s)")
    ax.set_yticks([])
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    if epoch_labels is not None:
        n_epochs = len(epoch_labels)
        for i in range(1, n_epochs):
            ax.axvline(i * epoch_seconds, color="#cbd5e1", linewidth=0.8, linestyle="--")
        ymax = ax.get_ylim()[1]
        for i, label in enumerate(epoch_labels):
            ax.text(i * epoch_seconds + epoch_seconds / 2, ymax * 0.92, STAGE_SHORT[label],
                    ha="center", va="top", fontsize=9, color="#334155", fontweight="bold")
    return ax
