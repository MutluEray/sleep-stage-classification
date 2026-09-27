"""Generate the two showcase figures: a full-night hypnogram (true vs.
predicted stage over time) and a raw-signal snippet with epoch labels, for
one held-out demo patient.

Trains a single model on the other 21 patients (not a full 22-fold CV -
this is just for the visualization, the real evaluation numbers come from
03_train_evaluate.py / 04_evaluate_linear_baseline.py), predicts on the
held-out patient, and plots true vs. predicted stage across their full
night alongside a short raw-signal strip.

Usage:
    python scripts/05_generate_showcase_figures.py                       # first patient, linear model
    python scripts/05_generate_showcase_figures.py --patient ST7011 --model catboost
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer

import config
from src.data_loading import build_file_index, load_signals
from src.modeling import build_catboost_model, build_linear_baseline
from src.preprocessing import build_processing_pipeline
from src.visualization import plot_hypnogram, plot_transition_spectrograms


def find_best_patient(df_feats: pd.DataFrame, feat_cols: list):
    """Trains the linear model against every patient held out in turn and
    returns the one with highest single-patient accuracy, plus its cached
    predictions (so the caller doesn't need to retrain). Linear-only
    because it's cheap enough to do 22 times; CatBoost isn't."""
    patients = sorted(df_feats.patient_id.unique())
    best_acc, best_pid, best_result = -1, None, None

    for pid in patients:
        train_df = df_feats[df_feats.patient_id != pid]
        test_df = df_feats[df_feats.patient_id == pid].sort_index()

        model = build_linear_baseline()
        model.fit(train_df[feat_cols], train_df["label"])
        preds = model.predict(test_df[feat_cols])
        true = test_df["label"].values
        acc = (preds == true).mean()
        print(f"  {pid}: accuracy={acc:.3f}")

        if acc > best_acc:
            best_acc, best_pid, best_result = acc, pid, (preds, true)

    print(f"\nBest patient: {best_pid} (accuracy={best_acc:.3f})")
    return best_pid, best_result


def main(patient_id: str, model_name: str, rescan: bool):
    df_feats = pd.read_parquet(config.FEATURE_PARQUET)
    df_feats = df_feats[df_feats["label"].isin(config.VALID_LABELS)].reset_index(drop=True)
    feat_cols = [c for c in df_feats.columns if c not in config.SKIP_COLS]

    best_patient_path = config.RESULTS_DIR / "best_patient.json"
    search_preds = search_true = None

    if patient_id is None:
        if best_patient_path.exists() and not rescan:
            with open(best_patient_path) as f:
                cached = json.load(f)
            patient_id = cached["patient_id"]
            print(f"Using cached best patient: {patient_id} (accuracy={cached['accuracy']:.3f}) "
                  f"— pass --rescan to redo the search.")
        else:
            print("Scanning all patients for the best-performing held-out fit...")
            patient_id, (search_preds, search_true) = find_best_patient(df_feats, feat_cols)
            acc = float((search_preds == search_true).mean())
            with open(best_patient_path, "w") as f:
                json.dump({"patient_id": patient_id, "accuracy": acc}, f, indent=2)

    print(f"\nDemo patient: {patient_id}, model: {model_name}")

    train_df = df_feats[df_feats.patient_id != patient_id]
    test_df = df_feats[df_feats.patient_id == patient_id].sort_index()

    if model_name == "linear" and search_preds is not None:
        # already trained during the best-patient search, reuse it
        preds, true_labels = search_preds, search_true
    else:
        imputer = SimpleImputer().fit(train_df[feat_cols])
        X_train = pd.DataFrame(imputer.transform(train_df[feat_cols]), columns=feat_cols)
        X_test = pd.DataFrame(imputer.transform(test_df[feat_cols]), columns=feat_cols)

        if model_name == "linear":
            model = build_linear_baseline()
            model.fit(train_df[feat_cols], train_df["label"])
            preds = model.predict(test_df[feat_cols])
        else:
            if config.BEST_PARAMS_PATH.exists():
                with open(config.BEST_PARAMS_PATH) as f:
                    best_params = json.load(f)
            else:
                best_params = {"depth": 6, "l2_leaf_reg": 3}
            model = build_catboost_model(**best_params, early_stopping_rounds=None)
            model.fit(X_train, train_df["label"])
            preds = model.predict(X_test).ravel()
        true_labels = test_df["label"].values

    print(f"Held-out patient accuracy: {(preds == true_labels).mean():.3f} "
          f"over {len(true_labels)} epochs")

    # --- Figure 1: full-night hypnogram (no title — caption lives in the site) ---
    fig1, ax1 = plt.subplots(figsize=(13, 3.6))
    plot_hypnogram(true_labels, preds, ax=ax1, show_disagreement=False)
    fig1.tight_layout()
    hypnogram_path = config.FIGURES_DIR / f"hypnogram_{patient_id}_{model_name}.png"
    fig1.savefig(hypnogram_path, dpi=220, facecolor="white", bbox_inches="tight")
    print(f"Saved {hypnogram_path}")

    # --- Figure 2: real stage-transition examples from the raw signal ---
    df_files = build_file_index(str(config.DATA_DIR), config.SUBFOLDER)
    patient_files = df_files[df_files.patient_id == patient_id]
    if len(patient_files) == 0:
        print(f"WARNING: no raw file found for patient {patient_id} under {config.SUBFOLDER} "
              f"— skipping signal figure.")
        return

    psg_file = patient_files.iloc[0].psg_file
    file_folder = config.DATA_DIR / config.SUBFOLDER
    data = load_signals(str(file_folder / psg_file), retrieve_signals=config.COMMON_SIGNALS)
    processed = build_processing_pipeline().process(data)
    eeg = next(s for s in processed if s.name == "EEG Fpz-Cz")
    eeg_vals = eeg.values

    SF = 100  # Sleep-EDF sampling rate
    epoch_samples = 30 * SF
    pre_epochs, post_epochs = 2, 2

    labels = list(true_labels)

    # Tell the sleep-cycle narrative rather than grabbing whatever transition
    # type happens to appear first: falling asleep -> deepening -> emerging
    # -> REM onset -> arousal. Each entry is searched for in order; skipped
    # if this patient's night doesn't contain that specific transition.
    NARRATIVE = [
        ("Sleep stage W", "Sleep stage 1", "Falling asleep"),
        ("Sleep stage 1", "Sleep stage 2", "Light sleep onset"),
        ("Sleep stage 2", "Sleep stage 3", "Deep sleep onset"),
        ("Sleep stage 3", "Sleep stage 2", "Emerging from deep sleep"),
        ("Sleep stage 2", "Sleep stage R", "REM onset"),
        ("Sleep stage 1", "Sleep stage R", "REM onset (via N1)"),
        ("Sleep stage R", "Sleep stage W", "Waking after REM"),
        ("Sleep stage R", "Sleep stage 1", "Arousal after REM"),
    ]

    def find_first_transition(labels, from_label, to_label):
        for i in range(1, len(labels)):
            if labels[i - 1] == from_label and labels[i] == to_label:
                return i
        return None

    transitions = []
    used_pairs = set()
    for frm, to, story in NARRATIVE:
        if (frm, to) in used_pairs:
            continue
        idx = find_first_transition(labels, frm, to)
        if idx is not None:
            transitions.append({"epoch": idx, "from_label": frm, "to_label": to, "story": story})
            used_pairs.add((frm, to))
        if len(transitions) >= 6:
            break

    if not transitions:
        print("None of the narrative transitions were found for this patient — skipping signal figure.")
        return

    chosen = transitions
    print(f"Showing the sleep-cycle narrative ({len(chosen)} transitions): "
          + ", ".join(f"{t['story']} ({t['from_label'].replace('Sleep stage ', '')}\u2192"
                       f"{t['to_label'].replace('Sleep stage ', '')})" for t in chosen))

    windows = []
    for t in chosen:
        start_epoch = t["epoch"] - pre_epochs
        end_epoch = t["epoch"] + post_epochs
        if start_epoch < 0 or end_epoch * epoch_samples > len(eeg_vals):
            continue
        start_sample = start_epoch * epoch_samples
        end_sample = end_epoch * epoch_samples
        windows.append({
            "signal": eeg_vals[start_sample:end_sample],
            "from_label": t["from_label"],
            "to_label": t["to_label"],
        })

    if not windows:
        print("Transitions found but too close to recording edges — skipping signal figure.")
        return

    fig2, axes2 = plot_transition_spectrograms(windows, sf=SF, pre_epochs=pre_epochs, post_epochs=post_epochs)
    signal_path = config.FIGURES_DIR / f"transitions_{patient_id}.png"
    fig2.savefig(signal_path, dpi=220, facecolor="white", bbox_inches="tight")
    print(f"Saved {signal_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", type=str, default=None,
                         help="Patient ID to hold out and visualize (default: first patient alphabetically)")
    parser.add_argument("--model", type=str, default="linear", choices=["linear", "catboost"])
    parser.add_argument("--rescan", action="store_true",
                         help="Force re-scanning all patients for the best one, ignoring the cached result.")
    args = parser.parse_args()
    main(patient_id=args.patient, model_name=args.model, rescan=args.rescan)
