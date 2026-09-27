"""Nested group-aware cross-validation: hyperparameter search + full
evaluation, both with feature selection and early stopping fit fresh inside
every outer fold (see feature_selection.py and modeling.py for what each
piece fixes)."""
import time

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from config import RANDOM_STATE, VALID_LABELS
from src.feature_selection import select_features
from src.modeling import build_catboost_model, build_linear_baseline


def _inner_split(train_df: pd.DataFrame, test_size: float = 0.2):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
    inner_train_idx, inner_val_idx = next(splitter.split(train_df, groups=train_df.patient_id))
    return train_df.iloc[inner_train_idx], train_df.iloc[inner_val_idx]


def quick_hparam_search(df_feats: pd.DataFrame, feat_cols: list, n_splits: int = 5) -> dict:
    """Small grid search over (depth, l2_leaf_reg) using grouped CV, so the
    full run below uses a tuned config instead of a guess."""
    grid = [
        {"depth": 4, "l2_leaf_reg": 3},
        {"depth": 4, "l2_leaf_reg": 10},
        {"depth": 6, "l2_leaf_reg": 3},
        {"depth": 6, "l2_leaf_reg": 10},
    ]
    gkfold = GroupKFold(n_splits=n_splits)
    results = []

    for params in grid:
        fold_scores = []
        for train_idx, test_idx in gkfold.split(df_feats, groups=df_feats.patient_id):
            train_df, test_df = df_feats.iloc[train_idx], df_feats.iloc[test_idx]
            inner_train, inner_val = _inner_split(train_df)

            imputer = SimpleImputer().fit(inner_train[feat_cols])
            X_tr = pd.DataFrame(imputer.transform(inner_train[feat_cols]), columns=feat_cols)
            X_val = pd.DataFrame(imputer.transform(inner_val[feat_cols]), columns=feat_cols)
            X_te = pd.DataFrame(imputer.transform(test_df[feat_cols]), columns=feat_cols)

            model = build_catboost_model(**params)
            model.fit(X_tr, inner_train["label"], eval_set=(X_val, inner_val["label"]), use_best_model=True)
            preds = model.predict(X_te).ravel()
            fold_scores.append(f1_score(test_df["label"], preds, average="macro"))

        results.append({**params, "mean_macro_f1": float(np.mean(fold_scores))})
        print(f"  depth={params['depth']:<2} l2_leaf_reg={params['l2_leaf_reg']:<3} "
              f"-> mean macro F1 = {np.mean(fold_scores):.4f}")

    best = max(results, key=lambda r: r["mean_macro_f1"])
    print(f"\nBest config: depth={best['depth']}, l2_leaf_reg={best['l2_leaf_reg']} "
          f"(mean macro F1 = {best['mean_macro_f1']:.4f})")
    return {"depth": best["depth"], "l2_leaf_reg": best["l2_leaf_reg"]}


def run_nested_cv(df_feats: pd.DataFrame, feat_cols: list, best_params: dict, n_splits: int = 22) -> dict:
    """Full outer-fold evaluation: feature selection + early-stopped
    CatBoost fit fresh per fold, pooled out-of-fold predictions for the
    final metrics."""
    gkfold = GroupKFold(n_splits=n_splits)

    train_f1, train_bal_acc, train_acc, train_ll = [], [], [], []
    all_preds, all_probas, all_trues = [], [], []

    t0 = time.time()
    for fold_ix, (train_idx, test_idx) in enumerate(gkfold.split(df_feats, groups=df_feats.patient_id)):
        train_df, test_df = df_feats.iloc[train_idx], df_feats.iloc[test_idx]
        inner_train, inner_val = _inner_split(train_df)

        selected_cols = select_features(inner_train[feat_cols], inner_train["label"])

        imputer = SimpleImputer().fit(inner_train[selected_cols])
        X_tr = pd.DataFrame(imputer.transform(inner_train[selected_cols]), columns=selected_cols)
        X_val = pd.DataFrame(imputer.transform(inner_val[selected_cols]), columns=selected_cols)
        X_te = pd.DataFrame(imputer.transform(test_df[selected_cols]), columns=selected_cols)

        model = build_catboost_model(**best_params)
        model.fit(X_tr, inner_train["label"], eval_set=(X_val, inner_val["label"]), use_best_model=True)

        train_preds = model.predict(X_tr).ravel()
        train_probas = model.predict_proba(X_tr)
        train_f1.append(f1_score(inner_train["label"], train_preds, average="macro"))
        train_bal_acc.append(balanced_accuracy_score(inner_train["label"], train_preds))
        train_acc.append(accuracy_score(inner_train["label"], train_preds))
        train_ll.append(log_loss(inner_train["label"], train_probas, labels=model.classes_))

        test_preds = model.predict(X_te).ravel()
        all_preds.append(test_preds)
        all_probas.append(model.predict_proba(X_te))
        all_trues.append(test_df["label"].values)

        print(f"Fold {fold_ix + 1}/{n_splits} done "
              f"({len(selected_cols)} features, best_iteration={model.get_best_iteration()}) "
              f"- {time.time() - t0:.0f}s elapsed")

    preds = np.hstack(all_preds)
    trues = np.hstack(all_trues)
    probas = np.vstack(all_probas)

    metrics = {
        "train": {
            "macro_f1": float(np.mean(train_f1)),
            "balanced_accuracy": float(np.mean(train_bal_acc)),
            "accuracy": float(np.mean(train_acc)),
            "log_loss": float(np.mean(train_ll)),
        },
        "test": {
            "macro_f1": float(f1_score(trues, preds, average="macro")),
            "balanced_accuracy": float(balanced_accuracy_score(trues, preds)),
            "accuracy": float(accuracy_score(trues, preds)),
            "kappa": float(cohen_kappa_score(trues, preds)),
            "log_loss": float(log_loss(trues, probas, labels=sorted(set(trues)))),
        },
    }

    print(f"\n{n_splits}-FOLD: TRAIN (mean across folds)")
    for k, v in metrics["train"].items():
        print(f"  {k:<18}", round(v, 4))

    print(f"\n{n_splits}-FOLD: TEST (pooled out-of-fold predictions)")
    for k, v in metrics["test"].items():
        print(f"  {k:<18}", round(v, 4))

    print("\n", classification_report(trues, preds))
    cm = pd.DataFrame(
        confusion_matrix(trues, preds, labels=VALID_LABELS),
        index=VALID_LABELS, columns=VALID_LABELS,
    )
    print("\nConfusion matrix (rows=true, cols=pred):\n", cm)

    return {"preds": preds, "trues": trues, "probas": probas, "metrics": metrics, "confusion_matrix": cm}


def run_linear_baseline_cv(df_feats: pd.DataFrame, feat_cols: list, n_splits: int = 22) -> dict:
    """Same GroupKFold structure as run_nested_cv, but for the linear SGD
    baseline — no early stopping (not applicable to SGD) and no feature
    selection (the original notebook's linear pipeline didn't use it either,
    and it wasn't the model that was overfitting). This gives an
    apples-to-apples comparison against the tuned CatBoost run on the same
    folds and same rebuilt feature set."""
    gkfold = GroupKFold(n_splits=n_splits)

    train_f1, train_bal_acc, train_acc, train_ll = [], [], [], []
    all_preds, all_probas, all_trues = [], [], []

    t0 = time.time()
    for fold_ix, (train_idx, test_idx) in enumerate(gkfold.split(df_feats, groups=df_feats.patient_id)):
        train_df, test_df = df_feats.iloc[train_idx], df_feats.iloc[test_idx]

        model = build_linear_baseline()
        model.fit(train_df[feat_cols], train_df["label"])

        train_preds = model.predict(train_df[feat_cols])
        train_probas = model.predict_proba(train_df[feat_cols])
        train_f1.append(f1_score(train_df["label"], train_preds, average="macro"))
        train_bal_acc.append(balanced_accuracy_score(train_df["label"], train_preds))
        train_acc.append(accuracy_score(train_df["label"], train_preds))
        train_ll.append(log_loss(train_df["label"], train_probas, labels=model.classes_))

        test_preds = model.predict(test_df[feat_cols])
        all_preds.append(test_preds)
        all_probas.append(model.predict_proba(test_df[feat_cols]))
        all_trues.append(test_df["label"].values)

        print(f"Fold {fold_ix + 1}/{n_splits} done - {time.time() - t0:.0f}s elapsed")

    preds = np.hstack(all_preds)
    trues = np.hstack(all_trues)
    probas = np.vstack(all_probas)

    metrics = {
        "train": {
            "macro_f1": float(np.mean(train_f1)),
            "balanced_accuracy": float(np.mean(train_bal_acc)),
            "accuracy": float(np.mean(train_acc)),
            "log_loss": float(np.mean(train_ll)),
        },
        "test": {
            "macro_f1": float(f1_score(trues, preds, average="macro")),
            "balanced_accuracy": float(balanced_accuracy_score(trues, preds)),
            "accuracy": float(accuracy_score(trues, preds)),
            "kappa": float(cohen_kappa_score(trues, preds)),
            "log_loss": float(log_loss(trues, probas, labels=sorted(set(trues)))),
        },
    }

    print(f"\n{n_splits}-FOLD LINEAR BASELINE: TRAIN (mean across folds)")
    for k, v in metrics["train"].items():
        print(f"  {k:<18}", round(v, 4))

    print(f"\n{n_splits}-FOLD LINEAR BASELINE: TEST (pooled out-of-fold predictions)")
    for k, v in metrics["test"].items():
        print(f"  {k:<18}", round(v, 4))

    print("\n", classification_report(trues, preds))
    cm = pd.DataFrame(
        confusion_matrix(trues, preds, labels=VALID_LABELS),
        index=VALID_LABELS, columns=VALID_LABELS,
    )
    print("\nConfusion matrix (rows=true, cols=pred):\n", cm)

    return {"preds": preds, "trues": trues, "probas": probas, "metrics": metrics, "confusion_matrix": cm}
