"""Model definitions.

build_catboost_model() is the fixed version of the notebook's untuned
`CatBoostClassifier(verbose=0, depth=5, n_estimators=2_000, random_state=0)`
— see evaluation.py for how early stopping and class weighting get applied.

build_linear_baseline() is unchanged from the notebook's linear pipeline
(cell 68) — it wasn't overfitting (5-point train/test gap) so left as-is;
kept here for side-by-side comparison against the fixed CatBoost model.
"""
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer

from config import RANDOM_STATE


def build_linear_baseline() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer()),
            ("scale", QuantileTransformer(n_quantiles=100, subsample=200_000, random_state=RANDOM_STATE)),
            (
                "linear_model",
                SGDClassifier(
                    loss="log_loss",
                    average=True,
                    class_weight="balanced",
                    n_jobs=5,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_catboost_model(depth: int, l2_leaf_reg: float, n_estimators: int = 2000,
                          learning_rate: float = 0.05, early_stopping_rounds: int = 100) -> CatBoostClassifier:
    """CatBoost with early stopping + balanced class weights + macro-F1
    aligned eval metric. Requires an eval_set at .fit() time — see
    evaluation.py for the grouped train/val split that provides it."""
    return CatBoostClassifier(
        verbose=0,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        auto_class_weights="Balanced",
        eval_metric="TotalF1",
        early_stopping_rounds=early_stopping_rounds,
        random_state=RANDOM_STATE,
    )
