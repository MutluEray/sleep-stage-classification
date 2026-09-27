"""Feature reduction — the fix for feeding all ~1000 raw columns into
CatBoost unfiltered.

Both steps here are meant to be called on TRAINING data only, fresh inside
every outer CV fold (see evaluation.py) — never on the full dataset — so
there is no leakage from held-out test patients into the selected feature
set.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

try:
    from powershap import PowerShap
except Exception as e:
    # Broad except (not just ImportError) on purpose: powershap's dependency
    # chain includes shap, which has had NumPy 2.x binary-compatibility
    # issues (AttributeError/_ARRAY_API errors at import time, not a clean
    # ImportError). Any failure here should degrade to the correlation-filter
    # fallback rather than crash the whole pipeline.
    PowerShap = None
    print(
        f"WARNING: powershap unavailable ({type(e).__name__}: {e}) — "
        "falling back to correlation filter only. See the fix suggestions "
        "in the code comments below if you want statistical selection back."
    )


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Drops features that are near-duplicates (|corr| > threshold) of a
    feature already kept. Cheap first pass before the more expensive
    PowerShap step — tsfresh/tsflex produce many highly correlated
    variants of the same underlying signal property."""

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y=None):
        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        self.drop_cols_ = [c for c in upper.columns if any(upper[c] > self.threshold)]
        self.keep_cols_ = [c for c in X.columns if c not in self.drop_cols_]
        return self

    def transform(self, X: pd.DataFrame):
        return X[self.keep_cols_]


def select_features(X_train: pd.DataFrame, y_train: pd.Series, max_features: int = 150):
    """Two-stage selection: correlation filter, then PowerShap. Returns the
    list of column names to keep — call with training-fold data only."""
    corr_filter = CorrelationFilter(threshold=0.95)
    X_reduced = corr_filter.fit_transform(X_train)

    if PowerShap is None:
        return list(X_reduced.columns[:max_features])

    selector = PowerShap(power_alpha=0.01, automatic=True)
    selector.fit(X_reduced.fillna(X_reduced.median()), y_train)
    selected = list(X_reduced.columns[selector.get_support()])

    if len(selected) == 0:
        # PowerShap found nothing significant at this alpha — don't return
        # an empty feature set, fall back to the correlation-filtered columns
        return list(X_reduced.columns[:max_features])
    return selected[:max_features] if max_features else selected
