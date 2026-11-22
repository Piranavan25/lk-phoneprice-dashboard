"""
utils.py - small helpers shared by the notebook and the Streamlit dashboard.

They live in a .py file (not inside the notebook) so that the saved models can
be loaded again by app.py with joblib.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CAT_COLS = ["brand", "model", "condition", "os", "network", "district", "membership_level"]
NUM_COLS = ["storage_gb", "ram_gb", "storage_missing", "ram_missing", "n_images", "desc_len"]
FEATURES = CAT_COLS + NUM_COLS


class PhoneImputer(BaseEstimator, TransformerMixin):
    """Fills missing RAM / storage using medians learned from the TRAINING data only.

    RAM:     median of the same model -> median of the same brand -> overall median
    Storage: median of the same model -> overall median
    Because it is fitted inside the sklearn pipeline, test rows never influence it.
    """

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.ram_model_ = X.groupby("model")["ram_gb"].median().dropna().to_dict()
        self.ram_brand_ = X.groupby("brand")["ram_gb"].median().dropna().to_dict()
        self.ram_all_ = float(X["ram_gb"].median())
        self.sto_model_ = X.groupby("model")["storage_gb"].median().dropna().to_dict()
        self.sto_all_ = float(X["storage_gb"].median())
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        ram = X["ram_gb"]
        ram = ram.fillna(X["model"].map(self.ram_model_))
        ram = ram.fillna(X["brand"].map(self.ram_brand_))
        X["ram_gb"] = ram.fillna(self.ram_all_)
        sto = X["storage_gb"].fillna(X["model"].map(self.sto_model_))
        X["storage_gb"] = sto.fillna(self.sto_all_)
        return X


def make_preprocessor(scale=False, min_frequency=10):
    """Impute -> one-hot the categoricals (rare categories grouped) -> (optionally) scale numbers.
    scale=True is used for the neural network, trees do not need it."""
    cat = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=min_frequency,
                        sparse_output=False)
    num = StandardScaler() if scale else "passthrough"
    return Pipeline([
        ("impute", PhoneImputer()),
        ("encode", ColumnTransformer([("cat", cat, CAT_COLS), ("num", num, NUM_COLS)])),
    ])


class AverageBlend(BaseEstimator, RegressorMixin):
    """Hybrid 1: weighted average of two fitted models (e.g. Random Forest + ANN).
    w is the weight of model A."""

    def __init__(self, model_a, model_b, w=0.5):
        self.model_a, self.model_b, self.w = model_a, model_b, w

    def fit(self, X, y):
        self.a_ = clone(self.model_a).fit(X, y)
        self.b_ = clone(self.model_b).fit(X, y)
        return self

    def predict(self, X):
        return self.w * self.a_.predict(X) + (1 - self.w) * self.b_.predict(X)


class ResidualHybrid(BaseEstimator, RegressorMixin):
    """Hybrid 2: the base model (Random Forest) predicts first, then a second model
    (ANN) learns the base model's mistakes.  Out-of-fold residuals are used, because
    in-sample forest residuals are unrealistically small."""

    def __init__(self, base_model, residual_model, cv=5, random_state=42):
        self.base_model, self.residual_model = base_model, residual_model
        self.cv, self.random_state = cv, random_state

    def fit(self, X, y):
        kf = KFold(self.cv, shuffle=True, random_state=self.random_state)
        oof = cross_val_predict(clone(self.base_model), X, y, cv=kf)
        self.base_ = clone(self.base_model).fit(X, y)
        self.res_ = clone(self.residual_model).fit(X, np.asarray(y) - oof)
        return self

    def predict(self, X):
        return self.base_.predict(X) + self.res_.predict(X)
