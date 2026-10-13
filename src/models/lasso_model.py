"""Lasso (L1 Regularized) Multinomial Logistic Regression.

This module implements Model 3 (TSK-017, Gate 5):
- L1 Taxicab sparsity and variable selection penalty
- Scikit-Learn SAGA solver with C = 1 / (n * lambda)
- Symmetric softmax link over K=3 classes
- Exact sparsity identification and feature retention accounting.
"""

from typing import Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


class LassoLogisticRegression:
    """Lasso (L1) Multinomial Logistic Regression Estimator."""

    def __init__(
        self,
        lambda_val: float = 1.0,
        n_train: int = 1002,
        tol: float = 1e-4,
        max_iter: int = 5000,
        fit_intercept: bool = True,
        random_state: int = 42,
    ):
        self.lambda_val = float(lambda_val)
        self.n_train = int(n_train)
        self.tol = tol
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.random_state = random_state

        # Map theoretical lambda to scikit-learn C: C = 1 / (n * lambda)
        if self.lambda_val <= 0:
            raise ValueError("lambda_val must be strictly positive.")
        self.C = 1.0 / (self.n_train * self.lambda_val)

        self.model = LogisticRegression(
            penalty="l1",
            solver="saga",
            C=self.C,
            tol=self.tol,
            max_iter=self.max_iter,
            fit_intercept=self.fit_intercept,
            random_state=self.random_state,
        )
        self.feature_names: list[str] = []
        self.classes_: np.ndarray = np.array([0, 1, 2])

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray, feature_names: Optional[list[str]] = None):
        """Fit Lasso multinomial model using SAGA proximal gradient descent."""
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X_arr = X.values
        else:
            self.feature_names = feature_names or [f"x{i}" for i in range(X.shape[1])]
            X_arr = np.asarray(X)

        y_arr = np.asarray(y)

        self.model.fit(X_arr, y_arr)
        self.classes_ = self.model.classes_
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict class probabilities via symmetric softmax."""
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        return self.model.predict_proba(X_arr)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict discrete class outcome."""
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        return self.model.predict(X_arr)

    def get_sparsity(self, threshold: float = 1e-5) -> dict:
        """Compute exact sparsity metrics and retained feature sets across classes."""
        coefs = self.model.coef_  # Shape (3, p)
        # A predictor is retained if non-zero in at least one class
        is_active = np.any(np.abs(coefs) > threshold, axis=0)  # Shape (p,)
        retained_features = [self.feature_names[i] for i in range(len(self.feature_names)) if is_active[i]]
        eliminated_features = [self.feature_names[i] for i in range(len(self.feature_names)) if not is_active[i]]

        return {
            "n_retained": int(np.sum(is_active)),
            "n_eliminated": int(len(self.feature_names) - np.sum(is_active)),
            "pct_retained": float(np.mean(is_active) * 100.0),
            "retained_features": retained_features,
            "eliminated_features": eliminated_features,
        }

    def get_symmetric_coefficients(self) -> pd.DataFrame:
        """Return symmetric coefficient matrix across all K=3 classes."""
        coefs = self.model.coef_
        intercepts = self.model.intercept_
        df_coef = pd.DataFrame(coefs, index=["Away Win (0)", "Draw (1)", "Home Win (2)"], columns=self.feature_names)
        df_coef.insert(0, "Intercept", intercepts)
        return df_coef

    def get_contrast_coefficients(self, reference_class: int = 1) -> pd.DataFrame:
        """Return coefficients expressed as contrasts relative to a reference class (default 1 = Draw)."""
        coefs = self.model.coef_
        intercepts = self.model.intercept_
        full_params = np.column_stack([intercepts, coefs])
        ref_params = full_params[reference_class, :]

        contrasts = full_params - ref_params
        col_names = ["Intercept"] + self.feature_names

        classes = ["Away Win (0)", "Draw (1)", "Home Win (2)"]
        rows = [c for idx, c in enumerate(classes) if idx != reference_class]
        contrast_matrix = np.delete(contrasts, reference_class, axis=0)

        return pd.DataFrame(contrast_matrix, index=rows, columns=col_names)
