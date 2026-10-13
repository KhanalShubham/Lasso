"""Ridge (L2 Regularized) Multinomial Logistic Regression.

This module implements Model 2 (TSK-016, Gate 5):
- L2 Euclidean shrinkage penalty
- Scikit-Learn L-BFGS solver with C = 1 / (2 * n * lambda)
- Symmetric softmax link over K=3 classes
- Effective degrees of freedom df(lambda) computation.
"""

from typing import Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


class RidgeLogisticRegression:
    """Ridge (L2) Multinomial Logistic Regression Estimator."""

    def __init__(
        self,
        lambda_val: float = 1.0,
        n_train: int = 1002,
        tol: float = 1e-5,
        max_iter: int = 5000,
        fit_intercept: bool = True,
    ):
        self.lambda_val = float(lambda_val)
        self.n_train = int(n_train)
        self.tol = tol
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept

        # Map theoretical lambda to scikit-learn C: C = 1 / (2 * n * lambda)
        if self.lambda_val <= 0:
            raise ValueError("lambda_val must be strictly positive.")
        self.C = 1.0 / (2.0 * self.n_train * self.lambda_val)

        self.model = LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            C=self.C,
            tol=self.tol,
            max_iter=self.max_iter,
            fit_intercept=self.fit_intercept,
        )
        self.feature_names: list[str] = []
        self.classes_: np.ndarray = np.array([0, 1, 2])
        self.df_lambda_: Optional[float] = None

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray, feature_names: Optional[list[str]] = None):
        """Fit Ridge multinomial model using L-BFGS."""
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X_arr = X.values
        else:
            self.feature_names = feature_names or [f"x{i}" for i in range(X.shape[1])]
            X_arr = np.asarray(X)

        y_arr = np.asarray(y)

        self.model.fit(X_arr, y_arr)
        self.classes_ = self.model.classes_

        # Compute effective degrees of freedom: df(lambda) = tr((X^T W X + 2n*lambda*I)^-1 X^T W X)
        probs = self.model.predict_proba(X_arr)
        # Average binary variance per sample across 3 classes
        w_diag = np.mean([probs[:, c] * (1.0 - probs[:, c]) for c in range(3)], axis=0)
        W_sqrt = np.sqrt(w_diag)[:, np.newaxis]
        X_w = X_arr * W_sqrt  # Shape (n, p)

        XtWX = X_w.T @ X_w  # Shape (p, p)
        p_dim = XtWX.shape[0]
        penalty_matrix = 2.0 * self.n_train * self.lambda_val * np.eye(p_dim)

        try:
            inv_matrix = np.linalg.inv(XtWX + penalty_matrix)
            self.df_lambda_ = float(np.trace(inv_matrix @ XtWX))
            if self.fit_intercept:
                # Add 3 unpenalized intercept parameters
                self.df_lambda_ += 3.0
        except np.linalg.LinAlgError:
            self.df_lambda_ = float(p_dim)

        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict class probabilities via symmetric softmax."""
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        return self.model.predict_proba(X_arr)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict discrete class outcome."""
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        return self.model.predict(X_arr)

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
