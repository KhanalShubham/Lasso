"""Unregularized Maximum Likelihood Estimation (MLE) Multinomial Logistic Regression.

This module implements Model 1 (TSK-015, Gate 5):
- Classical unpenalized multinomial logit (penalty=None, solver='lbfgs')
- Symmetric softmax link over K=3 classes
- statsmodels.MNLogit cross-check for exact Hessian inversion, asymptotic standard errors, and p-values.
"""

from typing import Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm


class MLEMultinomialLogistic:
    """Unregularized MLE Multinomial Logistic Regression Estimator."""

    def __init__(self, tol: float = 1e-5, max_iter: int = 5000, fit_intercept: bool = True):
        self.tol = tol
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.sklearn_model: Optional[LogisticRegression] = None
        self.sm_model: Optional[sm.MNLogit] = None
        self.sm_results = None
        self.feature_names: list[str] = []
        self.classes_: np.ndarray = np.array([0, 1, 2])

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray, feature_names: Optional[list[str]] = None):
        """Fit unregularized MLE via both Scikit-Learn (L-BFGS) and Statsmodels (Newton)."""
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X_arr = X.values
        else:
            self.feature_names = feature_names or [f"x{i}" for i in range(X.shape[1])]
            X_arr = np.asarray(X)

        y_arr = np.asarray(y)

        # 1. Scikit-Learn symmetric softmax fit
        self.sklearn_model = LogisticRegression(
            penalty=None,
            solver="lbfgs",
            tol=self.tol,
            max_iter=self.max_iter,
            fit_intercept=self.fit_intercept,
        )
        self.sklearn_model.fit(X_arr, y_arr)
        self.classes_ = self.sklearn_model.classes_

        # 2. Statsmodels MNLogit reference-class fit (for Hessian, SEs, p-values)
        if self.fit_intercept:
            X_sm = sm.add_constant(X_arr, has_constant="add")
        else:
            X_sm = X_arr

        # Use MNLogit with reference class = 1 (Draw) or default 0
        try:
            self.sm_model = sm.MNLogit(y_arr, X_sm)
            self.sm_results = self.sm_model.fit(method="newton", maxiter=self.max_iter, tol=self.tol, disp=False)
        except Exception as e:
            # Fall back to bfgs or powell if newton encounters ill-conditioned Hessian
            try:
                self.sm_results = self.sm_model.fit(method="lbfgs", maxiter=self.max_iter, pgtol=self.tol, disp=False)
            except Exception as e2:
                print(f"Warning: statsmodels MNLogit could not converge: {e2}")
                self.sm_results = None

        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict class probabilities via symmetric softmax."""
        if self.sklearn_model is None:
            raise RuntimeError("Model has not been fitted.")
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        return self.sklearn_model.predict_proba(X_arr)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict discrete class outcome."""
        if self.sklearn_model is None:
            raise RuntimeError("Model has not been fitted.")
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        return self.sklearn_model.predict(X_arr)

    def get_symmetric_coefficients(self) -> pd.DataFrame:
        """Return symmetric coefficient matrix across all K=3 classes."""
        if self.sklearn_model is None:
            raise RuntimeError("Model has not been fitted.")
        coefs = self.sklearn_model.coef_  # Shape (3, p)
        intercepts = self.sklearn_model.intercept_  # Shape (3,)

        df_coef = pd.DataFrame(coefs, index=["Away Win (0)", "Draw (1)", "Home Win (2)"], columns=self.feature_names)
        df_coef.insert(0, "Intercept", intercepts)
        return df_coef

    def get_contrast_coefficients(self, reference_class: int = 1) -> pd.DataFrame:
        """Return coefficients expressed as contrasts relative to a reference class (default 1 = Draw)."""
        if self.sklearn_model is None:
            raise RuntimeError("Model has not been fitted.")
        coefs = self.sklearn_model.coef_  # Shape (3, p)
        intercepts = self.sklearn_model.intercept_

        full_params = np.column_stack([intercepts, coefs])  # Shape (3, p+1)
        ref_params = full_params[reference_class, :]

        contrasts = full_params - ref_params
        col_names = ["Intercept"] + self.feature_names

        classes = ["Away Win (0)", "Draw (1)", "Home Win (2)"]
        rows = [c for idx, c in enumerate(classes) if idx != reference_class]
        contrast_matrix = np.delete(contrasts, reference_class, axis=0)

        return pd.DataFrame(contrast_matrix, index=rows, columns=col_names)
