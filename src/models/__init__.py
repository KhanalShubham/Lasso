"""Statistical and Machine Learning Model Architecture for EPL Match Outcome Prediction.

Contains implementations for:
1. MLE Multinomial Logistic Regression (mle_model.py)
2. Ridge L2 Multinomial Logistic Regression (ridge_model.py)
3. Lasso L1 Multinomial Logistic Regression (lasso_model.py)
"""

from .mle_model import MLEMultinomialLogistic
from .ridge_model import RidgeLogisticRegression
from .lasso_model import LassoLogisticRegression

__all__ = [
    "MLEMultinomialLogistic",
    "RidgeLogisticRegression",
    "LassoLogisticRegression",
]
