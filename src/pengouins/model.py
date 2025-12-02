"""
Model training and evaluation functions.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from pengouins.registry import load_model, save_model
import loguru

logger = loguru.logger

def train_model(X_train: pd.DataFrame, y_train: pd.Series):
    """Train a model on the training data."""
    logi = LogisticRegression()
    logi.fit(X_train, y_train)
    save_model(logi, "logistic_regression_model.pkl")
    logger.info("Model trained and saved successfully.")
    return logi


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    """Evaluate the model on the test data and return accuracy."""
    y_pred = model.predict(X_test)
    score = round(accuracy_score(y_test, y_pred), 4)
    logger.info(f"Model accuracy: {score}")
    return score
