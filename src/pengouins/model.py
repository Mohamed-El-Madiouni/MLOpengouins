from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd


def train_model(  X_train : pd.DataFrame
                , y_train: pd.Series):
    """Train a model on the training data."""
    logi = LogisticRegression()
    logi.fit(X_train, y_train)
    return logi

def evaluate_model( model
                    , X_test: pd.DataFrame
                    , y_test: pd.Series) -> float:
    """Evaluate the model on the test data and return accuracy."""
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    return score