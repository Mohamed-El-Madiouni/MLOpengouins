"""
Load and preprocess data.
"""

import os
from pathlib import Path

import loguru
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pengouins.registry import load_model, save_model

logger = loguru.logger


def load_data(path: str) -> pd.DataFrame:
    """Load data from seaborn data,
    put it in cache and return a DataFrame."""

    csv_path = Path("data", path)
    if csv_path.exists():
        return pd.read_csv(csv_path)

    pingouins = sns.load_dataset("penguins")
    if not os.path.exists("data"):
        os.makedirs("data")
    pingouins.drop(columns=["island"], inplace=True)
    pingouins.to_csv(csv_path, index=False)

    return pingouins


def get_X_y(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    """Split DataFrame into features and target."""
    if not target_column in df.columns:
        raise Exception("Missing Target in the Dataset")
    y = df.pop(target_column)
    X = df
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into training and testing sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return (X_train, X_test, y_train, y_test)


def preprocess_data(X: pd.DataFrame, fit=True) -> pd.DataFrame:
    """Preprocess data: handle missing values,
    encode categorical variables, scale numerical features."""
    if fit:
        logger.info("Initializing preprocessing pipeline...")
        num_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(sparse_output=False, drop="first")),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_pipe, make_column_selector(dtype_include="number")),
                ("cat", cat_pipe, make_column_selector(dtype_include=object)),
            ]
        )
        preprocessor.fit(X)

        save_model(preprocessor, "preprocessor")

    else:
        preprocessor = load_model("preprocessor")

    X_preproc = preprocessor.transform(X)

    return X_preproc
