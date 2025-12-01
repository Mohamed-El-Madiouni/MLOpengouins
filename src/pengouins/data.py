"""
Load and preprocess data.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

_preprocessor = None

def load_data(path: str) -> pd.DataFrame:
    """Load data from seaborn data,
    put it in cache and return a DataFrame."""
    pingouins = sns.load_dataset("penguins")
    pingouins.drop(columns=["island"], inplace=True)
    return pingouins

def get_X_y(
    df: pd.DataFrame, target_column: str, target:bool = True
) -> tuple[pd.DataFrame, pd.Series]:
    """Split DataFrame into features and target."""
    if target:
        y = df.pop(target_column)
        X = df
    else:
        X = df
        y = None
    return (X, y)

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

def preprocess_data(X: pd.DataFrame
                    ,fit = True) -> pd.DataFrame:
    """Preprocess data: handle missing values, encode categorical variables, scale numerical features."""
    num_pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()) 
    ])
    cat_pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(sparse_output=False, drop="first"))
    ])
    global _preprocessor

    if fit:
        _preprocessor = ColumnTransformer(transformers=[
            ('num', num_pipe, make_column_selector(dtype_include="number")),
            ('cat', cat_pipe, make_column_selector(dtype_include=object))
        ])
        return _preprocessor.fit_transform(X)
    
    return _preprocessor.transform(X)
    

