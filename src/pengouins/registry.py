"""
save and load models using pickle.
"""

import os
import pickle
from pathlib import Path

import loguru

logger = loguru.logger


def save_model(model, filepath):
    """Saves the model to the specified filepath using pickle."""
    path = Path("models", filepath)
    if not os.path.exists("models"):
        os.makedirs("models")
    logger.info(f"Saving model to {path}...")
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(filepath):
    """Loads the model from the specified filepath using pickle."""
    path = Path("models", filepath)

    with open(path, "rb") as f:
        logi_loaded = pickle.load(f)
    return logi_loaded
