import pickle
import os

def save_model(model, filepath):
    """Saves the model to the specified filepath using pickle."""
    print(os.listdir('./'))
    if not os.path.exists("models"):
        print("in if")
        os.makedirs("models")
    with open(filepath, "wb") as f:
        print("in with")
        pickle.dump(model, f)

def load_model(filepath):
    """Loads the model from the specified filepath using pickle."""
    with open(filepath, "rb") as f:
        logi_loaded = pickle.load(f)
    return logi_loaded