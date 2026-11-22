
from pathlib import Path
import joblib

# Project folders
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

# Load saved models
@staticmethod
def load_dengue_models():
    models = {
        2: joblib.load(MODELS_DIR / "dengue_model_2_weeks.pkl"),
        3: joblib.load(MODELS_DIR / "dengue_model_3_weeks.pkl"),
        4: joblib.load(MODELS_DIR / "dengue_model_4_weeks.pkl")
    }

    return models


def classify_dengue_risk(predicted_cases):
    if predicted_cases < 1:
        return "Low"
    elif predicted_cases <= 4:
        return "Medium"
    elif predicted_cases <= 10:
        return "High"
    else:
        return "Critical"
