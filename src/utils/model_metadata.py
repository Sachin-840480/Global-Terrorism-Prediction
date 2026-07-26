import json

from src.config import METRICS_PATH

def load_model_metadata():
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)