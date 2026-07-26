import json
from pathlib import Path

MODEL_DIR = Path("model")

with open(MODEL_DIR / "metrics.json", "r") as f:
    MODEL_INFO = json.load(f)