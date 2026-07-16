from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"
EXPORT_DIR = ROOT / "exports"

DATA_PATH = DATA_DIR / "gtd.csv"

MODEL_PATH = MODEL_DIR / "xgb_gtd_model.json"
FEATURES_PATH = MODEL_DIR / "features.txt"
METRICS_PATH = MODEL_DIR / "metrics.json"
COMPARISON_PATH = MODEL_DIR / "model_comparison.csv"

DEFAULT_FUTURE_YEAR = 2026
DEFAULT_SAMPLE_SIZE = 300

YEAR_MIN = 2025
YEAR_MAX = 2040

SIZE_MIN = 10
SIZE_MAX = 2000

ATTACK_COLORS = {
    1: "blue",
    2: "green",
    3: "orange",
    4: "purple",
    5: "darkred",
    6: "cadetblue",
    7: "darkpurple",
    8: "darkgreen",
    9: "pink",
    10: "lightblue",
}