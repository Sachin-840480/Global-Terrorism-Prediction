import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pathlib import Path

from src.config import EXPERIMENTS_DIR
from src.data.loader import load_data
from src.models.trainer import train_xgboost_cpu


# ==========================================================
# FIND NEXT EXPERIMENT NUMBER
# ==========================================================

def get_next_experiment_number():

    existing_numbers = []

    for folder in EXPERIMENTS_DIR.iterdir():

        if not folder.is_dir():
            continue

        if folder.name.startswith("model"):

            number = folder.name.replace("model", "")

            if number.isdigit():
                existing_numbers.append(int(number))

    if not existing_numbers:
        return 1

    return max(existing_numbers) + 1


# ==========================================================
# CREATE EXPERIMENT FOLDER
# ==========================================================

EXPERIMENTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

experiment_number = get_next_experiment_number()

experiment_dir = (
    EXPERIMENTS_DIR /
    f"model{experiment_number}"
)

experiment_dir.mkdir(
    parents=True,
    exist_ok=False
)

print("=" * 60)
print(f"Creating experiment: model{experiment_number}")
print(f"Folder: {experiment_dir}")
print("=" * 60)


# ==========================================================
# LOAD DATA
# ==========================================================

df = load_data()


# ==========================================================
# TRAIN EXPERIMENT
# ==========================================================

train_xgboost_cpu(
    df,
    output_dir=experiment_dir,
    model_name=f"xgb_gtd_model{experiment_number}.json"
)


print("\n" + "=" * 60)
print(f"Experiment model{experiment_number} completed.")
print(f"Saved in: {experiment_dir}")
print("=" * 60)