import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)

from pathlib import Path

from src.data.loader import load_data
from src.data.preprocessing import engineer_features
from src.data.features import ALL_FEATURES
from src.config import BENCHMARK_DIR, BENCHMARK_HISTORY_PATH

# ==========================================================
# PATHS
# ==========================================================

MODEL_PATHS = sorted(
    (
        path
        for path in BENCHMARK_DIR.glob("*.json")
        if path.name != "benchmark_history.json"
    ),
    key=lambda p: p.name
)

if len(MODEL_PATHS) != 2:
    raise ValueError(
        f"Expected exactly 2 model files in {BENCHMARK_DIR}, "
        f"found {len(MODEL_PATHS)}"
    )

MODEL1_PATH = MODEL_PATHS[0]
MODEL2_PATH = MODEL_PATHS[1]

print("Model 1:", MODEL1_PATH)
print("Model 2:", MODEL2_PATH)

# Outputs go here
ROOT = Path(__file__).resolve().parents[1]
EXPORT_DIR = ROOT / "exports" / "benchmark"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# LOAD DATA + FEATURES + PRE-PROCESSING + DATA SPLITTING
# ==========================================================

df = load_data()

D = engineer_features(df)

features = ALL_FEATURES

train_mask = D["iyear"] <= 2018

X_test = D.loc[~train_mask, features]

# Keep target on log scale (same as dashboard)
y_test = D.loc[~train_mask, "log_casualties"]

# ==========================================================
# EVALUATION FUNCTION
# ==========================================================

def evaluate(model_path):

    model = XGBRegressor()
    model.load_model(model_path)

    model_name = model_path.stem

    # Use the features the model was trained with
    model_features = model.get_booster().feature_names

    X_model = D.loc[~train_mask, model_features]

    # -----------------------------
    # Predictions
    # -----------------------------
    pred_log = model.predict(X_model)

    pred_actual = np.expm1(pred_log)
    y_actual = np.expm1(y_test)

    # -----------------------------
    # Metrics
    # -----------------------------
    r2 = r2_score(y_test, pred_log)
    mae = mean_absolute_error(y_actual, pred_actual)
    rmse = np.sqrt(mean_squared_error(y_actual, pred_actual))

    print("=" * 60)
    print(model_name)
    print("=" * 60)
    print(f"R² (log): {r2:.6f}")
    print(f"MAE     : {mae:.6f}")
    print(f"RMSE    : {rmse:.6f}")

    # ==========================================================
    # Actual vs Predicted
    # ==========================================================

    plt.figure(figsize=(8,6))

    plt.scatter(
        y_actual,
        pred_actual,
        alpha=0.30,
        edgecolor="black",
        linewidth=0.3
    )

    m = max(y_actual.max(), pred_actual.max())

    plt.plot(
        [0, m],
        [0, m],
        color="red",
        linewidth=2
    )

    plt.xlabel("Actual Casualties")
    plt.ylabel("Predicted Casualties")
    plt.title(f"Actual vs Predicted\n{model_name}")

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        EXPORT_DIR / f"{model_name}_actual_vs_predicted.png",
        dpi=400,
        bbox_inches="tight"
    )

    plt.close()

    # ==========================================================
    # Residual Plot
    # ==========================================================

    residuals = y_actual - pred_actual

    plt.figure(figsize=(8,6))

    plt.scatter(
        pred_actual,
        residuals,
        alpha=0.30,
        edgecolor="black",
        linewidth=0.3
    )

    plt.axhline(
        0,
        color="red",
        linestyle="--",
        linewidth=2
    )

    plt.xlabel("Predicted Casualties")
    plt.ylabel("Residuals")
    plt.title(f"Residual Plot\n{model_name}")

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        EXPORT_DIR / f"{model_name}_residual_plot.png",
        dpi=400,
        bbox_inches="tight"
    )

    plt.close()

    return pred_actual, r2, mae, rmse, len(model_features)



# ==========================================================
# RUN
# ==========================================================

pred1, r2_model1, mae_model1, rmse_model1, features_model1 = evaluate(MODEL1_PATH)

pred2, r2_model2, mae_model2, rmse_model2, features_model2 = evaluate(MODEL2_PATH)

results = pd.DataFrame([
    {
        "Model": MODEL1_PATH.name,
        "R2": r2_model1,
        "MAE": mae_model1,
        "RMSE": rmse_model1
    },
    {
        "Model": MODEL2_PATH.name,
        "R2": r2_model2,
        "MAE": mae_model2,
        "RMSE": rmse_model2
    }
])

# ==========================================================
# SAVE BENCHMARK HISTORY
# ==========================================================

# TO JSON
import json

HISTORY_PATH = BENCHMARK_HISTORY_PATH

if HISTORY_PATH.exists():
    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            history = json.load(f)

        # Ensure history is actually a list
        if not isinstance(history, list):
            history = []

    except (json.JSONDecodeError, OSError):
        # Empty or invalid history file
        history = []
else:
    history = []

benchmark_number = len(history) + 1

history.append({
    "benchmark": benchmark_number,

    "model1": {
        "name": MODEL1_PATH.name,
        "r2": float(r2_model1),
        "mae": float(mae_model1),
        "rmse": float(rmse_model1),
        "features": features_model1
    },

    "model2": {
        "name": MODEL2_PATH.name,
        "r2": float(r2_model2),
        "mae": float(mae_model2),
        "rmse": float(rmse_model2),
        "features": features_model2
    }
})

with open(HISTORY_PATH, "w", encoding="utf-8") as f:
    json.dump(history, f, indent=4)


# To CSV
results.to_csv(
    EXPORT_DIR / "benchmark_results.csv",
    index=False
)


# User Information
print("\nAverage prediction difference:",
      np.mean(np.abs(pred1 - pred2)))

print("Maximum prediction difference:",
      np.max(np.abs(pred1 - pred2)))

print("\n" + "=" * 65)
print("Model Comparison")
print("=" * 65)
print(results.to_string(index=False))

better_model = results.loc[results["R2"].idxmax(), "Model"]

print(f"\nBest Model : {better_model}")