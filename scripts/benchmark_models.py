import json
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)

# ==========================================================
# PATHS
# ==========================================================

DATA_PATH = r"./data/gtd.csv"

MODEL1_PATH = r"./model/xgb_gtd_model.json"
MODEL2_PATH = r"./model/xgb_gtd_model2.json"

# ==========================================================
# LOAD DATA
# ==========================================================

cols = [
    "iyear","imonth","country_txt","region_txt",
    "region","country","latitude","longitude",
    "attacktype1","attacktype1_txt",
    "targtype1","targtype1_txt",
    "weaptype1","weaptype1_txt",
    "gname","success","nkill","nwound"
]

df = pd.read_csv(
    DATA_PATH,
    usecols=cols,
    encoding="ISO-8859-1",
    low_memory=False
)

df["nkill"] = df["nkill"].fillna(0)
df["nwound"] = df["nwound"].fillna(0)

df["total_casualties"] = df["nkill"] + df["nwound"]
df["log_casualties"] = np.log1p(df["total_casualties"])

# ==========================================================
# FEATURE ENGINEERING
# ==========================================================

D = df.copy()

for col in [
    "region",
    "country",
    "attacktype1",
    "targtype1",
    "weaptype1"
]:
    freq = D[col].value_counts()
    D[col + "_freq"] = D[col].map(freq).astype(float)

D["region_attack"] = (
    D["region"].astype(str)
    + "_"
    + D["attacktype1"].astype(str)
)

freq = D["region_attack"].value_counts()

D["region_attack_freq"] = D["region_attack"].map(freq).astype(float)

D["region_mean"] = np.log1p(
    D.groupby("region")["total_casualties"].transform("mean")
)

D["attack_mean"] = np.log1p(
    D.groupby("attacktype1")["total_casualties"].transform("mean")
)

D["country_mean"] = np.log1p(
    D.groupby("country")["total_casualties"].transform("mean")
)

for col in [
    "region",
    "country",
    "attacktype1",
    "targtype1",
    "weaptype1"
]:
    D[col + "_cat"] = (
        D[col]
        .astype("category")
        .cat.codes
    )

D = D.sort_values(["country","iyear"])

D["country_5yr_mean"] = (
    D.groupby("country")["total_casualties"]
    .transform(lambda x: x.rolling(5, min_periods=1).mean())
)

D["country_5yr_mean"] = np.log1p(D["country_5yr_mean"])

D["year_trend"] = (
    D["iyear"] - D["iyear"].min()
) / (
    D["iyear"].max() - D["iyear"].min()
)

features = [
    "iyear",
    "imonth",
    "region_freq",
    "country_freq",
    "attacktype1_freq",
    "targtype1_freq",
    "weaptype1_freq",
    "success",
    "region_attack_freq",
    "region_cat",
    "country_cat",
    "attacktype1_cat",
    "targtype1_cat",
    "weaptype1_cat",
    "region_mean",
    "attack_mean",
    "country_mean",
    "year_trend",
    "country_5yr_mean",
]

train_mask = D["iyear"] <= 2018

X_test = D.loc[~train_mask, features]

# Keep target on log scale (same as dashboard)
y_test = D.loc[~train_mask, "log_casualties"]

# ==========================================================
# EVALUATION FUNCTION
# ==========================================================

import os
import matplotlib.pyplot as plt

EXPORT_DIR = "./model_comparison"
os.makedirs(EXPORT_DIR, exist_ok=True)

def evaluate(model_path):

    model = XGBRegressor()
    model.load_model(model_path)

    model_name = os.path.splitext(os.path.basename(model_path))[0]

    # -----------------------------
    # Predictions
    # -----------------------------
    pred_log = model.predict(X_test)

    y_test_log = D.loc[~train_mask, "log_casualties"]

    pred_actual = np.expm1(pred_log)
    y_actual = np.expm1(y_test_log)

    # -----------------------------
    # Metrics
    # -----------------------------
    r2 = r2_score(y_test_log, pred_log)
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
        np.log1p(y_actual),
        np.log1p(pred_actual),
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
        os.path.join(
            EXPORT_DIR,
            f"{model_name}_actual_vs_predicted.png"
        ),
        dpi=400
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
        os.path.join(
            EXPORT_DIR,
            f"{model_name}_residual_plot.png"
        ),
        dpi=400
    )

    plt.close()

    return pred_actual



# ==========================================================
# RUN
# ==========================================================

pred1 = evaluate(MODEL1_PATH)
pred2 = evaluate(MODEL2_PATH)

print("\nAverage prediction difference:",
      np.mean(np.abs(pred1 - pred2)))

print("Maximum prediction difference:",
      np.max(np.abs(pred1 - pred2)))