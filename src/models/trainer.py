import os
import json
import traceback

import numpy as np
import pandas as pd

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from xgboost import XGBRegressor

from src.config import (
    MODEL_DIR,
    MODEL_PATH,
    FEATURES_PATH,
    METRICS_PATH,
)


# ================================================================
# Model Training — XGBoost Regressor
# ================================================================

def train_xgboost_cpu(df):
    """
    Optimized CPU-only XGBoost training.
    Saves model to disk for future runs.
    """

    os.makedirs(MODEL_DIR, exist_ok=True)

    # -------------------------------------------------
    # 0. If model already exists -> Load instead of train
    # -------------------------------------------------
    if os.path.exists(MODEL_PATH):
        try:
            model = XGBRegressor()
            model.load_model(MODEL_PATH)
            print("⚡ Loaded saved model from disk.")

            # always load the saved features
            if os.path.exists(FEATURES_PATH):
                with open(FEATURES_PATH, "r", encoding="utf-8") as f:
                    features = [line.strip() for line in f.readlines() if line.strip()]

            loaded_r2 = None
            loaded_mae = None

            if os.path.exists(METRICS_PATH):
                with open(METRICS_PATH, "r", encoding="utf-8") as f:
                    m = json.load(f)
                    loaded_r2 = m.get("r2")
                    loaded_mae = m.get("mae")

                return model, loaded_r2, loaded_mae, features

            # If model exists but features missing, force retrain
            print("⚠ Model file found but features missing -> retraining.")
        except:
            print("⚠ Saved model file exists but could not load. Retraining...")
            print(traceback.format_exc())

    # -------------------------------------------------
    # 1. Feature engineering
    # -------------------------------------------------
    D = df.copy()

    D["nkill"] = D["nkill"].fillna(0)
    D["nwound"] = D["nwound"].fillna(0)
    D["total_casualties"] = D["nkill"] + D["nwound"]
    D["log_casualties"] = np.log1p(D["total_casualties"])

    # Frequency encoding
    for col in ["region","country","attacktype1","targtype1","weaptype1"]:
        freq = D[col].value_counts()
        D[col + "_freq"] = D[col].map(freq).astype(float)

    # Interaction
    D["region_attack"] = D["region"].astype(str) + "_" + D["attacktype1"].astype(str)
    freq_int = D["region_attack"].value_counts()
    D["region_attack_freq"] = D["region_attack"].map(freq_int).astype(float)

    # Historical means
    D["region_mean"]  = np.log1p(D.groupby("region")["total_casualties"].transform("mean"))
    D["attack_mean"]  = np.log1p(D.groupby("attacktype1")["total_casualties"].transform("mean"))
    D["country_mean"] = np.log1p(D.groupby("country")["total_casualties"].transform("mean"))

    # Category codes
    for col in ["region","country","attacktype1","targtype1","weaptype1"]:
        D[col + "_cat"] = D[col].astype("category").cat.codes

    # Time trend
    ymin, ymax = D["iyear"].min(), D["iyear"].max()
    D["year_trend"] = (D["iyear"] - ymin) / (ymax - ymin)

    # Rolling country mean
    D = D.sort_values(["country","iyear"])
    D["country_5yr_mean"] = (
        D.groupby("country")["total_casualties"]
         .transform(lambda x: x.rolling(5, min_periods=1).mean())
    )
    D["country_5yr_mean"] = np.log1p(D["country_5yr_mean"])

    # -------------------------------------------------
    # 2. Split
    # -------------------------------------------------
    features = [
        "iyear","imonth",
        "region_freq","country_freq","attacktype1_freq",
        "targtype1_freq","weaptype1_freq",
        "success","region_attack_freq",
        "region_cat","country_cat","attacktype1_cat",
        "targtype1_cat","weaptype1_cat",
        "region_mean","attack_mean","country_mean",
        "year_trend","country_5yr_mean"
    ]

    train_mask = D["iyear"] <= 2018

    X_train = D.loc[train_mask, features]
    X_test  = D.loc[~train_mask, features]
    y_train = D.loc[train_mask, "log_casualties"]
    y_test  = D.loc[~train_mask, "log_casualties"]

    # -------------------------------------------------
    # 3. Optimized XGBoost CPU model
    # -------------------------------------------------

    n_jobs = max(1, (os.cpu_count() or 1) - 1)  # leave one core free

    # model = XGBRegressor(
    #     n_estimators=1200,
    #     learning_rate=0.03,
    #     max_depth=6,
    #     subsample=0.9,
    #     colsample_bytree=0.9,
    #     tree_method="hist",         # CPU optimized
    #     n_jobs=n_jobs,              # use all cores except one
    #     random_state=42,
    #     verbosity=0
    # )
    
    model = XGBRegressor(
        n_estimators=2000,
        learning_rate=0.005,
        max_depth=9,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        reg_alpha=0.4,
        tree_method="hist",
        n_jobs=n_jobs,
        random_state=42,
        verbosity=0
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # -------------------------------------------------
    # 4. Metrics
    # -------------------------------------------------
    pred = model.predict(X_test)
    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(np.expm1(y_test), np.expm1(pred))

    # ============================================================
    # OPTIONAL METRIC BLOCK (FOR PPT / DOCUMENTATION) — KEEP COMMENTED
    # ============================================================


    # # Convert regression outputs into severity classes
    # y_test_actual = np.expm1(y_test)
    # y_test_pred = np.expm1(pred)

    # def to_severity(y):
    #     if y <= 1: 
    #         return 0
    #     elif y <= 10:
    #         return 1
    #     else:
    #         return 2

    # y_test_cls = np.array([to_severity(v) for v in y_test_actual])
    # y_pred_cls = np.array([to_severity(v) for v in y_test_pred])

    # cls_accuracy = accuracy_score(y_test_cls, y_pred_cls)
    # cls_precision = precision_score(y_test_cls, y_pred_cls, average='macro')
    # cls_recall = recall_score(y_test_cls, y_pred_cls, average='macro')
    # cls_f1 = f1_score(y_test_cls, y_pred_cls, average='macro')

    # print("\n=== Classification Metrics (Severity Buckets) ===")
    # print(f"Accuracy:  {cls_accuracy:.4f}")
    # print(f"Precision: {cls_precision:.4f}")
    # print(f"Recall:    {cls_recall:.4f}")
    # print(f"F1-Score:  {cls_f1:.4f}")


    # -------------------------------------------------
    # 5. SAVE THE MODEL
    # -------------------------------------------------
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_model(MODEL_PATH) 

    # Save metrics for the saved model.
    metrics = {"r2": float(r2), "mae": float(mae)}
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    # Save model + features
    with open(FEATURES_PATH, "w", encoding="utf-8") as f:
        for col in features:
            f.write(col + "\n")

    print("💾 Model saved to", MODEL_PATH)
    print("💾 Metrics saved to metrics.json")
    print("💾 Features saved to", FEATURES_PATH)

    return model, r2, mae, features

