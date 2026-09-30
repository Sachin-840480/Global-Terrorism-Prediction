import os
import json
import traceback

import numpy as np
import pandas as pd

from datetime import datetime

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from xgboost import XGBRegressor

from src.config import MODEL_DIR

from src.data.preprocessing import engineer_features
from src.data.features import ALL_FEATURES

# ================================================================
# Model Training — XGBoost Regressor
# ================================================================

def train_xgboost_cpu(df, output_dir=MODEL_DIR, model_name="xgb_gtd_model.json"):
    """
    Optimized CPU-only XGBoost training.
    Saves model to disk for future runs.
    """

    output_dir = os.fspath(output_dir)

    model_path = os.path.join(output_dir, model_name)
    features_path = os.path.join(output_dir, "features.txt")
    metrics_path = os.path.join(output_dir, "metrics.json")

    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------
    # 0. If model already exists -> Load instead of train
    # -------------------------------------------------
    if os.path.exists(model_path):
        try:
            model = XGBRegressor()
            model.load_model(model_path)
            print("⚡ Loaded saved model from disk.")

            features = None

            # always load the saved features
            if os.path.exists(features_path):
                with open(features_path, "r", encoding="utf-8") as f:
                    features = [line.strip() for line in f.readlines() if line.strip()]

            loaded_r2 = None
            loaded_mae = None

            if os.path.exists(metrics_path):
                with open(metrics_path, "r", encoding="utf-8") as f:
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

    D = engineer_features(df)

    # -------------------------------------------------
    # 2. Split
    # -------------------------------------------------

    # Import features from the features.py file.
    features = ALL_FEATURES 

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


    # model = XGBRegressor(
    #         n_estimators=2500,
    #         learning_rate=0.01,
    #         max_depth=8,
    #         min_child_weight=5,          # Prevents creating splits for single outlier events
    #         subsample=0.8,
    #         colsample_bytree=0.8,
    #         reg_lambda=2.0,              # Increased L2 regularization to combat overfitting
    #         reg_alpha=0.5,               # Increased L1 regularization 
    #         objective="reg:pseudohubererror", # MASSIVE FOR OUTLIERS: Less sensitive to extreme casualty counts than squarederror
    #         tree_method="hist",
    #         n_jobs=n_jobs,
    #         random_state=42,
    #         verbosity=0,
    #         early_stopping_rounds=50     # Stops training if test metrics degrade for 50 rounds
    # )

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
    rmse = np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(pred)))
    mse = mean_squared_error(np.expm1(y_test), np.expm1(pred))

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
    
    os.makedirs(output_dir, exist_ok=True)
    model.save_model(model_path) 

    # Save metrics for the saved model.
    metrics = {
    "model_name": model.__class__.__name__,
    "algorithm": "XGBoost",
    "version": "1.0",

    "r2": float(r2),
    "mae": float(mae),
    "rmse": float(rmse),
    "mse": float(mse),

    "dataset_size": int(len(df)),
    "train_samples": int(len(X_train)),
    "test_samples": int(len(X_test)),

    "num_features": len(features),
    "feature_names": features,

    "training_years": f"{df['iyear'].min()}-{df['iyear'].max()}",
    "train_period": "≤ 2016",
    "test_period": "> 2017",

    "last_trained": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)


    # Save model + features
    with open(features_path, "w", encoding="utf-8") as f:
        for col in features:
            f.write(col + "\n")

    print("💾 Model saved to", model_path)
    print("💾 Metrics saved to", metrics_path)
    print("💾 Features saved to", features_path)

    return model, r2, mae, features

