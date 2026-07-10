# ================================================================
# 🌍 Global Terrorism Dashboard — Streamlit + Fallback Hybrid
# Combines:
#  - Multi-page Streamlit dashboard
#  - Fallback for Jupyter/CLI (inline plots + map saved)
# ================================================================

from os import environ
environ["STREAMLIT_COOKIE_SECRET"] = "dummy_secret"  # ✅ Avoid Streamlit KeyError

import os
import traceback
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import folium
import json

# Optional Streamlit import
try:
    import streamlit as st
    STREAMLIT = True
except Exception:
    STREAMLIT = False

# Optional display for Jupyter fallback
try:
    from IPython.display import display
except ImportError:
    def display(x):
        plt.show()

from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBRegressor

plt.style.use("seaborn-v0_8-darkgrid")


# ================================================================
# 0️⃣ Config
# ================================================================

DATA_PATH = r'./data/gtd.csv'   # change if needed
COMPARISON_PATH = r'./model/model_comparison.csv'

DEFAULT_FUTURE_YEAR = 2026
DEFAULT_SAMPLE_SIZE = 300

YEAR_MIN, YEAR_MAX = 2025, 2040
SIZE_MIN, SIZE_MAX = 10, 2000

# Attack type color palette (for the map)
ATTACK_COLORS = {
    1: 'blue', 2: 'green', 3: 'orange', 4: 'purple',
    5: 'darkred', 6: 'cadetblue', 7: 'darkpurple',
    8: 'darkgreen', 9: 'pink', 10: 'lightblue'
}

MODEL_DIR = "./model"
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_gtd_model.json")
FEATURES_PATH = os.path.join(MODEL_DIR, "features.txt")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
# from random import randint
# _ = randint(4,7)  # for slight boost

# ================================================================
# 1️⃣ Load Dataset (or Cached)
# ================================================================

def load_data_local(path=DATA_PATH):
    cols = [
        'iyear', 'imonth', 'country_txt', 'region_txt',
        'region', 'country', 'latitude', 'longitude',
        'attacktype1', 'attacktype1_txt', 'targtype1', 'targtype1_txt',
        'weaptype1', 'weaptype1_txt', 'gname','success', 'nkill', 'nwound',
    ]
    df = pd.read_csv(path, usecols=cols, encoding='ISO-8859-1', low_memory=False)
    df['nkill'] = df['nkill'].fillna(0)
    df['nwound'] = df['nwound'].fillna(0)
    df['total_casualties'] = df['nkill'] + df['nwound']
    df.dropna(subset=['latitude', 'longitude'], inplace=True)
    return df

if STREAMLIT:
    @st.cache_data(show_spinner=False)
    def cached_load(path=DATA_PATH):
        return load_data_local(path)
    df = cached_load()
else:
    print("🔹 Loading dataset...")
    df = load_data_local()

# ================================================================
# 2️⃣ Model Training — Decision Tree
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


# ================================================================
# 3️⃣ Prediction + Folium Map
# ================================================================

def predict_and_map(df, model, features, future_year, sample_size=300, country_filter=None):
    D = df.copy()
    if country_filter:
        D = D[D["country_txt"].isin(country_filter)]
        if D.empty:
            D = df.copy()

    sample = D.sample(n=min(sample_size, len(D)), random_state=future_year).copy()
    sample["iyear"] = future_year
    sample["nkill"] = sample["nkill"].fillna(0)
    sample["nwound"] = sample["nwound"].fillna(0)
    sample["total_casualties"] = sample["nkill"] + sample["nwound"]
    sample["log_casualties"] = np.log1p(sample["total_casualties"])

    # recreate frequency/interactions using whole df
    for col in ["region", "country", "attacktype1", "targtype1", "weaptype1"]:
        freq = df[col].value_counts().to_dict()
        sample[col + "_freq"] = sample[col].map(freq).fillna(0).astype(float)

    # Interaction frequency (map)
    region_attack_freq_map = (df["region"].astype(str) + "_" + df["attacktype1"].astype(str)).value_counts().to_dict()
    sample["region_attack"] = sample["region"].astype(str) + "_" + sample["attacktype1"].astype(str)
    sample["region_attack_freq"] = sample["region_attack"].map(region_attack_freq_map).fillna(0).astype(float)

    # Historical means: use mapping to avoid misalignment
    region_mean_map = df.groupby("region")["total_casualties"].mean().to_dict()
    attack_mean_map = df.groupby("attacktype1")["total_casualties"].mean().to_dict()
    country_mean_map = df.groupby("country")["total_casualties"].mean().to_dict()

    sample["region_mean"]  = np.log1p(sample["region"].map(region_mean_map).fillna(0).astype(float))
    sample["attack_mean"]  = np.log1p(sample["attacktype1"].map(attack_mean_map).fillna(0).astype(float))
    sample["country_mean"] = np.log1p(sample["country"].map(country_mean_map).fillna(0).astype(float))

    # Category codes (on sample)
    for col in ["region", "country", "attacktype1", "targtype1", "weaptype1"]:
        sample[col + "_cat"] = sample[col].astype("category").cat.codes

    # Year trend relative to original df
    sample["year_trend"] = (future_year - df["iyear"].min()) / max(1, (df["iyear"].max() - df["iyear"].min()))

    # Rolling country 5-year mean (on sample) — use sample grouping (safe)
    sample = sample.sort_values(["country", "iyear"])
    sample["country_5yr_mean"] = (
        sample.groupby("country")["total_casualties"].transform(lambda x: x.rolling(5, min_periods=1).mean())
    )
    sample["country_5yr_mean"] = np.log1p(sample["country_5yr_mean"].fillna(0))

    # Predict
    if model is None:
        sample["predicted_casualties"] = sample["country_mean"].apply(lambda v: np.expm1(v) if pd.notnull(v) else 0)
    else:
        try:
            pred_log = model.predict(sample[features])
            sample["predicted_casualties"] = np.expm1(pred_log).clip(0, None)
        except Exception:
            print("⚠️ Model prediction failed; falling back to country_mean.")
            print(traceback.format_exc())
            sample["predicted_casualties"] = sample["country_mean"].apply(lambda v: np.expm1(v) if pd.notnull(v) else 0)

    # Build folium map
    fmap = folium.Map(location=[sample["latitude"].mean(), sample["longitude"].mean()], zoom_start=2)
    for _, row in sample.iterrows():
        try:
            attack_idx = int(row.get("attacktype1", 0)) if not pd.isna(row.get("attacktype1", None)) else 0
        except Exception:
            attack_idx = 0
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=3 + np.sqrt(max(0, float(row["predicted_casualties"]))),
            color=ATTACK_COLORS.get(attack_idx, 'gray'),
            fill=True, fill_opacity=0.6,
            popup=(f"<b>Country:</b> {row['country_txt']}<br>"
                   f"<b>Attack Type:</b> {row.get('attacktype1_txt', 'unknown')}<br>"
                   f"<b>Predicted Casualties:</b> {int(row['predicted_casualties'])}")
        ).add_to(fmap)

    fmap.get_root().html.add_child(folium.Element("""
        <div style="position: fixed; bottom: 50px; left: 50px; width: 260px;
         border:2px solid grey; z-index:9999; font-size:14px;
         background-color:white; padding: 10px; line-height: 1.4;">
         <b>Attack Type Legend</b><br>
         <i style="color:blue">●</i> Assassination<br>
         <i style="color:green">●</i> Armed Assault<br>
         <i style="color:orange">●</i> Bombing/Explosion<br>
         <i style="color:purple">●</i> Hijacking<br>
         <i style="color:darkred">●</i> Hostage (Kidnapping)<br>
         <i style="color:cadetblue">●</i> Hostage (Barricade)<br>
         <i style="color:darkpurple">●</i> Infrastructure Attack<br>
         <i style="color:darkgreen">●</i> Unknown<br>
         <i style="color:pink">●</i> Other<br>
         <i style="color:lightblue">●</i> Unclassified<br>
        </div>
    """))

    return fmap, sample


# ================================================================
# LOADER FUNCTION FOR MODEL COMPARISON TABLE
# ================================================================

def load_model_comparison():
    if os.path.exists(COMPARISON_PATH):
        return pd.read_csv(COMPARISON_PATH)

    return pd.DataFrame()


# ================================================================
# 4️⃣ Streamlit Dashboard (Main)
# ================================================================

if STREAMLIT:
    st.set_page_config(page_title="Global Terrorism Dashboard", layout="wide")
    st.title("🌍 Global Terrorism — Analytics & Prediction Dashboard")
    st.markdown("""
    Interactive visualization of the **Global Terrorism Database (GTD)** with:
    - Global and regional terrorism trends.  
    - Most affected countries and attack types.  
    - XGBoost regression model predicting casualty severity.  
    - Interactive world map of predicted attacks.
    """)

    # Sidebar
    st.sidebar.header("Controls")
    future_year = st.sidebar.slider("Future Year", YEAR_MIN, YEAR_MAX, DEFAULT_FUTURE_YEAR, 1)
    sample_size = st.sidebar.slider("Sample Size (map)", SIZE_MIN, SIZE_MAX, DEFAULT_SAMPLE_SIZE, 50)
    countries = sorted(df['country_txt'].unique().tolist())
    country_filter = st.sidebar.multiselect("Filter by Country", countries, default=[])

    tab_eda, tab_model, tab_compare, tab_map = st.tabs(["📊 Data Insights", "🧠 Model", "📈 Model Comparison", "🗺️ Map"])

    # EDA Tab
    with tab_eda:
        st.header("📊 Historical Insights")

        st.markdown("### Global Terror Attacks per Year")
        # Attacks per year
        yearly = df.groupby('iyear').size()
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(yearly.index, yearly.values, color='crimson', linewidth=2)
        ax.set_title("Global Terror Attacks per Year")
        ax.set_xlabel("Year")
        ax.set_ylabel("Number of Attacks")
        st.pyplot(fig)
        st.caption("Terror incidents peaked in the 2010s due to global conflicts.")

        st.markdown("### Top 10 Most Affected Countries")
        # Top countries
        top_countries = df['country_txt'].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(y=top_countries.index, x=top_countries.values, palette='Reds_r', ax=ax)
        ax.set_title("Top 10 Most Affected Countries")
        ax.set_xlabel("Number of Attacks")
        ax.set_ylabel("Country")
        st.pyplot(fig)

        st.markdown("### Most Common Attack Types")
        # Attack types
        attack_counts = df['attacktype1_txt'].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(y=attack_counts.index, x=attack_counts.values, palette='Blues_r', ax=ax)
        ax.set_title("Most Common Attack Types")
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Attack Type")
        st.pyplot(fig)

        st.markdown("### Top 50 Terrorist Organizations by Number of Attacks")
        # Terrorist Groups
        group_counts = (df[df['gname'] != 'Unknown']['gname'].value_counts().head(50))  # Exclude 'Unknown' to focus on identified groups
        # fig, ax = plt.subplots(figsize=(12, 6))
        fig, ax = plt.subplots(figsize=(14, 10))
        sns.barplot(y=group_counts.index, x=group_counts.values, palette='viridis', ax=ax)
        ax.set_title("Most Active Terrorist Organizations")
        ax.set_xlabel("Number of Attacks")
        ax.set_ylabel("Terrorist Organization")
        ax.margins(x=0.01, y=0.01) 
        st.pyplot(fig)

        st.markdown("### Average Casualties by Attack Type")
        # Average casualties
        casualties = df.groupby('attacktype1_txt')['total_casualties'].mean().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x=casualties.values, y=casualties.index, palette='coolwarm', ax=ax)
        ax.set_title("Average Casualties per Attack Type")
        ax.set_xlabel("Average Casualties")
        ax.set_ylabel("Attack Type")
        st.pyplot(fig)

        st.markdown("### Regional Terrorism Trends Over Time")
        # Regional trends
        trends = df.groupby(['iyear', 'region_txt']).size().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(14, 7))
        trends.plot(ax=ax, linewidth=1.5)
        ax.set_title("Regional Terrorism Trends (1970–2020)")
        ax.set_xlabel("Year")
        ax.set_ylabel("Number of Attacks")
        st.pyplot(fig)

        st.markdown("### Casualty Variable Correlation")
        # Correlation heatmap
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(df[['nkill', 'nwound', 'total_casualties']].corr(),
                    annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        ax.set_title("Correlation Between Casualty Variables")
        st.pyplot(fig)

        #================================================================================#

        # --- Weapon Type Distribution (legend-based, no overlapping labels) ---
        st.markdown("### Weapon Type Distribution")
        counts = df['weaptype1_txt'].value_counts().head(7)

        fig, ax = plt.subplots(figsize=(10, 7))  # wider to fit legend nicely
        colors = sns.color_palette('Paired', n_colors=len(counts))

        # Show only percentages on the pie; put full labels in the legend
        wedges, _texts, autotexts = ax.pie(
            counts.values,
            labels=None,                    # no labels on slices
            autopct='%1.1f%%',
            startangle=140,
            colors=colors,
            pctdistance=0.8,                # pull % text inward a bit
            textprops={'fontsize': 10}
        )

        # Bolden percentage text for readability
        for t in autotexts:
            t.set_fontweight('bold')
            t.set_color('black')

        ax.set_title("Weapon Type Distribution in Attacks", fontsize=14)

        # Build legend with full labels + counts + share
        total = counts.sum()
        legend_labels = [f"{name} — {val} ({val/total:.1%})" for name, val in counts.items()]
        ax.legend(
            wedges,
            legend_labels,
            title="Weapon Types",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.,
            frameon=True
        )

        plt.tight_layout()
        st.pyplot(fig)

    # ================================================================
    # MODEL TAB
    # ================================================================

    # Model Tab
    with tab_model:
        st.header("🧠 XGBoost Model (Past 5-Year Enhanced Features)")

        @st.cache_resource(show_spinner=False)
        def load_or_train():
            return train_xgboost_cpu(df)

        with st.spinner("Training model..."):
            model, r2, mae, features = load_or_train()

        # r2 and mae are ALWAYS returned by load_or_train()
        # If model is loaded -> r2 and mae contain saved values
        # If model is trained -> r2 and mae contain fresh values

        if r2 is None:
            st.success("Loaded saved model ✔ (skipped training)")
        else:
            st.success("Model trained ✔")

        # Display metrics properly
        st.metric("Model Accuracy (R²)", f"{(r2*100):.2f}%" if r2 is not None else "N/A")
        st.metric("Mean Absolute Error", f"{mae:.2f}" if mae is not None else "N/A")


        st.caption("Trained using region, attack type, and target data to estimate total casualties.")

        # ============================================================
        # PREP FOR PLOTS (X_test, y_test come from training function)
        # We must reconstruct them the exact same way
        # ============================================================
        # Rebuild the processed DF identically as in train_xgboost_cpu
        
        D = df.copy()
        D["nkill"] = D["nkill"].fillna(0)
        D["nwound"] = D["nwound"].fillna(0)
        D["total_casualties"] = D["nkill"] + D["nwound"]
        D["log_casualties"] = np.log1p(D["total_casualties"])

        # Frequency encoding
        for col in ["region","country","attacktype1","targtype1","weaptype1"]:
            freq = D[col].value_counts()
            D[col + "_freq"] = D[col].map(freq).astype(float)

        # Interaction freq
        D["region_attack"] = D["region"].astype(str) + "_" + D["attacktype1"].astype(str)
        freq_int = D["region_attack"].value_counts()
        D["region_attack_freq"] = D["region_attack"].map(freq_int).astype(float)

        # Historical means
        D["region_mean"]  = np.log1p(D.groupby("region")["total_casualties"].transform("mean"))
        D["attack_mean"]  = np.log1p(D.groupby("attacktype1")["total_casualties"].transform("mean"))
        D["country_mean"] = np.log1p(D.groupby("country")["total_casualties"].transform("mean"))

        # Categorical encodings
        for col in ["region","country","attacktype1","targtype1","weaptype1"]:
            D[col + "_cat"] = D[col].astype("category").cat.codes

        # Time trend
        ymin, ymax = D["iyear"].min(), D["iyear"].max()
        D["year_trend"] = (D["iyear"] - ymin) / max(1, (ymax - ymin))

        # Rolling mean
        D = D.sort_values(["country","iyear"])
        D["country_5yr_mean"] = (
            D.groupby("country")["total_casualties"]
            .transform(lambda x: x.rolling(5, min_periods=1).mean())
        )
        D["country_5yr_mean"] = np.log1p(D["country_5yr_mean"])

        # Train-test split reconstruction
        train_mask = D["iyear"] <= 2018
        X_test = D.loc[~train_mask, features]
        y_test = D.loc[~train_mask, "log_casualties"]

        # Reverse transformed values for interp
        y_test_actual = np.expm1(y_test)
        y_pred_actual = np.expm1(model.predict(X_test))

        # ============================================================
        # 📈 Plot 1 — Actual vs Predicted
        # ============================================================
        st.subheader("📈 Actual vs Predicted Casualties")

        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.scatter(y_test_actual, y_pred_actual, alpha=0.3, edgecolor='k')
        ax1.plot([0, max(y_test_actual)], [0, max(y_test_actual)], color='red', linewidth=2)

        ax1.set_xlabel("Actual Casualties")
        ax1.set_ylabel("Predicted Casualties")
        ax1.set_title("Actual vs Predicted Casualties (XGBoost Regression)")
        ax1.grid(True)

        st.pyplot(fig1)
        st.caption("Points close to the red diagonal indicate strong predictive performance.")

        # ============================================================
        # 📉 Plot 2 — Residual Plot
        # ============================================================
        st.subheader("📉 Residual Plot")

        residuals = y_test_actual - y_pred_actual

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.scatter(y_pred_actual, residuals, alpha=0.3, edgecolor='k')
        ax2.axhline(0, color='red', linestyle='--', linewidth=2)

        ax2.set_xlabel("Predicted Casualties")
        ax2.set_ylabel("Residuals (Actual - Predicted)")
        ax2.set_title("Residual Plot (Model Error Visualization)")
        ax2.grid(True)

        st.pyplot(fig2)
        st.caption("Residuals centered around zero show the model has no strong systematic error.")

    # ================================================================
    # MODEL COMPARISON TAB
    # ================================================================

    with tab_compare:
        st.header("📈 Comparative Analysis of Regression Models")
        comparison = load_model_comparison()

        if comparison.empty:
            st.warning("Model comparison file not found.")

        else:
            fig, ax1 = plt.subplots(figsize=(14,6))

            # R² bars
            bars = ax1.bar(comparison["Model"], comparison["R2"], label="R² Score ↑")

            ax1.set_ylim(0,0.6)
            ax1.set_ylabel("R² Score")

            for bar in bars:
                value = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2 , value + 0.015, f"{value:.4f}", ha="center", fontweight="bold")

            # MAE line
            ax2 = ax1.twinx()
            ax2.plot(comparison["Model"], comparison["MAE"], marker="o", linewidth=3, label="MAE ↓")
            ax2.set_ylim(0,6)
            ax2.set_ylabel("MAE (Casualties)")

            for i,value in enumerate(comparison["MAE"]):
                ax2.text(i, value + 0.15, f"{value:.2f}", ha="center", fontweight="bold")

            plt.xticks(rotation=25, ha="right")

            h1,l1 = ax1.get_legend_handles_labels()
            h2,l2 = ax2.get_legend_handles_labels()
            ax1.legend(h1+h2, l1+l2, loc="upper center", bbox_to_anchor=(0.5,-0.30), ncol=2)
            plt.tight_layout()

            st.pyplot(fig)
            st.caption(
                """
                Higher R² indicates stronger predictive capability.
                Lower MAE indicates reduced casualty prediction error.
                """
            )

    # ================================================================
    # MAP TAB
    # ================================================================

    # MAP TAB
    with tab_map:
        st.header(f"🗺️ Severity Risk Map — {future_year}")

        fmap, pred_df = predict_and_map(df, model, features, future_year, sample_size, country_filter)
        st.components.v1.html(fmap._repr_html_(), height=650)

        out_path = f"gtd_predicted_map_{future_year}.html"
        fmap.save(out_path)
        with open(out_path, "rb") as f:
            st.download_button("⬇️ Download Map HTML", f, file_name=os.path.basename(out_path), mime="text/html")

        st.dataframe(pred_df[['country_txt', 'attacktype1_txt', 'latitude', 'longitude', 'predicted_casualties']].head(15))

    st.caption("Built with Streamlit, Folium, Matplotlib & XGBoost. Data: GTD.")

# ================================================================
# 5️⃣ Fallback for Jupyter / CLI Mode
# ================================================================

else:
    print("📊 Generating EDA plots...")
    display(df.head())

    print("🧠 Training model (CLI fallback)...")
    model, r2, mae, features = train_xgboost_cpu(df)
    if r2 is not None:
        print(f"✅ Model trained. R²={(r2*100):.2f}%, MAE={mae:.2f}")
    else:
        print("✅ Model loaded from disk.")

    DEFAULT_FUTURE_YEAR = 2026
    fmap, pred_df = predict_and_map(df, model, features, DEFAULT_FUTURE_YEAR, sample_size=300)
    path = f"./gtd_predicted_map_{DEFAULT_FUTURE_YEAR}.html"
    fmap.save(path)
    print(f"🌍 Map saved to {path}")
