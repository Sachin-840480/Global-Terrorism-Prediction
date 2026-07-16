# ================================================================
# 🌍 Global Terrorism Dashboard
# Publication Quality Figure Export Script
# (600 DPI)
# ================================================================

import os
import json
import traceback

import shap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (r2_score,mean_absolute_error,)
from xgboost import XGBRegressor

plt.style.use("seaborn-v0_8-darkgrid")

# ================================================================
# CONFIG
# ================================================================

DATA_PATH = r"./data/gtd.csv"

MODEL_DIR = "./model"

MODEL_PATH = os.path.join(MODEL_DIR, "xgb_gtd_model.json")
FEATURES_PATH = os.path.join(MODEL_DIR, "features.txt")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
COMPARISON_PATH = os.path.join(MODEL_DIR, "model_comparison.csv")

EXPORT_ROOT = "./exports"

EDA_DIR = os.path.join(EXPORT_ROOT, "eda")
MODEL_EXPORT_DIR = os.path.join(EXPORT_ROOT, "model")
SHAP_DIR = os.path.join(EXPORT_ROOT, "shap")
COMPARE_DIR = os.path.join(EXPORT_ROOT, "comparison")

for folder in [EXPORT_ROOT,EDA_DIR,MODEL_EXPORT_DIR,SHAP_DIR,COMPARE_DIR]:
    os.makedirs(folder, exist_ok=True)

EXPORT_DPI = 600

print("=" * 70)
print(" Global Terrorism Figure Export")
print("=" * 70)

# ================================================================
# SAVE FIGURE
# ================================================================

def save_chart(fig, folder, filename):
    path = os.path.join(folder, filename)
    fig.savefig(path,dpi=EXPORT_DPI,bbox_inches="tight",facecolor="white")
    plt.close(fig)
    print(f"✅ {filename}")

# ================================================================
# SAVE DATAFRAME
# ================================================================

def save_csv(df, folder, filename):
    path = os.path.join(folder, filename)
    df.to_csv(path,index=False)
    print(f"✅ {filename}")

# ================================================================
# LOAD GTD
# ================================================================

cols = ["iyear","imonth","country_txt","region_txt","region","country","latitude","longitude","attacktype1","attacktype1_txt","targtype1","targtype1_txt","weaptype1","weaptype1_txt","gname","success","nkill","nwound"]

print("Loading dataset...")

df = pd.read_csv(DATA_PATH,usecols=cols,encoding="ISO-8859-1",low_memory=False)
df["nkill"] = df["nkill"].fillna(0)
df["nwound"] = df["nwound"].fillna(0)
df["total_casualties"] = df["nkill"] + df["nwound"]
df = df.dropna(subset=["latitude","longitude"])
print(f"Loaded {len(df):,} events.")

# ================================================================
# LOAD SAVED MODEL
# ================================================================

print("Loading saved XGBoost model...")

model = XGBRegressor()
model.load_model(MODEL_PATH)

with open(FEATURES_PATH, "r") as f:
    features = [line.strip() for line in f.readlines() if line.strip()]

with open(METRICS_PATH, "r") as f:
    metrics = json.load(f)

r2 = metrics["r2"]
mae = metrics["mae"]

print(f"Model R² : {r2:.4f}")
print(f"Model MAE: {mae:.4f}")


# ================================================================
# FEATURE ENGINEERING
# ================================================================

D = df.copy()

D["nkill"] = D["nkill"].fillna(0)
D["nwound"] = D["nwound"].fillna(0)
D["total_casualties"] = D["nkill"] + D["nwound"]
D["log_casualties"] = np.log1p(D["total_casualties"])

for col in ["region","country","attacktype1","targtype1","weaptype1"]:
    freq = D[col].value_counts()
    D[col + "_freq"] = D[col].map(freq).astype(float)

D["region_attack"] = (D["region"].astype(str)+ "_"+ D["attacktype1"].astype(str))
freq = D["region_attack"].value_counts()

D["region_attack_freq"] = D["region_attack"].map(freq).astype(float)
D["region_mean"] = np.log1p(D.groupby("region")["total_casualties"].transform("mean"))
D["attack_mean"] = np.log1p(D.groupby("attacktype1")["total_casualties"].transform("mean"))
D["country_mean"] = np.log1p(D.groupby("country")["total_casualties"].transform("mean"))

for col in ["region","country","attacktype1","targtype1","weaptype1"]:
    D[col + "_cat"] = (D[col].astype("category").cat.codes)

ymin = D["iyear"].min()
ymax = D["iyear"].max()

D["year_trend"] = (D["iyear"] - ymin) / (ymax - ymin)
D = D.sort_values(["country","iyear"])
D["country_5yr_mean"] = (D.groupby("country")["total_casualties"].transform(lambda x: x.rolling(5,min_periods=1).mean()))
D["country_5yr_mean"] = np.log1p(D["country_5yr_mean"])

train_mask = D["iyear"] <= 2018
X_test = D.loc[~train_mask, features]
y_test = D.loc[~train_mask, "log_casualties"]
y_test_actual = np.expm1(y_test)
y_pred_actual = np.expm1(model.predict(X_test))

print("Feature engineering complete.")

# ================================================================
# 📊 EXPORT EDA FIGURES
# ================================================================

print("\nExporting EDA figures...")

# ================================================================
# 1. Global Terror Attacks per Year
# ================================================================

yearly = df.groupby("iyear").size()
fig, ax = plt.subplots(figsize=(14,6))

ax.plot(yearly.index,yearly.values,color="crimson",linewidth=2)
ax.set_title("Global Terror Attacks per Year", fontsize=16)
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Number of Attacks", fontsize=12)
ax.grid(True)

save_chart(fig,EDA_DIR,"01_attacks_per_year.png")

# ================================================================
# 2. Top Countries
# ================================================================

top_countries = (df["country_txt"].value_counts().head(10))
fig, ax = plt.subplots(figsize=(12,6))

sns.barplot(y=top_countries.index,x=top_countries.values,palette="Reds_r",ax=ax)
ax.set_title("Top 10 Most Affected Countries",fontsize=16)
ax.set_xlabel("Number of Attacks",fontsize=12)
ax.set_ylabel("Country",fontsize=12)

save_chart(fig,EDA_DIR,"02_top_countries.png")

# ================================================================
# 3. Attack Types
# ================================================================

attack_counts = (df["attacktype1_txt"].value_counts().head(10))
fig, ax = plt.subplots(figsize=(12,6))

sns.barplot(y=attack_counts.index,x=attack_counts.values,palette="Blues_r",ax=ax)
ax.set_title("Most Common Attack Types",fontsize=16)
ax.set_xlabel("Frequency",fontsize=12)
ax.set_ylabel("Attack Type",fontsize=12)

save_chart(fig,EDA_DIR,"03_attack_types.png")

# ================================================================
# 4. Terrorist Organizations
# ================================================================

groups = (df[df["gname"]!="Unknown"]["gname"].value_counts().head(50))
fig, ax = plt.subplots(figsize=(14,10))

sns.barplot(y=groups.index,x=groups.values,palette="viridis",ax=ax)
ax.set_title("Most Active Terrorist Organizations",fontsize=16)
ax.set_xlabel("Number of Attacks",fontsize=12)
ax.set_ylabel("Terrorist Organization",fontsize=12)
ax.margins(x=0.01)

save_chart(fig,EDA_DIR,"04_terrorist_groups.png")

# ================================================================
# 5. Average Casualties
# ================================================================

casualties = (df.groupby("attacktype1_txt")["total_casualties"].mean().sort_values(ascending=False).head(10))
fig, ax = plt.subplots(figsize=(12,6))

sns.barplot(x=casualties.values,y=casualties.index,palette="coolwarm",ax=ax)
ax.set_title("Average Casualties per Attack Type",fontsize=16)
ax.set_xlabel("Average Casualties",fontsize=12)
ax.set_ylabel("Attack Type",fontsize=12)

save_chart(fig,EDA_DIR,"05_average_casualties.png")

# ================================================================
# 6. Regional Trends
# ================================================================

trends = (df.groupby(["iyear","region_txt"]).size().unstack(fill_value=0))
fig, ax = plt.subplots(figsize=(14,7))

trends.plot(ax=ax,linewidth=1.5)
ax.set_title("Regional Terrorism Trends (1970-2020)",fontsize=16)
ax.set_xlabel("Year",fontsize=12)
ax.set_ylabel("Number of Attacks",fontsize=12)

save_chart(fig,EDA_DIR,"06_regional_trends.png")

# ================================================================
# 7. Correlation Heatmap
# ================================================================

fig, ax = plt.subplots(figsize=(8,5))

sns.heatmap(df[["nkill","nwound","total_casualties"]].corr(),annot=True,cmap="coolwarm",fmt=".2f",ax=ax)
ax.set_title("Correlation Between Casualty Variables",fontsize=16)

save_chart(fig,EDA_DIR,"07_correlation_heatmap.png")

# ================================================================
# 8. Weapon Distribution
# ================================================================

counts = (df["weaptype1_txt"].value_counts().head(7))
fig, ax = plt.subplots(figsize=(10,7))
colors = sns.color_palette("Paired",len(counts))
wedges, _, autotexts = ax.pie(counts.values,labels=None,autopct="%1.1f%%",startangle=140,pctdistance=0.80,colors=colors)

for t in autotexts:
    t.set_fontweight("bold")

legend_labels = [f"{k} — {v} ({v/counts.sum():.1%})" for k,v in counts.items()]

ax.legend(wedges,legend_labels,title="Weapon Types",loc="center left",bbox_to_anchor=(1.02,0.5),frameon=True)
ax.set_title("Weapon Type Distribution",fontsize=16)
plt.tight_layout()

save_chart(fig,EDA_DIR,"08_weapon_distribution.png")

print("EDA export complete.")

# ================================================================
# 📈 EXPORT MODEL EVALUATION FIGURES
# ================================================================

print("\nExporting model evaluation figures...")

# ================================================================
# 1. Actual vs Predicted
# ================================================================

fig, ax = plt.subplots(figsize=(10,6))

ax.scatter(y_test_actual,y_pred_actual,alpha=0.30,edgecolor="black")
m = max(y_test_actual.max(), y_pred_actual.max())
ax.plot([0, m],[0, m],color="red",linewidth=2)
ax.set_title("Actual vs Predicted Casualties (XGBoost Regression)",fontsize=16)
ax.set_xlabel("Actual Casualties",fontsize=12)
ax.set_ylabel("Predicted Casualties",fontsize=12)
ax.grid(True)

save_chart(fig,MODEL_EXPORT_DIR,"01_actual_vs_predicted.png")

# ================================================================
# 2. Residual Plot
# ================================================================

residuals = y_test_actual - y_pred_actual
fig, ax = plt.subplots(figsize=(10,6))

ax.scatter(y_pred_actual,residuals,alpha=0.30,edgecolor="black")
ax.axhline(0,color="red",linestyle="--",linewidth=2)
ax.set_title("Residual Plot (Model Error Visualization)",fontsize=16)
ax.set_xlabel("Predicted Casualties",fontsize=12)
ax.set_ylabel("Residuals (Actual − Predicted)",fontsize=12)
ax.grid(True)

save_chart(fig,MODEL_EXPORT_DIR,"02_residual_plot.png")

# ================================================================
# 📈 EXPORT MODEL COMPARISON
# ================================================================

print("Exporting model comparison...")

comparison = pd.read_csv(COMPARISON_PATH)
comparison["Model"] = comparison["Model"].str.replace(" ", "\n")

fig, ax1 = plt.subplots(figsize=(14,6))
fig.patch.set_facecolor("white")
ax1.set_facecolor("white")

# ==========================================================
# GRID
# ==========================================================

ax1.set_axisbelow(True)
ax1.grid(axis="y",color="#D0D0D0",linewidth=0.8,alpha=0.8)
ax1.grid(axis="x",visible=False)

# ==========================================================
# R² BAR
# ==========================================================

bars = ax1.bar(comparison["Model"],comparison["R2"],width=0.5,color="#4FC3CF",label="R² Score ↑",zorder=3)
ax1.set_ylim(0,0.6)
ax1.set_yticks([0,0.2,0.4,0.6])
ax1.margins(x=0.08)
ax1.set_ylabel("R² Score",fontsize=11)

# ==========================================================
# MAE LINE
# ==========================================================

ax2 = ax1.twinx()
ax2.grid(False)
ax2.plot(comparison["Model"],comparison["MAE"],marker="o",linewidth=3,markersize=6,color="#F5B83D",label="MAE ↓",zorder=5)
ax2.set_ylim(0,6)
ax2.set_ylabel("MAE (Casualties)",fontsize=11)

# ==========================================================
# REMOVE SPINES
# ==========================================================

for ax in [ax1, ax2]:
    for spine in ax.spines.values():
        spine.set_visible(False)

# ==========================================================
# BAR LABELS
# ==========================================================

for bar in bars:
    value = bar.get_height()
    ax1.text(bar.get_x()+bar.get_width()/2,value+0.015,f"{value:.4f}",ha="center",fontsize=10,fontweight="bold")

# ==========================================================
# MAE LABELS
# ==========================================================

for i, value in enumerate(comparison["MAE"]):
    ax2.text(i,value+0.15,f"{value:.2f}",ha="center",fontsize=10,fontweight="bold")

# ==========================================================
# AXES
# ==========================================================

plt.xticks(rotation=0,ha="center",fontweight="bold")
ax1.tick_params(axis="x",labelsize=9)
ax1.tick_params(axis="y",labelsize=10)
ax2.tick_params(axis="y",labelsize=10)

# ==========================================================
# LEGEND
# ==========================================================

h1,l1 = ax1.get_legend_handles_labels()
h2,l2 = ax2.get_legend_handles_labels()

ax1.legend(h1+h2,l1+l2,loc="upper center",bbox_to_anchor=(0.5,-0.16),ncol=2,frameon=False,fontsize=11)
plt.tight_layout()
save_chart(fig,COMPARE_DIR,"01_model_comparison.png")

print("Model evaluation export complete.")


# ================================================================
# 🔍 EXPORT SHAP FIGURES
# ================================================================

print("\nExporting SHAP figures...")

# ------------------------------------------------
# SHAP SAMPLE
# ------------------------------------------------

X_shap = X_test.sample(min(1000, len(X_test)),random_state=42)

print("Computing SHAP values...")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_shap)

print("SHAP values computed.")

# ================================================================
# SHAP SUMMARY
# ================================================================

plt.figure(figsize=(10,7))
shap.summary_plot(shap_values,X_shap,show=False)
save_chart(plt.gcf(),SHAP_DIR,"01_shap_summary.png")

# ================================================================
# SHAP BAR
# ================================================================

plt.figure(figsize=(10,6))
shap.summary_plot(shap_values,X_shap,plot_type="bar",show=False)
save_chart(plt.gcf(),SHAP_DIR,"02_shap_bar.png")

# ================================================================
# SHAP WATERFALL
# ================================================================

sample = 0

explanation = shap.Explanation(values=shap_values[sample],base_values=explainer.expected_value,data=X_shap.iloc[sample],feature_names=X_shap.columns)

fig = plt.figure(figsize=(12,6))
shap.plots.waterfall(explanation,show=False)
save_chart(fig,SHAP_DIR,"03_shap_waterfall.png")

# ================================================================
# SHAP FEATURE IMPORTANCE
# ================================================================

importance = pd.DataFrame({"Feature": X_shap.columns,"Mean |SHAP|": np.abs(shap_values).mean(axis=0)})
importance = (importance.sort_values(by="Mean |SHAP|",ascending=False).reset_index(drop=True))
importance.insert(0,"Rank",range(1,len(importance)+1))

save_csv(importance,SHAP_DIR,"shap_feature_importance.csv")

# ================================================================
# EXPORT LOG
# ================================================================

log_path = os.path.join(EXPORT_ROOT,"export_log.txt")

with open(log_path, "w", encoding="utf-8") as f:

    f.write("="*60 + "\n")
    f.write("GLOBAL TERRORISM DASHBOARD\n")
    f.write("Publication Figure Export\n")
    f.write("="*60 + "\n\n")

    f.write(f"Dataset Records : {len(df):,}\n")
    f.write(f"Features        : {len(features)}\n")
    f.write(f"Model           : XGBoost Regressor\n")
    f.write(f"Trees           : 2000\n")
    f.write(f"Max Depth       : 9\n")
    f.write(f"Learning Rate   : 0.005\n")
    f.write(f"R² Score        : {r2:.4f}\n")
    f.write(f"MAE             : {mae:.4f}\n")
    f.write(f"DPI             : {EXPORT_DPI}\n")
    f.write("\n")

    f.write("Exported folders\n")
    f.write("----------------\n")
    f.write("EDA\n")
    f.write("MODEL\n")
    f.write("COMPARISON\n")
    f.write("SHAP\n")

print("\nExport log created.")

# ================================================================
# FINAL MESSAGE !!!!
# ================================================================

print("\n" + "="*70)
print("🎉 ALL FIGURES EXPORTED SUCCESSFULLY")
print("="*70)

print(f"\nExport Folder : {EXPORT_ROOT}")

print("\nEDA Figures        : 8")
print("Model Figures      : 2")
print("Comparison Figures : 1")
print("SHAP Figures       : 3")
print("CSV Tables         : 1")

print("\nAll figures exported at 600 DPI.")