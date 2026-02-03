# ================================================================
# 🌍 GTD — High-Resolution Chart Export Script + Model Evaluation
# Exports all analytics charts AND model evaluation charts (400 DPI PNGs)
# ================================================================

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error

plt.style.use("seaborn-v0_8-darkgrid")

# ================================================================
# CONFIG
# ================================================================
DATA_PATH = r'D:/VS Code Programs/Python/gtd/data/gtd.csv'
EXPORT_FOLDER = r'D:/VS Code Programs/Python/gtd/exports'
os.makedirs(EXPORT_FOLDER, exist_ok=True)

def save_chart(fig, name):
    path = os.path.join(EXPORT_FOLDER, f"{name}.png")
    fig.savefig(path, dpi=400, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Saved: {path}")

# ================================================================
# LOAD GTD
# ================================================================
cols = [
    'iyear', 'imonth', 'country_txt', 'region_txt',
    'region', 'country', 'latitude', 'longitude',
    'attacktype1', 'attacktype1_txt', 'targtype1', 'targtype1_txt',
    'weaptype1', 'weaptype1_txt', 'gname','success', 'nkill', 'nwound'
]
df = pd.read_csv(DATA_PATH, usecols=cols, encoding='ISO-8859-1', low_memory=False)

df['nkill'].fillna(0, inplace=True)
df['nwound'].fillna(0, inplace=True)
df['total_casualties'] = df['nkill'] + df['nwound']
df.dropna(subset=['latitude', 'longitude'], inplace=True)

print(f"📊 Loaded dataset with {len(df):,} records.")

# ================================================================
# 📌 1) EXPORT ALL EDA CHARTS (same as your version)
# ================================================================

# 1️⃣ Attacks per year
yearly = df.groupby('iyear').size()
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(yearly.index, yearly.values, color='crimson', linewidth=2)
ax.set_title("Global Terror Attacks per Year")
ax.set_xlabel("Year"); ax.set_ylabel("Number of Attacks")
save_chart(fig, "attacks_per_year")

# 2️⃣ Top countries
top_countries = df['country_txt'].value_counts().head(10)
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(y=top_countries.index, x=top_countries.values, palette='Reds_r', ax=ax, legend=False)
ax.set_title("Top 10 Most Affected Countries")
ax.set_xlabel("Number of Attacks"); ax.set_ylabel("Country")
save_chart(fig, "top_countries")

# 3️⃣ Attack types
attack_counts = df['attacktype1_txt'].value_counts().head(10)
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(y=attack_counts.index, x=attack_counts.values, palette='Blues_r', ax=ax, legend=False)
ax.set_title("Most Common Attack Types")
ax.set_xlabel("Frequency"); ax.set_ylabel("Attack Type")
save_chart(fig, "attack_types")

# 4️⃣ Terrorist Groups
group_counts = (df[df['gname'] != 'Unknown']['gname'].value_counts().head(50))
fig, ax = plt.subplots(figsize=(14, 10))
sns.barplot(y=group_counts.index, x=group_counts.values, palette='viridis', ax=ax)
ax.set_title("Most Active Terrorist Organizations")
ax.set_xlabel("Number of Attacks")
ax.set_ylabel("Terrorist Organization")
ax.margins(x=0.01, y=0.01) 
save_chart(fig, "most_active_terrorist_organizations")

# 5️⃣ Avg casualties
casualties = df.groupby('attacktype1_txt')['total_casualties'].mean().sort_values(ascending=False).head(10)
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x=casualties.values, y=casualties.index, palette='coolwarm', ax=ax, legend=False)
ax.set_title("Average Casualties per Attack Type")
ax.set_xlabel("Average Casualties"); ax.set_ylabel("Attack Type")
save_chart(fig, "avg_casualties_per_attack_type")

# 6️⃣ Regional Trends
region_trends = df.groupby(['iyear', 'region_txt']).size().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(14, 7))
region_trends.plot(ax=ax, linewidth=1.5)
ax.set_title("Regional Terrorism Trends (1970–2020)")
ax.set_xlabel("Year"); ax.set_ylabel("Number of Attacks")
save_chart(fig, "regional_trends")

# 7️⃣ Correlation Heatmap
fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(df[['nkill', 'nwound', 'total_casualties']].corr(),
            annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
ax.set_title("Correlation Between Casualty Variables")
save_chart(fig, "casualty_correlation")

# 8️⃣ Weapon Type Distribution
weapon_counts = df['weaptype1_txt'].value_counts().head(7)
fig, ax = plt.subplots(figsize=(12, 9))
colors = sns.color_palette('Paired', n_colors=len(weapon_counts))

wedges, _texts, autotexts = ax.pie(
    weapon_counts.values,
    autopct='%1.1f%%',
    startangle=140,
    labels=None,
    colors=colors,
    pctdistance=0.8
)

for t in autotexts:
    t.set_fontweight('bold')

ax.set_title("Weapon Type Distribution", fontsize=16)

legend_labels = [
    f"{label} — {value} ({value/sum(weapon_counts):.1%})"
    for label, value in zip(weapon_counts.index, weapon_counts.values)
]

ax.legend(
    wedges,
    legend_labels,
    title="Weapon Types",
    loc="center left",
    bbox_to_anchor=(1.15, 0.5),
    fontsize=10,
    title_fontsize=12
)

plt.tight_layout()
save_chart(fig, "weapon_type_distribution")

# ================================================================
# 📌 2) MODEL TRAINING — (Same exact pipeline as your dashboard)
# ================================================================

D = df.copy()

# Clean
D["nkill"].fillna(0, inplace=True)
D["nwound"].fillna(0, inplace=True)
D["total_casualties"] = D["nkill"] + D["nwound"]
D["log_casualties"] = np.log1p(D["total_casualties"])

# Feature engineering
for col in ["region","country","attacktype1","targtype1","weaptype1"]:
    freq = D[col].value_counts()
    D[col + "_freq"] = D[col].map(freq).astype(float)

D["region_attack"] = D["region"].astype(str) + "_" + D["attacktype1"].astype(str)
freq_int = D["region_attack"].value_counts()
D["region_attack_freq"] = D["region_attack"].map(freq_int).astype(float)

D["region_mean"]  = np.log1p(D.groupby("region")["total_casualties"].transform("mean"))
D["attack_mean"]  = np.log1p(D.groupby("attacktype1")["total_casualties"].transform("mean"))
D["country_mean"] = np.log1p(D.groupby("country")["total_casualties"].transform("mean"))

for col in ["region","country","attacktype1","targtype1","weaptype1"]:
    D[col + "_cat"] = D[col].astype("category").cat.codes

D = D.sort_values(["country", "iyear"])
D["country_5yr_mean"] = (
    D.groupby("country")["total_casualties"]
    .transform(lambda x: x.rolling(5, min_periods=1).mean())
)
D["country_5yr_mean"] = np.log1p(D["country_5yr_mean"])

D["year_trend"] = (D["iyear"] - D["iyear"].min()) / (D["iyear"].max() - D["iyear"].min())

features = [
    "iyear","imonth","region_freq","country_freq",
    "attacktype1_freq","targtype1_freq","weaptype1_freq",
    "success","region_attack_freq",
    "region_cat","country_cat","attacktype1_cat",
    "targtype1_cat","weaptype1_cat",
    "region_mean","attack_mean","country_mean",
    "year_trend","country_5yr_mean"
]

train_mask = D["iyear"] <= 2018
X_train = D.loc[train_mask, features]
y_train = D.loc[train_mask, "log_casualties"]
X_test  = D.loc[~train_mask, features]
y_test  = D.loc[~train_mask, "log_casualties"]

model = XGBRegressor(
    n_estimators=900,
    learning_rate=0.04,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    tree_method="hist"
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_test_actual = np.expm1(y_test)
y_pred_actual = np.expm1(y_pred)

# ================================================================
# 📈 EXPORT — Actual vs Predicted
# ================================================================
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(y_test_actual, y_pred_actual, alpha=0.3, edgecolor='k')
ax.plot([0, max(y_test_actual)], [0, max(y_test_actual)], color='red', linewidth=2)
ax.set_title("Actual vs Predicted Casualties (XGBoost)")
ax.set_xlabel("Actual Casualties")
ax.set_ylabel("Predicted Casualties")
ax.grid(True)

save_chart(fig, "xgboost_actual_vs_predicted")

# ================================================================
# 📉 EXPORT — Residual Plot
# ================================================================
residuals = y_test_actual - y_pred_actual

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(y_pred_actual, residuals, alpha=0.3, edgecolor='k')
ax.axhline(0, color='red', linestyle='--')
ax.set_title("Residual Plot (Actual - Predicted)")
ax.set_xlabel("Predicted Casualties")
ax.set_ylabel("Residuals")
ax.grid(True)

save_chart(fig, "xgboost_residual_plot")

print("\n🎉 ALL charts exported successfully!")
print(f"📁 Saved in: {EXPORT_FOLDER}")
