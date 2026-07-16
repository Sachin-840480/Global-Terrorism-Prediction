# ================================================================
# 🌍 GTD — High-Resolution Chart Export Script
# Standalone script to export all analytics charts as 400 DPI PNGs
# ================================================================

# It only saves charts and does not display them interactively and the xgboost plots are not saved in high DPI.

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-darkgrid")

DATA_PATH = r'F:/VS Code Programs/Python/gtd demo/data/gtd.csv'
EXPORT_FOLDER = "./gtd demo/exports"
os.makedirs(EXPORT_FOLDER, exist_ok=True)

# ---------------------------------------------------------------
# Helper to save charts in 400 DPI
# ---------------------------------------------------------------
def save_chart(fig, name):
    path = os.path.join(EXPORT_FOLDER, f"{name}.png")
    fig.savefig(path, dpi=400, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Saved: {path}")

# ---------------------------------------------------------------
# Load GTD dataset
# ---------------------------------------------------------------
cols = [
    'iyear', 'imonth', 'country_txt', 'region_txt',
    'region', 'country', 'latitude', 'longitude',
    'attacktype1', 'attacktype1_txt', 'targtype1', 'targtype1_txt',
    'weaptype1', 'weaptype1_txt', 'success', 'nkill', 'nwound'
]
df = pd.read_csv(DATA_PATH, usecols=cols, encoding='ISO-8859-1', low_memory=False)
df['nkill'].fillna(0, inplace=True)
df['nwound'].fillna(0, inplace=True)
df['total_casualties'] = df['nkill'] + df['nwound']
df.dropna(subset=['latitude', 'longitude'], inplace=True)
print(f"📊 Loaded dataset with {len(df):,} records.")

# ---------------------------------------------------------------
# 1️⃣ Global Attacks per Year
# ---------------------------------------------------------------
yearly = df.groupby('iyear').size()
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(yearly.index, yearly.values, color='crimson', linewidth=2)
ax.set_title("Global Terror Attacks per Year")
ax.set_xlabel("Year"); ax.set_ylabel("Number of Attacks")
save_chart(fig, "attacks_per_year")

# ---------------------------------------------------------------
# 2️⃣ Top 10 Countries by Attacks
# ---------------------------------------------------------------
top_countries = df['country_txt'].value_counts().head(10)
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(y=top_countries.index, x=top_countries.values, palette='Reds_r', ax=ax)
ax.set_title("Top 10 Most Affected Countries")
ax.set_xlabel("Number of Attacks"); ax.set_ylabel("Country")
save_chart(fig, "top_countries")

# ---------------------------------------------------------------
# 3️⃣ Common Attack Types
# ---------------------------------------------------------------
attack_counts = df['attacktype1_txt'].value_counts().head(10)
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(y=attack_counts.index, x=attack_counts.values, palette='Blues_r', ax=ax)
ax.set_title("Most Common Attack Types")
ax.set_xlabel("Frequency"); ax.set_ylabel("Attack Type")
save_chart(fig, "attack_types")

# ---------------------------------------------------------------
# 4️⃣ Average Casualties by Attack Type
# ---------------------------------------------------------------
casualties = df.groupby('attacktype1_txt')['total_casualties'].mean().sort_values(ascending=False).head(10)
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x=casualties.values, y=casualties.index, palette='coolwarm', ax=ax)
ax.set_title("Average Casualties per Attack Type")
ax.set_xlabel("Average Casualties"); ax.set_ylabel("Attack Type")
save_chart(fig, "avg_casualties_per_attack_type")

# ---------------------------------------------------------------
# 5️⃣ Regional Trends
# ---------------------------------------------------------------
region_trends = df.groupby(['iyear', 'region_txt']).size().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(14, 7))
region_trends.plot(ax=ax, linewidth=1.5)
ax.set_title("Regional Terrorism Trends (1970–2020)")
ax.set_xlabel("Year"); ax.set_ylabel("Number of Attacks")
save_chart(fig, "regional_trends")

# ---------------------------------------------------------------
# 6️⃣ Correlation Between Casualty Variables
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(df[['nkill', 'nwound', 'total_casualties']].corr(),
            annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
ax.set_title("Correlation Between Casualty Variables")
save_chart(fig, "casualty_correlation")

# ---------------------------------------------------------------
# 7️⃣ Weapon Type Distribution (pie + legend)
# ---------------------------------------------------------------
weapon_counts = df['weaptype1_txt'].value_counts().head(7)
fig, ax = plt.subplots(figsize=(10, 8))
colors = sns.color_palette('Paired', n_colors=len(weapon_counts))
wedges, _texts, autotexts = ax.pie(
    weapon_counts.values,
    autopct='%1.1f%%',
    startangle=140,
    colors=colors,
    textprops={'fontsize': 9}
)
ax.set_title("Weapon Type Distribution in Attacks")
for t in autotexts:
    t.set_fontweight('bold')
ax.legend(
    wedges,
    [f"{label} — {val}" for label, val in zip(weapon_counts.index, weapon_counts.values)],
    title="Weapon Types",
    loc="center left",
    bbox_to_anchor=(1.05, 0.5)
)
plt.tight_layout()
save_chart(fig, "weapon_type_distribution")

print("\n✅ All charts exported successfully in high-resolution to './exports/'")
