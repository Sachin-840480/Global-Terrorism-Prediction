import matplotlib.pyplot as plt
import streamlit as st

from src.data.loader import load_data
from src.visualization.eda import (
    plot_attacks_per_year,
    plot_top_countries,
    plot_attack_types,
    plot_top_groups,
    plot_average_casualties,
    plot_regional_trends,
    plot_correlation_heatmap,
    plot_weapon_distribution,
)

from src.utils.helpers import page_footer

# ================================================================
# Load Dataset
# ================================================================

df = load_data()

# ================================================================
# Page Title
# ================================================================

st.header("📊 Historical Insights")

st.markdown("""
Explore historical terrorism trends from the
**Global Terrorism Database (GTD)**.
""")

# ================================================================
# Attacks Per Year
# ================================================================

st.markdown("###📈 Global Terror Attacks per Year")

fig = plot_attacks_per_year(df)
st.pyplot(fig)
plt.close(fig)

st.caption(
    "Terror incidents peaked during the 2010s due to major global conflicts."
)

# ================================================================
# Top Countries
# ================================================================

st.markdown("###🌍 Top 10 Most Affected Countries")

fig = plot_top_countries(df)
st.pyplot(fig)
plt.close(fig)

# ================================================================
# Attack Types
# ================================================================

st.markdown("###💥 Most Common Attack Types")

fig = plot_attack_types(df)
st.pyplot(fig)
plt.close(fig)

# ================================================================
# Terrorist Organizations
# ================================================================

st.markdown("###🏴 Top 50 Terrorist Organizations")

fig = plot_top_groups(df)
st.pyplot(fig)
plt.close(fig)

# ================================================================
# Average Casualties
# ================================================================

st.markdown("###☠️ Average Casualties by Attack Type")

fig = plot_average_casualties(df)
st.pyplot(fig)
plt.close(fig)

# ================================================================
# Regional Trends
# ================================================================

st.markdown("###🌎 Regional Terrorism Trends")

fig = plot_regional_trends(df)
st.pyplot(fig)
plt.close(fig)

# ================================================================
# Correlation Heatmap
# ================================================================

st.markdown("###📊 Casualty Correlation Matrix")

fig = plot_correlation_heatmap(df)
st.pyplot(fig)
plt.close(fig)

# ================================================================
# Weapon Distribution
# ================================================================

st.markdown("###🔫 Weapon Type Distribution")

fig = plot_weapon_distribution(df)
st.pyplot(fig)
plt.close(fig)


# Footer
page_footer()