import streamlit as st

from src.utils.cache import get_data
from src.utils.plot_style import configure_plot_style

# ================================================================
# Page Configuration
# ================================================================

st.set_page_config(
    page_title="Global Terrorism Dashboard",
    layout="wide",
)

configure_plot_style()

# ================================================================
# Load Dataset
# ================================================================

df = get_data()

# ================================================================
# Header
# ================================================================

st.title("🌍 Global Terrorism Dashboard")

st.markdown(
"""
## Machine Learning–Based Terrorism Forecasting Using XGBoost

An interactive analytics platform built using the **Global Terrorism Database (GTD)** for
historical analysis, machine learning prediction, model explainability, and future terrorism severity forecasting.
"""
)

st.divider()

# ================================================================
# Dataset Summary
# ================================================================

st.subheader("📊 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Incidents", f"{len(df):,}")

with col2:
    st.metric("Countries", df["country_txt"].nunique())

with col3:
    st.metric(
        "Time Span",
        f"{df['iyear'].min()}–{df['iyear'].max()}"
    )

with col4:
    st.metric(
        "Attack Types",
        df["attacktype1_txt"].nunique()
    )

st.divider()

# ================================================================
# Dashboard Modules
# ================================================================

st.subheader("🧭 Dashboard Modules")

col1, col2 = st.columns(2)

with col1:

    with st.container(border=True):
        st.markdown("### 📊 Data Insights")
        st.write(
            "Explore historical terrorism trends, regional activity, "
            "attack types, casualties, weapon distributions and "
            "terrorist organizations."
        )

    with st.container(border=True):
        st.markdown("### 🔍 SHAP Explainability")
        st.write(
            "Interpret model predictions using SHAP Summary, Feature "
            "Importance and Waterfall plots."
        )

    with st.container(border=True):
        st.markdown("### 🗺️ Future Prediction")
        st.write(
            "Generate future terrorism severity predictions with an "
            "interactive Folium world map and downloadable reports."
        )

with col2:

    with st.container(border=True):
        st.markdown("### 🧠 Model Performance")
        st.write(
            "Evaluate the trained XGBoost regression model using "
            "performance metrics, Actual vs Predicted plots and "
            "Residual analysis."
        )

    with st.container(border=True):
        st.markdown("### 📈 Model Comparison")
        st.write(
            "Compare Decision Tree, Random Forest, Gradient Boosting "
            "and XGBoost models using R² Score and MAE."
        )

    with st.container(border=True):
        st.markdown("### ⚡ Optimized Dashboard")
        st.write(
            "Caching and modular architecture provide fast loading, "
            "reusable components and an organized codebase."
        )

st.divider()

# ================================================================
# Technologies
# ================================================================

st.subheader("🛠️ Technology Stack")

st.markdown(
"""
- **Machine Learning:** XGBoost, Scikit-learn
- **Explainable AI:** SHAP
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Folium
- **Web Framework:** Streamlit
- **Dataset:** Global Terrorism Database (1970–2020)
"""
)

st.divider()

# ================================================================
# Navigation
# ================================================================

st.info(
"""
👈 **Use the navigation menu on the left to explore each module of the dashboard.**

The pages are organized in the recommended workflow:

1. 📊 Data Insights
2. 🧠 Model Performance
3. 🔍 SHAP Explainability
4. 📈 Model Comparison
5. 🗺️ Future Prediction
"""
)

st.caption(
    "Built with Streamlit, XGBoost, SHAP, Matplotlib & Folium | Dataset: Global Terrorism Database (GTD)"
)