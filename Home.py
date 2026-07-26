# import streamlit as st

# from src.utils.cache import get_data
# from src.utils.plot_style import configure_plot_style

# # ================================================================
# # Page Configuration
# # ================================================================

# st.set_page_config(
#     page_title="Global Terrorism Dashboard",
#     layout="wide",
# )

# configure_plot_style()

# # ================================================================
# # Load Dataset
# # ================================================================

# df = get_data()

# # ================================================================
# # Header
# # ================================================================

# st.title("🌍 Global Terrorism Dashboard")

# st.markdown(
# """
# ## Machine Learning–Based Terrorism Forecasting Using XGBoost

# An interactive analytics platform built using the **Global Terrorism Database (GTD)** for
# historical analysis, machine learning prediction, model explainability, and future terrorism severity forecasting.
# """
# )

# st.divider()

# # ================================================================
# # Dataset Summary
# # ================================================================

# st.subheader("📊 Dataset Overview")

# col1, col2, col3, col4 = st.columns(4)

# with col1:
#     st.metric("Total Incidents", f"{len(df):,}")

# with col2:
#     st.metric("Countries", df["country_txt"].nunique())

# with col3:
#     st.metric(
#         "Time Span",
#         f"{df['iyear'].min()}–{df['iyear'].max()}"
#     )

# with col4:
#     st.metric(
#         "Attack Types",
#         df["attacktype1_txt"].nunique()
#     )

# st.divider()

# # ================================================================
# # Dashboard Modules
# # ================================================================

# st.subheader("🧭 Dashboard Modules")

# col1, col2 = st.columns(2)

# with col1:

#     with st.container(border=True):
#         st.markdown("### 📊 Data Insights")
#         st.write(
#             "Explore historical terrorism trends, regional activity, "
#             "attack types, casualties, weapon distributions and "
#             "terrorist organizations."
#         )

#     with st.container(border=True):
#         st.markdown("### 🔍 SHAP Explainability")
#         st.write(
#             "Interpret model predictions using SHAP Summary, Feature "
#             "Importance and Waterfall plots."
#         )

#     with st.container(border=True):
#         st.markdown("### 🗺️ Future Prediction")
#         st.write(
#             "Generate future terrorism severity predictions with an "
#             "interactive Folium world map and downloadable reports."
#         )

# with col2:

#     with st.container(border=True):
#         st.markdown("### 🧠 Model Performance")
#         st.write(
#             "Evaluate the trained XGBoost regression model using "
#             "performance metrics, Actual vs Predicted plots and "
#             "Residual analysis."
#         )

#     with st.container(border=True):
#         st.markdown("### 📈 Model Comparison")
#         st.write(
#             "Compare Decision Tree, Random Forest, Gradient Boosting "
#             "and XGBoost models using R² Score and MAE."
#         )

#     with st.container(border=True):
#         st.markdown("### ⚡ Optimized Dashboard")
#         st.write(
#             "Caching and modular architecture provide fast loading, "
#             "reusable components and an organized codebase."
#         )

# st.divider()

# # ================================================================
# # Technologies
# # ================================================================

# st.subheader("🛠️ Technology Stack")

# st.markdown(
# """
# - **Machine Learning:** XGBoost, Scikit-learn
# - **Explainable AI:** SHAP
# - **Data Processing:** Pandas, NumPy
# - **Visualization:** Matplotlib, Folium
# - **Web Framework:** Streamlit
# - **Dataset:** Global Terrorism Database (1970–2020)
# """
# )

# st.divider()

# # ================================================================
# # Navigation
# # ================================================================

# st.info(
# """
# 👈 **Use the navigation menu on the left to explore each module of the dashboard.**

# The pages are organized in the recommended workflow:

# 1. 📊 Data Insights
# 2. 🧠 Model Performance
# 3. 🔍 SHAP Explainability
# 4. 📈 Model Comparison
# 5. 🗺️ Future Prediction
# """
# )

# st.caption(
#     "Built with Streamlit, XGBoost, SHAP, Matplotlib & Folium | Dataset: Global Terrorism Database (GTD)"
# )


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

st.title("🌍 Global Terrorism Prediction System")

st.markdown("""
## AI-Powered Terrorism Severity Forecasting

Predict future terrorism severity using an optimized **XGBoost Regression Model**
trained on the **Global Terrorism Database (GTD)**.

The dashboard combines **historical analytics**, **machine learning prediction**,
**model explainability (SHAP)**, and **future forecasting** in one interactive platform.
""")

# ================================================================
# HERO SECTION
# ================================================================

st.success("""
# 🚀 Main Feature

### Future Terrorism Severity Prediction

This dashboard's primary objective is to **predict future terrorism severity**
using historical terrorism data and an optimized XGBoost model.

Click the button below to launch the prediction engine.
""")

if st.button(
    "🌍 START FUTURE PREDICTION",
    type="primary",
    use_container_width=True,
):
    st.switch_page("pages/5_Future_Prediction.py")

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
# Navigation Cards
# ================================================================

st.subheader("🧭 Explore Dashboard Modules")

col1, col2 = st.columns(2)

# ================================================================
# LEFT COLUMN
# ================================================================

with col1:

    with st.container(border=True):
        st.markdown("## 📊 Historical Data Insights")

        st.write(
            "Explore terrorism trends, countries, attack types, "
            "casualties, weapon types and terrorist organizations."
        )

        if st.button(
            "Open Data Insights",
            use_container_width=True,
            key="data_btn",
        ):
            st.switch_page("pages/1_Data_Insights.py")

    st.write("")

    with st.container(border=True):
        st.markdown("## 🔍 SHAP Explainability")

        st.write(
            "Interpret model predictions using SHAP Summary, "
            "Feature Importance and Waterfall plots."
        )

        if st.button(
            "Open SHAP Analysis",
            use_container_width=True,
            key="shap_btn",
        ):
            st.switch_page("pages/3_SHAP_Explainability.py")

    st.write("")

    with st.container(border=True):
        st.markdown("## 🌍 Future Prediction ⭐")

        st.write(
            "Generate future terrorism severity predictions using "
            "the trained XGBoost model and visualize them on the world map."
        )

        if st.button(
            "🚀 Launch Prediction Engine",
            type="primary",
            use_container_width=True,
            key="prediction_btn",
        ):
            st.switch_page("pages/5_Future_Prediction.py")

# ================================================================
# RIGHT COLUMN
# ================================================================

with col2:

    with st.container(border=True):
        st.markdown("## 🧠 Model Performance")

        st.write(
            "Evaluate the trained XGBoost regression model using "
            "R² Score, MAE, residual analysis and prediction plots."
        )

        if st.button(
            "Open Model Performance",
            use_container_width=True,
            key="performance_btn",
        ):
            st.switch_page("pages/2_Model_Performance.py")

    st.write("")

    with st.container(border=True):
        st.markdown("## 📈 Model Comparison")

        st.write(
            "Compare Decision Tree, Random Forest, Gradient Boosting "
            "and XGBoost using R² Score and MAE."
        )

        if st.button(
            "Open Model Comparison",
            use_container_width=True,
            key="comparison_btn",
        ):
            st.switch_page("pages/4_Model_Comparison.py")

    st.write("")

    with st.container(border=True):
        st.markdown("## ⚡ Optimized Dashboard")

        st.write(
            "Built using modular architecture, caching and reusable "
            "components for fast and efficient analysis."
        )

        st.success(
            "✔ Optimized for interactive exploration and rapid prediction."
        )

st.divider()

# ================================================================
# Workflow
# ================================================================

st.subheader("⚙️ Project Workflow")

st.markdown("""
📂 GTD Dataset
      ↓
⚙️ Feature Engineering
      ↓
🧠 XGBoost Model
      ↓
🌍 Future Prediction
      ↓
🔍 SHAP Explainability
""")

st.divider()

# ================================================================
# Technology Stack
# ================================================================

st.subheader("🛠️ Technology Stack")

st.markdown("""
- **Machine Learning:** XGBoost, Scikit-learn
- **Explainable AI:** SHAP
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Folium
- **Web Framework:** Streamlit
- **Dataset:** Global Terrorism Database (1970–2020)
""")

st.divider()

# ================================================================
# Footer
# ================================================================

st.info("""
👈 **Use the sidebar or the navigation cards above to explore each module.**

⭐ **Recommended starting point:** Launch the **Future Prediction Engine** to generate AI-powered terrorism severity forecasts.
""")

st.caption(
    "Built with Streamlit • XGBoost • SHAP • Matplotlib • Folium • Global Terrorism Database (GTD)"
)