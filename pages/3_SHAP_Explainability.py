import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.data.preprocessing import engineer_features

from src.utils.cache import (
    get_data,
    get_model,
    get_shap,
)

from src.visualization.shap import (
    plot_summary,
    plot_bar,
    plot_waterfall,
)

from src.utils.helpers import page_footer

# ================================================================
# Load Data
# ================================================================

df = get_data()

# ================================================================
# Page
# ================================================================

st.title("🔍 Model Explainability using SHAP")

st.markdown("""
SHAP (**SHapley Additive exPlanations**) explains why the XGBoost
model predicts higher or lower casualties.

Unlike traditional feature importance, SHAP shows:

- Which features matter the most.
- Whether they increase or decrease the prediction.
- The contribution of every feature for every prediction.
""")

# ================================================================
# Load Model
# ================================================================

with st.spinner("Loading XGBoost model..."):
    model, r2, mae, features = get_model(df)

# ================================================================
# Feature Engineering
# ================================================================

D = engineer_features(df)

# ================================================================
# Test Data
# ================================================================

train_mask = D["iyear"] <= 2018
X_test = D.loc[~train_mask, features]

# ================================================================
# SHAP
# ================================================================

X_shap = X_test.sample(min(1000, len(X_test)),random_state=42)

with st.spinner("Generating SHAP explanations (first run only)..."):
    explainer, shap_values = get_shap(model,X_shap)

st.success("SHAP values ready.")

# ================================================================
# Summary Plot
# ================================================================

st.markdown("### 📌 SHAP Summary Plot")

st.markdown("""
Shows the influence of every feature across all predictions.

- Red = High feature value
- Blue = Low feature value
- Features are ranked by importance.
""")

fig = plot_summary(shap_values,X_shap)

st.pyplot(fig)
plt.close(fig)

# ================================================================
# Bar Plot
# ================================================================

st.markdown("### 📊 SHAP Feature Importance")

st.markdown("""
Average absolute SHAP values for each feature.

Larger bars indicate greater influence on model predictions.
""")

fig = plot_bar(shap_values,X_shap)

st.pyplot(fig)
plt.close(fig)

# ================================================================
# Feature Importance Table
# ================================================================

st.markdown("### 🏆 Top Features")

importance = pd.DataFrame({"Feature": X_shap.columns,"Mean |SHAP|": np.abs(shap_values).mean(axis=0)})
importance = (importance.sort_values(by="Mean |SHAP|",ascending=False).reset_index(drop=True))
importance.insert(0,"Rank",range(1, len(importance)+1))

st.dataframe(importance.style.format({"Mean |SHAP|": "{:.4f}"}),use_container_width=True,hide_index=True)

# ================================================================
# Waterfall Plot
# ================================================================

st.markdown("### 🌊 SHAP Waterfall Plot")

st.markdown("""
Shows how each feature contributes to one prediction.

- Red features increase the predicted casualties.

- Blue features decrease the predicted casualties.
""")

fig = plot_waterfall(explainer,shap_values,X_shap,sample=0)

st.pyplot(fig)
plt.close(fig)

# ================================================================
# Interpretation
# ================================================================

st.markdown("### 📝 Interpretation")

st.info("""
• Features near the top of the Summary Plot have the greatest impact on predictions.

• The Bar Plot ranks features according to their overall contribution.

• Red points indicate higher feature values, while blue points indicate lower feature values.

• The Waterfall Plot explains why the selected prediction is high or low.

SHAP improves model transparency by providing both global and local explanations for every prediction.
""")

# Footer
page_footer()