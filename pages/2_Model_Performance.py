import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.utils.cache import (
    get_data,
    get_model,
)

from src.data.preprocessing import engineer_features

from src.visualization.model import (
    plot_actual_vs_predicted,
    plot_residuals,
)

from src.utils.helpers import page_footer

# ================================================================
# Page
# ================================================================

st.title("🧠 XGBoost Model (Past 5-Year Enhanced Features)")

# ================================================================
# Load Cached Data & Model
# ================================================================

df = get_data()

with st.spinner("Loading XGBoost model..."):
    model, r2, mae, features = get_model(df)

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

st.caption(
    "Trained using region, attack type, target type, weapon type, "
    "historical statistics, and rolling 5-year features to estimate total casualties."
)

# ================================================================
# Feature Engineering and Recreating Test Set
# ================================================================

D = engineer_features(df)

train_mask = D["iyear"] <= 2018

X_test = D.loc[~train_mask, features]
y_test = D.loc[~train_mask, "log_casualties"]

y_test_actual = np.expm1(y_test)
y_pred_actual = np.expm1(model.predict(X_test))


# ================================================================
# Actual vs Predicted
# ================================================================

st.markdown("### 📈 Actual vs Predicted Casualties")

fig = plot_actual_vs_predicted(y_test_actual,y_pred_actual)

st.pyplot(fig)
plt.close(fig)

st.caption("Points close to the red diagonal indicate strong predictive performance.")

# ================================================================
# Residual Plot
# ================================================================

st.markdown("### 📉 Residual Plot")

fig = plot_residuals(y_test_actual,y_pred_actual)

st.pyplot(fig)
plt.close(fig)

st.caption("Residuals centered around zero indicate minimal systematic prediction error.")


# Footer
page_footer()