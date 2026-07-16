import matplotlib.pyplot as plt
import streamlit as st

from src.utils.page_width import configure_page
configure_page()

from src.utils.cache import get_model_comparison

from src.visualization.comparison import (
    plot_model_comparison,
)

from src.utils.helpers import page_footer

# ================================================================
# Page
# ================================================================

st.title("📈 Comparative Analysis of Regression Models")

st.markdown("""
Comparison of multiple regression algorithms evaluated on the
Global Terrorism Database.

The chart compares:

- **R² Score (higher is better)**

- **Mean Absolute Error (lower is better)**
""")

# ================================================================
# Load Comparison Data
# ================================================================

comparison = get_model_comparison()

# ================================================================
# Plot
# ================================================================

fig = plot_model_comparison(comparison)

if fig is None:
    st.warning("Model comparison file not found.")

else:
    st.pyplot(fig)
    plt.close(fig)

# ================================================================
# Interpretation
# ================================================================

st.markdown("### 📝 Interpretation")

st.info("""
• Higher R² indicates better prediction accuracy.

• Lower MAE indicates smaller prediction errors.

• The rolling 5-year XGBoost model provides the best balance
between accuracy and robustness.

• Traditional tree-based models perform reasonably well but
cannot match the predictive capability of the enhanced XGBoost model.
""")

# Footer
page_footer()