import os
import streamlit as st

from src.utils.page_width import configure_page
configure_page()

from src.config import (
    YEAR_MIN,
    YEAR_MAX,
    DEFAULT_FUTURE_YEAR,
    SIZE_MIN,
    SIZE_MAX,
    DEFAULT_SAMPLE_SIZE,
)

from src.utils.cache import (
    get_data,
    get_model,
    get_countries,
)

from src.models.predictor import predict_and_map

from src.utils.helpers import page_footer

# ================================================================
# Page
# ================================================================

st.title("🗺️ Future Terrorism Prediction")

st.markdown("""
Generate future terrorism severity predictions using the trained
XGBoost model.

Adjust the prediction year, sample size and optional country filter
to visualize estimated casualty severity on an interactive world map.
""")

# ================================================================
# Load Cached Data & Model
# ================================================================

df = get_data()

with st.spinner("Loading XGBoost model..."):
    model, r2, mae, features = get_model(df)

# ================================================================
# Sidebar Controls
# ================================================================

st.sidebar.header("Prediction Controls")
future_year = st.sidebar.slider("Future Year",YEAR_MIN,YEAR_MAX,DEFAULT_FUTURE_YEAR,1,)
sample_size = st.sidebar.slider("Sample Size (Map)",SIZE_MIN,SIZE_MAX,DEFAULT_SAMPLE_SIZE,50,)
countries = get_countries(df)
country_filter = st.sidebar.multiselect("Filter by Country",countries,default=[],)
deterministic = st.sidebar.checkbox("Deterministic Prediction",value=True,
    help=(
        "When enabled, the same year always produces the same prediction map. "
        "Disable for a different random sample on each run."
    ),
)

# ================================================================
# Prediction
# ================================================================

with st.spinner("Generating prediction map..."):
    fmap, pred_df = predict_and_map(df,model,features,future_year,sample_size,country_filter,deterministic,)

# ================================================================
# Map
# ================================================================

st.markdown(f"### 🌍 Severity Risk Map — {future_year}")
st.components.v1.html(fmap._repr_html_(),height=650,)

# ================================================================
# Download
# ================================================================

out_path = f"gtd_predicted_map_{future_year}.html"
fmap.save(out_path)
with open(out_path, "rb") as f:
    st.download_button("⬇️ Download Map HTML",f,file_name=os.path.basename(out_path),mime="text/html",)

# ================================================================
# Prediction Table
# ================================================================

st.markdown("### Predicted Events")
st.dataframe(pred_df[["country_txt","attacktype1_txt","latitude","longitude","predicted_casualties",]].head(15),use_container_width=True,)

# Footer
page_footer()

