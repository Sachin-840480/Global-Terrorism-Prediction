import streamlit as st

from src.data.loader import load_data
from src.models.trainer import train_xgboost_cpu
from src.models.shap_utils import compute_shap
from src.models.metrics import load_model_comparison


# ================================================================
# Dataset
# ================================================================

@st.cache_data(show_spinner=False)
def get_data():
    return load_data()


# ================================================================
# Model
# ================================================================

@st.cache_resource(show_spinner=False)
def get_model(_df):
    return train_xgboost_cpu(_df)


# ================================================================
# SHAP
# ================================================================

@st.cache_resource(show_spinner=False)
def get_shap(_model, _X):
    return compute_shap(_model, _X)


# ================================================================
# Model Comparison
# ================================================================

@st.cache_data(show_spinner=False)
def get_model_comparison():
    return load_model_comparison()

# ================================================================
# Country List
# ================================================================

@st.cache_data(show_spinner=False)
def get_countries(df):
    return sorted(df["country_txt"].dropna().unique().tolist())