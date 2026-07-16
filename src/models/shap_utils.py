import shap
import streamlit as st


@st.cache_resource(show_spinner=False)
def compute_shap(
    model,
    X
):

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X)

    return explainer, shap_values