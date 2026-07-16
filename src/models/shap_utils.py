import shap
import streamlit as st

def compute_shap(_model,X):

    explainer = shap.TreeExplainer(_model)
    shap_values = explainer.shap_values(X)

    return explainer, shap_values