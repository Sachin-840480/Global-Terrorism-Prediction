import streamlit as st

from src.utils.plot_style import configure_plot_style

st.set_page_config(
    page_title="Global Terrorism Dashboard",
    page_icon="🌍",
    layout="wide",
)

configure_plot_style()

st.title("🌍 Global Terrorism Dashboard")

st.markdown(
"""
### Machine Learning Based Terrorism Forecasting

Choose a page from the left sidebar.
"""
)

st.write(st.__version__)
