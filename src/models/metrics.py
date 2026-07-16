import pandas as pd

from src.config import COMPARISON_PATH


# ================================================================
# LOADER FUNCTION FOR MODEL COMPARISON TABLE
# ================================================================

def load_model_comparison():

    try:
        return pd.read_csv(COMPARISON_PATH)

    except Exception:
        return pd.DataFrame()