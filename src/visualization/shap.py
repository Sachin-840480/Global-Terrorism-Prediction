import shap
import matplotlib.pyplot as plt


# ============================================================
# SHAP SUMMARY PLOT
# ============================================================

def plot_summary(shap_values,X_shap):

    # SHAP Summary Plot
    fig = plt.figure(figsize=(10,7))

    shap.summary_plot(
        shap_values,
        X_shap,
        show=False
    )

    return fig

# ============================================================
# SHAP BAR PLOT
# ============================================================

def plot_bar(shap_values,X_shap):

    # SHAP Bar Plot
    fig = plt.figure(figsize=(10,6))

    shap.summary_plot(
        shap_values,
        X_shap,
        plot_type="bar",
        show=False
    )

    return fig


# ============================================================
# SHAP FORCE PLOT
# ============================================================

def plot_waterfall(explainer,shap_values,X_shap,sample=0):

    fig, ax = plt.subplots(figsize=(12,6))

    explanation = shap.Explanation(
        values=shap_values[sample],
        base_values=explainer.expected_value,
        data=X_shap.iloc[sample],
        feature_names=X_shap.columns
    )

    shap.plots.waterfall(
        explanation,
        show=False
    )

    return fig
