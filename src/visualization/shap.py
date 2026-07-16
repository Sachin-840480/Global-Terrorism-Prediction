# ============================================================
# SHAP SUMMARY PLOT
# ============================================================

def plot_summary(shap_values,X):

    # SHAP Summary Plot
    fig = plt.figure(figsize=(10,7))

    shap.summary_plot(
        shap_values,
        X_shap,
        show=False
    )

    pyplot(fig)
    plt.close(fig)

    return fig

# ============================================================
# SHAP BAR PLOT
# ============================================================

def plot_bar(shap_values,X):

    # SHAP Bar Plot
    fig = plt.figure(figsize=(10,6))

    shap.summary_plot(
        shap_values,
        X_shap,
        plot_type="bar",
        show=False
    )

    pyplot(fig)
    plt.close(fig)

    return fig


# ============================================================
# SHAP FORCE PLOT
# ============================================================

def plot_waterfall(explainer,shap_values,X,sample=0):

    # SHAP Waterfall plot
    sample = 0

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

    pyplot(fig)
    plt.close(fig)

    return fig
