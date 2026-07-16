import matplotlib.pyplot as plt

# ============================================================
# 📈 Plot 1 — Actual vs Predicted
# ============================================================

def plot_actual_vs_predicted(y_test_actual,y_pred_actual):
        
        # Actual vs Predicted Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_test_actual, y_pred_actual, alpha=0.3, edgecolor='k')

        # limit = max(max(y_test_actual), max(y_pred_actual))
        # ax.plot([0, limit], [0, limit], color="red", linewidth=2)

        ax.plot([0, max(y_test_actual)], [0, max(y_test_actual)], color='red', linewidth=2)
        ax.set_xlabel("Actual Casualties")
        ax.set_ylabel("Predicted Casualties")
        ax.set_title("Actual vs Predicted Casualties (XGBoost Regression)")
        ax.grid(True)

        # plt.tight_layout()
        
        return fig

# ============================================================
# 📉 Plot 2 — Residual Plot
# ============================================================

def plot_residuals(y_test_actual,y_pred_actual):
        
        # Residual Plot
        residuals = y_test_actual - y_pred_actual
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_pred_actual, residuals, alpha=0.3, edgecolor='k')
        ax.axhline(0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel("Predicted Casualties")
        ax.set_ylabel("Residuals (Actual - Predicted)")
        ax.set_title("Residual Plot (Model Error Visualization)")
        ax.grid(True)

        return fig
