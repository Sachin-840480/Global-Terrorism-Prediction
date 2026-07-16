# ============================================================
# 📈 Plot 1 — Actual vs Predicted
# ============================================================

def plot_actual_vs_predicted(y_actual,y_pred):
        
        # Actual vs Predicted Polot
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.scatter(y_test_actual, y_pred_actual, alpha=0.3, edgecolor='k')
        ax1.plot([0, max(y_test_actual)], [0, max(y_test_actual)], color='red', linewidth=2)
        ax1.set_xlabel("Actual Casualties")
        ax1.set_ylabel("Predicted Casualties")
        ax1.set_title("Actual vs Predicted Casualties (XGBoost Regression)")
        ax1.grid(True)
        pyplot(fig1)
        
        return fig

# ============================================================
# 📉 Plot 2 — Residual Plot
# ============================================================

def plot_residuals(y_actual,y_pred):
        
        # Residual Plot
        residuals = y_test_actual - y_pred_actual
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.scatter(y_pred_actual, residuals, alpha=0.3, edgecolor='k')
        ax2.axhline(0, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel("Predicted Casualties")
        ax2.set_ylabel("Residuals (Actual - Predicted)")
        ax2.set_title("Residual Plot (Model Error Visualization)")
        ax2.grid(True)
        pyplot(fig2)

        return fig
