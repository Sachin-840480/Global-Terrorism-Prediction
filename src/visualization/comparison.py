# ================================================================
# 📈 MODEL COMPARISON TAB
# ================================================================

def plot_model_comparison(comparison):

    # Comparative Analysis of Regression Models
    comparison = load_model_comparison()

    if comparison.empty:
        st.warning("Model comparison file not found.")
    else:
        comparison["Model"] = comparison["Model"].str.replace(" ", "\n")
        fig, ax1 = plt.subplots(figsize=(14,6))
        fig.patch.set_facecolor("white")
        ax1.set_facecolor("white")

        # ==========================
        # GRID BEHIND EVERYTHING
        # ==========================

        ax1.set_axisbelow(True)
        ax1.grid(axis="y", color="#D0D0D0", linewidth=0.8, alpha=0.8)
        ax1.grid(axis="x", visible=False)

        # ==========================
        # R² BAR CHART
        # ==========================

        bars = ax1.bar(comparison["Model"], comparison["R2"], width=0.5, color="#4FC3CF", label="R² Score ↑", zorder=3)
        ax1.margins(x=0.08)
        ax1.set_ylim(0,0.6)
        ax1.set_yticks([0,0.2,0.4,0.6])
        ax1.set_ylabel("R² Score", fontsize=11)

        # ==========================
        # MAE LINE AXIS
        # ==========================

        ax2 = ax1.twinx()
        ax2.grid(False)
        ax2.plot(comparison["Model"], comparison["MAE"], marker="o", markersize=6, linewidth=3, color="#F5B83D", label="MAE ↓", zorder=5)
        ax2.set_ylim(0,6)
        ax2.set_ylabel("MAE (Casualties)", fontsize=11)

        # ==========================
        # REMOVE BORDERS
        # ==========================

        for ax in [ax1, ax2]:
            for spine in ax.spines.values():
                spine.set_visible(False)

        # ==========================
        # VALUE LABELS
        # ==========================

        # R² labels
        for bar in bars:
            value = bar.get_height()
            ax1.text(bar.get_x()+bar.get_width()/2, value+0.015, f"{value:.4f}", ha="center", fontsize=10, fontweight="bold", zorder=6)

        # MAE labels
        for i,value in enumerate(comparison["MAE"]):
            ax2.text(i, value+0.15, f"{value:.2f}", ha="center", fontsize=10, fontweight="bold", zorder=6)

        # ==========================
        # AXIS FORMATTING
        # ==========================

        ax1.tick_params(axis="x", labelsize=9)
        ax1.tick_params(axis="y", labelsize=10)
        ax2.tick_params(axis="y", labelsize=10)

        plt.xticks(rotation=0, ha="center", fontweight="bold")

        # ==========================
        # LEGEND
        # ==========================

        h1,l1 = ax1.get_legend_handles_labels()
        h2,l2 = ax2.get_legend_handles_labels()

        ax1.legend(h1+h2, l1+l2, loc="upper center", bbox_to_anchor=(0.5,-0.16), ncol=2, frameon=False, fontsize=11)

        plt.tight_layout()
        pyplot(fig)

        return fig