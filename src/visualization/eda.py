def plot_attacks_per_year(df):

    # Attacks per year
    yearly = df.groupby('iyear').size()
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(yearly.index, yearly.values, color='crimson', linewidth=2)
    ax.set_title("Global Terror Attacks per Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Attacks")
    pyplot(fig)

    return fig

def plot_top_countries(df):
       
    # Top countries
    top_countries = df['country_txt'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(y=top_countries.index, x=top_countries.values, palette='Reds_r', ax=ax)
    ax.set_title("Top 10 Most Affected Countries")
    ax.set_xlabel("Number of Attacks")
    ax.set_ylabel("Country")
    pyplot(fig)

    return fig

def plot_top_countries(df):

    # Attack types
    attack_counts = df['attacktype1_txt'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(y=attack_counts.index, x=attack_counts.values, palette='Blues_r', ax=ax)
    ax.set_title("Most Common Attack Types")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Attack Type")
    pyplot(fig)

    return fig

def plot_top_countries(df):
    
    # Terrorist Groups
    group_counts = (df[df['gname'] != 'Unknown']['gname'].value_counts().head(50))  # Exclude 'Unknown' to focus on identified groups
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.barplot(y=group_counts.index, x=group_counts.values, palette='viridis', ax=ax)
    ax.set_title("Most Active Terrorist Organizations")
    ax.set_xlabel("Number of Attacks")
    ax.set_ylabel("Terrorist Organization")
    ax.margins(x=0.01, y=0.01) 
    pyplot(fig)

    return fig

def plot_top_countries(df):

    # Average casualties
    casualties = df.groupby('attacktype1_txt')['total_casualties'].mean().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=casualties.values, y=casualties.index, palette='coolwarm', ax=ax)
    ax.set_title("Average Casualties per Attack Type")
    ax.set_xlabel("Average Casualties")
    ax.set_ylabel("Attack Type")
    st.pyplot(fig)

    return fig


def plot_top_countries(df):
        
    # Regional trends
    trends = df.groupby(['iyear', 'region_txt']).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(14, 7))
    trends.plot(ax=ax, linewidth=1.5)
    ax.set_title("Regional Terrorism Trends (1970–2020)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Attacks")
    st.pyplot(fig)

    return fig


def plot_top_countries(df):
        
    # Correlation heatmap
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(df[['nkill', 'nwound', 'total_casualties']].corr(),annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title("Correlation Between Casualty Variables")
    st.pyplot(fig)

    return fig


def plot_weapon_distribution(df):

    # --- Weapon Type Distribution (legend-based, no overlapping labels) ---
    counts = df['weaptype1_txt'].value_counts().head(7)

    fig, ax = plt.subplots(figsize=(10, 7))  # wider to fit legend nicely
    colors = sns.color_palette('Paired', n_colors=len(counts))

    # Show only percentages on the pie; put full labels in the legend
    wedges, _texts, autotexts = ax.pie(
        counts.values,
        labels=None,                    # no labels on slices
        autopct='%1.1f%%',
        startangle=140,
        colors=colors,
        pctdistance=0.8,                # pull % text inward a bit
        textprops={'fontsize': 10}
    )

    # Bolden percentage text for readability
    for t in autotexts:
        t.set_fontweight('bold')
        t.set_color('black')

    ax.set_title("Weapon Type Distribution in Attacks", fontsize=14)

    # Build legend with full labels + counts + share
    total = counts.sum()
    legend_labels = [f"{name} — {val} ({val/total:.1%})" for name, val in counts.items()]
    ax.legend(
        wedges,
        legend_labels,
        title="Weapon Types",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.,
        frameon=True
    )

    plt.tight_layout()
    pyplot(fig)

    return fig
   