import traceback

import numpy as np
import pandas as pd
import folium

from src.config import ATTACK_COLORS


# ================================================================
# Prediction + Folium Map
# ================================================================

def predict_and_map(df, model, features, future_year, sample_size=300, country_filter=None):
    D = df.copy()
    if country_filter:
        D = D[D["country_txt"].isin(country_filter)]
        if D.empty:
            D = df.copy()

    sample = D.sample(n=min(sample_size, len(D)), random_state=future_year).copy()
    sample["iyear"] = future_year
    sample["nkill"] = sample["nkill"].fillna(0)
    sample["nwound"] = sample["nwound"].fillna(0)
    sample["total_casualties"] = sample["nkill"] + sample["nwound"]
    sample["log_casualties"] = np.log1p(sample["total_casualties"])

    # recreate frequency/interactions using whole df
    for col in ["region", "country", "attacktype1", "targtype1", "weaptype1"]:
        freq = df[col].value_counts().to_dict()
        sample[col + "_freq"] = sample[col].map(freq).fillna(0).astype(float)

    # Interaction frequency (map)
    region_attack_freq_map = (df["region"].astype(str) + "_" + df["attacktype1"].astype(str)).value_counts().to_dict()
    sample["region_attack"] = sample["region"].astype(str) + "_" + sample["attacktype1"].astype(str)
    sample["region_attack_freq"] = sample["region_attack"].map(region_attack_freq_map).fillna(0).astype(float)

    # Historical means: use mapping to avoid misalignment
    region_mean_map = df.groupby("region")["total_casualties"].mean().to_dict()
    attack_mean_map = df.groupby("attacktype1")["total_casualties"].mean().to_dict()
    country_mean_map = df.groupby("country")["total_casualties"].mean().to_dict()

    sample["region_mean"]  = np.log1p(sample["region"].map(region_mean_map).fillna(0).astype(float))
    sample["attack_mean"]  = np.log1p(sample["attacktype1"].map(attack_mean_map).fillna(0).astype(float))
    sample["country_mean"] = np.log1p(sample["country"].map(country_mean_map).fillna(0).astype(float))

    # Category codes (on sample)
    for col in ["region", "country", "attacktype1", "targtype1", "weaptype1"]:
        sample[col + "_cat"] = sample[col].astype("category").cat.codes

    # Year trend relative to original df
    sample["year_trend"] = (future_year - df["iyear"].min()) / max(1, (df["iyear"].max() - df["iyear"].min()))

    # Rolling country 5-year mean (on sample) — use sample grouping (safe)
    sample = sample.sort_values(["country", "iyear"])
    sample["country_5yr_mean"] = (
        sample.groupby("country")["total_casualties"].transform(lambda x: x.rolling(5, min_periods=1).mean())
    )
    sample["country_5yr_mean"] = np.log1p(sample["country_5yr_mean"].fillna(0))

    # Predict
    if model is None:
        sample["predicted_casualties"] = sample["country_mean"].apply(lambda v: np.expm1(v) if pd.notnull(v) else 0)
    else:
        try:
            pred_log = model.predict(sample[features])
            sample["predicted_casualties"] = np.expm1(pred_log).clip(0, None)
        except Exception:
            print("⚠️ Model prediction failed; falling back to country_mean.")
            print(traceback.format_exc())
            sample["predicted_casualties"] = sample["country_mean"].apply(lambda v: np.expm1(v) if pd.notnull(v) else 0)

    # Build folium map
    fmap = folium.Map(location=[sample["latitude"].mean(), sample["longitude"].mean()], zoom_start=2)
    for _, row in sample.iterrows():
        try:
            attack_idx = int(row.get("attacktype1", 0)) if not pd.isna(row.get("attacktype1", None)) else 0
        except Exception:
            attack_idx = 0
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=3 + np.sqrt(max(0, float(row["predicted_casualties"]))),
            color=ATTACK_COLORS.get(attack_idx, 'gray'),
            fill=True, fill_opacity=0.6,
            popup=(f"<b>Country:</b> {row['country_txt']}<br>"
                   f"<b>Attack Type:</b> {row.get('attacktype1_txt', 'unknown')}<br>"
                   f"<b>Predicted Casualties:</b> {int(row['predicted_casualties'])}")
        ).add_to(fmap)

    fmap.get_root().html.add_child(folium.Element("""
        <div style="position: fixed; bottom: 50px; left: 50px; width: 260px;
         border:2px solid grey; z-index:9999; font-size:14px;
         background-color:white; padding: 10px; line-height: 1.4;">
         <b>Attack Type Legend</b><br>
         <i style="color:blue">●</i> Assassination<br>
         <i style="color:green">●</i> Armed Assault<br>
         <i style="color:orange">●</i> Bombing/Explosion<br>
         <i style="color:purple">●</i> Hijacking<br>
         <i style="color:darkred">●</i> Hostage (Kidnapping)<br>
         <i style="color:cadetblue">●</i> Hostage (Barricade)<br>
         <i style="color:darkpurple">●</i> Infrastructure Attack<br>
         <i style="color:darkgreen">●</i> Unknown<br>
         <i style="color:pink">●</i> Other<br>
         <i style="color:lightblue">●</i> Unclassified<br>
        </div>
    """))

    return fmap, sample

