# -------------------------------------------------
# 1. Feature engineering
# -------------------------------------------------

def engineer_features(df):

    D = df.copy()

    D["nkill"] = D["nkill"].fillna(0)
    D["nwound"] = D["nwound"].fillna(0)
    D["total_casualties"] = D["nkill"] + D["nwound"]
    D["log_casualties"] = np.log1p(D["total_casualties"])

    # Frequency encoding
    for col in ["region","country","attacktype1","targtype1","weaptype1"]:
        freq = D[col].value_counts()
        D[col + "_freq"] = D[col].map(freq).astype(float)

    # Interaction
    D["region_attack"] = D["region"].astype(str) + "_" + D["attacktype1"].astype(str)
    freq_int = D["region_attack"].value_counts()
    D["region_attack_freq"] = D["region_attack"].map(freq_int).astype(float)

    # Historical means
    D["region_mean"]  = np.log1p(D.groupby("region")["total_casualties"].transform("mean"))
    D["attack_mean"]  = np.log1p(D.groupby("attacktype1")["total_casualties"].transform("mean"))
    D["country_mean"] = np.log1p(D.groupby("country")["total_casualties"].transform("mean"))

    # Category codes
    for col in ["region","country","attacktype1","targtype1","weaptype1"]:
        D[col + "_cat"] = D[col].astype("category").cat.codes

    # Time trend
    ymin, ymax = D["iyear"].min(), D["iyear"].max()
    D["year_trend"] = (D["iyear"] - ymin) / (ymax - ymin)

    # Rolling country mean
    D = D.sort_values(["country","iyear"])
    D["country_5yr_mean"] = (
        D.groupby("country")["total_casualties"]
         .transform(lambda x: x.rolling(5, min_periods=1).mean())
    )
    D["country_5yr_mean"] = np.log1p(D["country_5yr_mean"])

    return D