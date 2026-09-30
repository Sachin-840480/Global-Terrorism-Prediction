BASE_FEATURES = [

        # baseline
        "iyear", "imonth",
        "region_freq", "country_freq", "attacktype1_freq",
        "targtype1_freq", "weaptype1_freq",
        "success", "region_attack_freq",

        # categories
        "region_cat", "country_cat", "attacktype1_cat",
        "targtype1_cat", "weaptype1_cat",

        # historical means
        "region_mean", "attack_mean", "country_mean",

        # time & rolling
        "year_trend", "country_5yr_mean",
]

EXPERIMENTAL_FEATURES = [

    # NEW features (boost accuracy)
    "region_year_mean",
    "rolling_casualties",
    "casualty_density",
    "country_attack",
    "weapon_target",
]

ALL_FEATURES = BASE_FEATURES + EXPERIMENTAL_FEATURES

print(ALL_FEATURES)