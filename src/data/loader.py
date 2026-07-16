import pandas as pd
import streamlit as st

from src.config import DATA_PATH


@st.cache_data(show_spinner=False)
def load_data():

    cols = [
        "iyear",
        "imonth",
        "country_txt",
        "region_txt",
        "region",
        "country",
        "latitude",
        "longitude",
        "attacktype1",
        "attacktype1_txt",
        "targtype1",
        "targtype1_txt",
        "weaptype1",
        "weaptype1_txt",
        "gname",
        "success",
        "nkill",
        "nwound",
    ]

    df = pd.read_csv(
        DATA_PATH,
        usecols=cols,
        encoding="ISO-8859-1",
        low_memory=False,
    )

    df["nkill"] = df["nkill"].fillna(0)
    df["nwound"] = df["nwound"].fillna(0)

    df["total_casualties"] = df["nkill"] + df["nwound"]

    df.dropna(
        subset=["latitude", "longitude"],
        inplace=True,
    )

    return df