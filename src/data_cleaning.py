import pandas as pd
import numpy as np
from config.product_factory_mapping import PRODUCT_FACTORY

def clean_data(df):
    x = df.copy()
    x["Order Date Parsed"] = pd.to_datetime(x["Order Date"], dayfirst=True, errors="coerce")
    x["Ship Date Parsed"] = pd.to_datetime(x["Ship Date"], dayfirst=True, errors="coerce")
    x["Shipping Lead Time"] = (x["Ship Date Parsed"] - x["Order Date Parsed"]).dt.days
    x["Current Factory"] = x["Product Name"].map(PRODUCT_FACTORY)

    x["Gross Margin %"] = np.where(x["Sales"] != 0, x["Gross Profit"] / x["Sales"] * 100, np.nan)
    x["Profit per Unit"] = np.where(x["Units"] != 0, x["Gross Profit"] / x["Units"], np.nan)
    x["Sales per Unit"] = np.where(x["Units"] != 0, x["Sales"] / x["Units"], np.nan)
    x["Cost per Unit"] = np.where(x["Units"] != 0, x["Cost"] / x["Units"], np.nan)

    x["Data Quality Flag"] = ""
    x.loc[x["Order Date Parsed"].isna() | x["Ship Date Parsed"].isna(), "Data Quality Flag"] += "Invalid date; "
    x.loc[x["Shipping Lead Time"] < 0, "Data Quality Flag"] += "Negative lead time; "
    x.loc[x["Current Factory"].isna(), "Data Quality Flag"] += "Missing factory mapping; "
    x.loc[x["Sales"] < 0, "Data Quality Flag"] += "Negative sales; "
    x.loc[x["Cost"] < 0, "Data Quality Flag"] += "Negative cost; "
    x.loc[x["Units"] < 0, "Data Quality Flag"] += "Negative units; "

    # Do not silently discard records. Use a modeling flag.
    x["Use For Modeling"] = (
        x["Order Date Parsed"].notna()
        & x["Ship Date Parsed"].notna()
        & (x["Shipping Lead Time"] >= 0)
        & x["Current Factory"].notna()
        & (x["Sales"] >= 0)
        & (x["Cost"] >= 0)
        & (x["Units"] >= 0)
    )
    return x
