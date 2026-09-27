def add_features(df):
    x = df.copy()
    x["Order Year"] = x["Order Date Parsed"].dt.year
    x["Order Month"] = x["Order Date Parsed"].dt.month
    x["Order Day"] = x["Order Date Parsed"].dt.day
    x["Order Day of Week"] = x["Order Date Parsed"].dt.dayofweek
    x["Ship Year"] = x["Ship Date Parsed"].dt.year
    x["Ship Month"] = x["Ship Date Parsed"].dt.month

    from config.factories import FACTORIES
    x["Current Factory Latitude"] = x["Current Factory"].map(lambda f: FACTORIES.get(f, {}).get("latitude"))
    x["Current Factory Longitude"] = x["Current Factory"].map(lambda f: FACTORIES.get(f, {}).get("longitude"))
    return x
