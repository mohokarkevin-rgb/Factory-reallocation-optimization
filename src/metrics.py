def lead_time_summary(df):
    s = df["Shipping Lead Time"].dropna()
    return {
        "min": s.min(),
        "max": s.max(),
        "mean": s.mean(),
        "median": s.median(),
        "std": s.std(),
        "p25": s.quantile(.25),
        "p75": s.quantile(.75),
        "p95": s.quantile(.95),
    }
