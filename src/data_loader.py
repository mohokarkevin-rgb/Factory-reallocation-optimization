from pathlib import Path
import pandas as pd

EXPECTED_COLUMNS = [
    "Row ID","Order ID","Order Date","Ship Date","Ship Mode","Customer ID",
    "Country/Region","City","State/Province","Postal Code","Division","Region",
    "Product ID","Product Name","Sales","Units","Gross Profit","Cost"
]

def load_data(path):
    df = pd.read_csv(path)
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df

def quality_audit(df):
    report = {
        "rows": len(df),
        "columns": len(df.columns),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_order_ids": int(df["Order ID"].duplicated().sum()),
        "missing_cells": int(df.isna().sum().sum()),
        "missing_by_column": df.isna().sum().to_dict(),
        "negative_sales": int((df["Sales"] < 0).sum()),
        "negative_cost": int((df["Cost"] < 0).sum()),
        "negative_units": int((df["Units"] < 0).sum()),
    }
    od = pd.to_datetime(df["Order Date"], dayfirst=True, errors="coerce")
    sd = pd.to_datetime(df["Ship Date"], dayfirst=True, errors="coerce")
    lead = (sd - od).dt.days
    report["invalid_order_dates"] = int(od.isna().sum())
    report["invalid_ship_dates"] = int(sd.isna().sum())
    report["negative_lead_times"] = int((lead < 0).sum())
    report["lead_time_min"] = float(lead.min())
    report["lead_time_max"] = float(lead.max())
    report["lead_time_mean"] = float(lead.mean())
    report["lead_time_median"] = float(lead.median())
    report["lead_time_p25"] = float(lead.quantile(.25))
    report["lead_time_p75"] = float(lead.quantile(.75))
    report["lead_time_p95"] = float(lead.quantile(.95))
    return report
