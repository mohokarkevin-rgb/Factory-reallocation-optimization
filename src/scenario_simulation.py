import pandas as pd
from config.factories import FACTORIES
from src.model_training import CATEGORICAL, NUMERICAL, TARGET

def build_scenario_row(row, factory):
    x = row.copy()
    x["Current Factory"] = factory
    x["Current Factory Latitude"] = FACTORIES[factory]["latitude"]
    x["Current Factory Longitude"] = FACTORIES[factory]["longitude"]
    return x

def simulate_product(product_name, df, model):
    base = df[df["Product Name"] == product_name].copy()
    if base.empty:
        return pd.DataFrame()
    # Use representative product inputs: median numeric values and modal categorical values.
    r = base.iloc[0].copy()
    r["Units"] = base["Units"].median()
    for c in ["Sales","Cost","Gross Profit","Gross Margin %","Profit per Unit","Sales per Unit","Cost per Unit"]:
        if c in base:
            r[c] = base[c].median()
    out = []
    current = base["Current Factory"].mode().iloc[0]
    for factory in FACTORIES:
        s = build_scenario_row(r, factory)
        X = pd.DataFrame([s])
        pred = float(model.predict(X[CATEGORICAL + NUMERICAL])[0])
        out.append({
            "Product": product_name,
            "Current Factory": current,
            "Scenario Factory": factory,
            "Predicted Lead Time": pred,
            "Units": float(r["Units"]),
            "Gross Profit": float(r["Gross Profit"]),
        })
    result = pd.DataFrame(out)
    current_pred = float(result.loc[result["Scenario Factory"] == current, "Predicted Lead Time"].iloc[0])
    result["Lead Time Reduction"] = current_pred - result["Predicted Lead Time"]
    result["Lead Time Reduction %"] = result["Lead Time Reduction"] / current_pred * 100 if current_pred else 0
    return result
