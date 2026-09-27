import numpy as np
from config.factories import FACTORIES

def estimate_scenario_financials(result, cost_per_mile=0.0, cost_per_unit=0.0,
                                  handling_cost=0.0, reassignment_penalty=0.0):
    # Destination coordinates are not present in the supplied CSV, so distance is
    # intentionally not fabricated. Cost parameters remain configurable.
    x = result.copy()
    x["Estimated Shipping Cost"] = x["Units"] * cost_per_unit
    x["Estimated Incremental Cost"] = (
        x["Estimated Shipping Cost"] + handling_cost + reassignment_penalty
    )
    x["Estimated Scenario Profit Impact"] = -x["Estimated Incremental Cost"]
    return x

def score_recommendations(result, speed_weight=0.5, profit_weight=0.5):
    x = result.copy()
    if x.empty:
        return x
    x["Speed Score"] = x["Lead Time Reduction %"].clip(lower=0)
    if x["Estimated Scenario Profit Impact"].nunique() > 1:
        p = x["Estimated Scenario Profit Impact"]
        x["Profit Score"] = (p - p.min()) / (p.max() - p.min())
    else:
        x["Profit Score"] = 0.0
    x["Recommendation Score"] = (
        speed_weight * (x["Speed Score"] / max(x["Speed Score"].max(), 1))
        + profit_weight * x["Profit Score"]
    ) * 100
    x["Risk"] = np.where(
        x["Lead Time Reduction %"] <= 0, "High Risk",
        np.where(x["Lead Time Reduction %"] < 10, "Medium Risk", "Low Risk")
    )
    return x.sort_values("Recommendation Score", ascending=False)
