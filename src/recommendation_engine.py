from src.optimization import estimate_scenario_financials, score_recommendations

def recommend(result, speed_weight=.5, profit_weight=.5,
              cost_per_unit=0.0, handling_cost=0.0, reassignment_penalty=0.0):
    x = estimate_scenario_financials(
        result,
        cost_per_unit=cost_per_unit,
        handling_cost=handling_cost,
        reassignment_penalty=reassignment_penalty,
    )
    x = score_recommendations(x, speed_weight, profit_weight)
    x["Confidence Score"] = 75.0
    x["Reason"] = x.apply(
        lambda r: (
            f"Predicted lead-time change is {r['Lead Time Reduction %']:.1f}%. "
            f"Scenario financial impact is estimated from configurable assumptions. "
            f"Risk is {r['Risk']}."
        ), axis=1
    )
    return x
