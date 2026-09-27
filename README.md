# Factory Reallocation & Shipping Optimization Recommendation System

## Objective

A decision-intelligence system for Nassau Candy Distributor that combines historical shipping analytics, lead-time prediction, what-if factory simulation, optimization scoring and explainable reassignment recommendations.

## Dataset

The supplied CSV contains 10,194 rows and 18 fields. It does not contain a factory column, so current factory assignment is derived strictly from the supplied Product → Factory mapping.

## Critical data validation finding

The supplied `Order Date` and `Ship Date` values currently produce lead times of approximately 904–1,642 days when parsed as day-first dates. This is a major data-quality/modeling consideration. The application flags it rather than silently correcting or deleting the observations.

## Installation

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Copy the supplied CSV to:

`data/raw/dataset.csv`

## Train the models

```powershell
python train_model.py
```

The project compares:
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

Metrics:
- RMSE
- MAE
- R²

The best model is saved to `models/shipping_model.pkl`.

## Run Streamlit

```powershell
streamlit run app.py
```

## Important limitation

Destination latitude/longitude is not supplied in the CSV. Therefore the current implementation does not fabricate destination coordinates. A reliable destination geocoding/reference dataset must be supplied before distance-based route optimization is treated as operationally valid.

Likewise, the dataset has Gross Profit but no factory-specific manufacturing or shipping cost. Scenario financial values are therefore configurable estimates, not historical shipping costs.

## Workflow

Data → Validation → Feature Engineering → ML Prediction → Factory Scenario Simulation → Optimization Score → Risk/Confidence → Recommendation

## Future improvements

1. Add a verified city/postal-code coordinate reference.
2. Add real factory capacity constraints.
3. Add actual freight rates.
4. Add factory production costs by product.
5. Add prediction intervals/calibrated uncertainty.
6. Add constrained optimization using capacity and service-level requirements.
