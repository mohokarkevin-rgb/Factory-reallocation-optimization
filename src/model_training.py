from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

CATEGORICAL = ["Ship Mode","Region","Division","Product ID","Product Name","Current Factory"]
NUMERICAL = [
    "Units","Sales","Cost","Gross Profit","Gross Margin %",
    "Profit per Unit","Sales per Unit","Cost per Unit",
    "Order Year","Order Month","Order Day","Order Day of Week",
    "Ship Year","Ship Month"
]
TARGET = "Shipping Lead Time"

def train_models(df, model_dir="models"):
    data = df[df["Use For Modeling"]].copy()
    cols = [c for c in CATEGORICAL + NUMERICAL if c in data.columns]
    X, y = data[cols], data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.2, random_state=42
    )

    pre = ColumnTransformer([
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), CATEGORICAL),
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), NUMERICAL)
    ])

    candidates = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=250, random_state=42, n_jobs=-1, max_depth=None
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42)
    }

    results, fitted = [], {}
    for name, estimator in candidates.items():
        pipe = Pipeline([("preprocessor", pre), ("model", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        rmse = mean_squared_error(y_test, pred) ** 0.5
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        results.append({"Model": name, "RMSE": rmse, "MAE": mae, "R2": r2})
        fitted[name] = pipe

    metrics = pd.DataFrame(results).sort_values(["RMSE","MAE"]).reset_index(drop=True)
    best_name = metrics.iloc[0]["Model"]
    best_model = fitted[best_name]

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, Path(model_dir) / "shipping_model.pkl")
    metrics.to_csv(Path(model_dir) / "model_comparison.csv", index=False)

    pred_df = X_test.copy()
    pred_df["Actual Lead Time"] = y_test.values
    pred_df["Predicted Lead Time"] = best_model.predict(X_test)
    pred_df.to_csv(Path(model_dir) / "test_predictions.csv", index=False)

    return best_model, metrics, pred_df
