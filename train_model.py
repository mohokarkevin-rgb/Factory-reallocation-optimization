from pathlib import Path
import json
from src.data_loader import load_data, quality_audit
from src.data_cleaning import clean_data
from src.feature_engineering import add_features
from src.model_training import train_models

DATA = Path("data/raw/dataset.csv")
OUT = Path("outputs/reports")
OUT.mkdir(parents=True, exist_ok=True)

df = add_features(clean_data(load_data(DATA)))
audit = quality_audit(df)

with open(OUT / "data_quality.json", "w", encoding="utf-8") as f:
    json.dump(audit, f, indent=2, default=str)

model, metrics, predictions = train_models(df)

print("Training complete.")
print(metrics.to_string(index=False))
print(f"Best model: {metrics.iloc[0]['Model']}")
print("Important: inspect data_quality.json before interpreting the lead-time target.")
