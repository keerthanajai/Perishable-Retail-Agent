import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "data" / "model_artifacts" / "demand_model.pkl"
FEATURES_PATH = BASE_DIR / "data" / "model_artifacts" / "feature_list.pkl"
LOOKUP_PATH = BASE_DIR / "data" / "sample_features.parquet"

# Load once at startup
model = joblib.load(MODEL_PATH)
FEATURES = joblib.load(FEATURES_PATH)
lookup_df = pd.read_parquet(LOOKUP_PATH)

logger = logging.getLogger(__name__)
logger.info(f"Model loaded | {len(FEATURES)} features | {lookup_df.shape[0]} store-item combos")


def get_feature_row(store_id: int, item_id: int) -> pd.DataFrame:
    """
    Fetches the most recent known feature row for a given store+item.
    This provides the lag/rolling features the model needs.
    """
    row = lookup_df[
        (lookup_df["store_id"] == store_id) &
        (lookup_df["item_id"] == item_id)
    ]

    if row.empty:
        raise ValueError(f"No data found for store_id={store_id}, item_id={item_id}. "
                         f"Try store_id between 1-54 and a valid item_id.")
    return row.copy()


def predict(store_id: int, item_id: int, onpromotion: int = 0, date: str = None) -> dict:
    """
    Runs inference for a given store + item.
    Overrides onpromotion and date-based features if provided.
    """
    row = get_feature_row(store_id, item_id)

    row["onpromotion"] = onpromotion
    if date:
        dt = pd.to_datetime(date)
        row["date"] = dt
        row["day_of_week"] = dt.dayofweek
        row["month"] = dt.month
        row["year"] = dt.year
        row["day_of_month"] = dt.day
        row["week_of_year"] = dt.isocalendar()[1]
        row["is_weekend"] = int(dt.dayofweek >= 5)
        row["is_month_start"] = int(dt.is_month_start)
        row["is_month_end"] = int(dt.is_month_end)
        row["quarter"] = dt.quarter

    row["lag7_x_promo"] = row["lag_7"] * row["onpromotion"]
    row["rolling_x_promo"] = row["rolling_mean_7"] * row["onpromotion"]

    X = row[FEATURES]

    # Predict (model was trained on log1p target)
    pred_log = model.predict(X)[0]
    pred = float(np.expm1(pred_log))
    pred = max(0.0, round(pred, 2))  # clip negatives

    return {
        "store_id": store_id,
        "item_id": item_id,
        "date": str(date) if date else "latest",
        "onpromotion": onpromotion,
        "family": str(row["family"].values[0]),
        "predicted_unit_sales": pred,
        "feature_row": X
    }