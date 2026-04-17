import shap
import pandas as pd
import numpy as np
from api.predictor import model, FEATURES

explainer = shap.TreeExplainer(model)
print("SHAP explainer ready")


# Human-readable labels for features
FEATURE_LABELS = {
    "lag_7": "sales 7 days ago",
    "lag_14": "sales 14 days ago",
    "lag_28": "sales 28 days ago",
    "lag_365": "sales same day last year",
    "rolling_mean_7": "7-day average sales",
    "rolling_mean_14": "14-day average sales",
    "rolling_mean_28": "28-day average sales",
    "rolling_std_7": "sales volatility (7-day)",
    "rolling_max_7": "peak sales this week",
    "rolling_min_7": "lowest sales this week",
    "onpromotion": "item is on promotion",
    "oil_price": "oil price",
    "temp_avg_c": "average temperature",
    "humidity_avg_pct": "humidity",
    "precip_total_mm": "rainfall",
    "is_holiday": "holiday effect",
    "is_national_holiday": "national holiday",
    "is_weekend": "weekend effect",
    "days_since_promo": "days since last promotion",
    "days_to_promo": "days until next promotion",
    "avg_sales_store_dow": "typical store sales this weekday",
    "avg_sales_item_dow": "typical item sales this weekday",
    "transactions": "store transaction volume",
    "import_value": "import trade volume",
    "export_value": "export trade volume",
    "promo_factor": "promotion lift factor",
    "is_spike": "recent sales spike detected",
    "lag7_x_promo": "promo interaction with recent sales",
    "rolling_x_temp": "temperature interaction with sales trend",
}


def explain(feature_row: pd.DataFrame, top_n: int = 5) -> dict:
    """
    Runs SHAP on the feature row and returns top drivers as plain English.
    """
    shap_values = explainer.shap_values(feature_row)

    #series of feature → shap value
    shap_series = pd.Series(shap_values[0], index=FEATURES)
    shap_abs = shap_series.abs().sort_values(ascending=False)

    top_features = shap_abs.head(top_n).index.tolist()

    drivers = []
    for feat in top_features:
        val = float(feature_row[feat].values[0])
        shap_val = float(shap_series[feat])
        direction = "increased" if shap_val > 0 else "decreased"
        label = FEATURE_LABELS.get(feat, feat.replace("_", " "))

        drivers.append({
            "feature": feat,
            "label": label,
            "value": round(val, 3),
            "shap_impact": round(shap_val, 4),
            "direction": direction,
            "summary": f"{label} ({round(val, 2)}) {direction} the forecast"
        })

    # plain English paragraph for the LLM to use
    top_3 = drivers[:3]
    plain_english = (
        f"The forecast is primarily driven by: "
        + ", ".join([d["summary"] for d in top_3])
        + "."
    )

    return {
        "top_drivers": drivers,
        "plain_english": plain_english
    }