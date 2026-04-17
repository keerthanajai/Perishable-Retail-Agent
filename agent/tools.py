import requests
from typing import Optional

API_BASE = "http://127.0.0.1:8000"


TOOL_DEFINITIONS = [
    {
        "name": "forecast_demand",
        "description": (
            "Predicts the unit sales (demand) for a specific store and item. "
            "Use this when the user asks about expected sales, demand forecast, "
            "or how much of a product will sell."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "store_id": {
                    "type": "integer",
                    "description": "The store ID (1-54)"
                },
                "item_id": {
                    "type": "integer",
                    "description": "The item ID"
                },
                "onpromotion": {
                    "type": "integer",
                    "description": "Whether the item is on promotion: 1=yes, 0=no",
                    "default": 0
                },
                "date": {
                    "type": "string",
                    "description": "Date for the forecast in YYYY-MM-DD format. Optional."
                }
            },
            "required": ["store_id", "item_id"]
        }
    },
    {
        "name": "explain_forecast",
        "description": (
            "Explains WHY the model made a specific forecast using SHAP feature importance. "
            "Use this when the user asks why a forecast is high or low, what factors "
            "are driving demand, or wants to understand the prediction."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "store_id": {
                    "type": "integer",
                    "description": "The store ID"
                },
                "item_id": {
                    "type": "integer",
                    "description": "The item ID"
                },
                "onpromotion": {
                    "type": "integer",
                    "description": "Whether the item is on promotion: 1=yes, 0=no",
                    "default": 0
                },
                "date": {
                    "type": "string",
                    "description": "Date in YYYY-MM-DD format. Optional."
                }
            },
            "required": ["store_id", "item_id"]
        }
    },
    {
        "name": "query_sales_history",
        "description": (
            "Retrieves recent historical sales context for a store and item. "
            "Use this when the user asks about past performance, trends, "
            "or wants context before making a decision."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "store_id": {
                    "type": "integer",
                    "description": "The store ID"
                },
                "item_id": {
                    "type": "integer",
                    "description": "The item ID"
                }
            },
            "required": ["store_id", "item_id"]
        }
    },
    {
        "name": "recommend_action",
        "description": (
            "Generates a full business recommendation: forecast + explanation + "
            "inventory action (STOCK UP / MAINTAIN / REDUCE ORDER). "
            "Use this when the user asks what they should do, whether to order more, "
            "or wants a complete analysis."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "store_id": {
                    "type": "integer",
                    "description": "The store ID"
                },
                "item_id": {
                    "type": "integer",
                    "description": "The item ID"
                },
                "onpromotion": {
                    "type": "integer",
                    "description": "Whether the item is on promotion: 1=yes, 0=no",
                    "default": 0
                },
                "date": {
                    "type": "string",
                    "description": "Date in YYYY-MM-DD format. Optional."
                }
            },
            "required": ["store_id", "item_id"]
        }
    }
]


# --- Tool Execution Functions ---

def forecast_demand(
    store_id: int,
    item_id: int,
    onpromotion: int = 0,
    date: Optional[str] = None
) -> str:
    try:
        resp = requests.post(f"{API_BASE}/predict", json={
            "store_id": store_id,
            "item_id": item_id,
            "onpromotion": onpromotion,
            "date": date
        })
        data = resp.json()

        if resp.status_code != 200:
            return f"Error: {data.get('detail', 'Unknown error')}"

        return (
            f"Forecast for Store {store_id}, Item {item_id} "
            f"({'on promotion' if onpromotion else 'no promotion'}): "
            f"{data['predicted_unit_sales']} units predicted. "
            f"Product family: {data['family']}."
        )
    except Exception as e:
        return f"Tool error: {str(e)}"


def explain_forecast(
    store_id: int,
    item_id: int,
    onpromotion: int = 0,
    date: Optional[str] = None
) -> str:
    try:
        resp = requests.post(f"{API_BASE}/explain", json={
            "store_id": store_id,
            "item_id": item_id,
            "onpromotion": onpromotion,
            "date": date
        })
        data = resp.json()

        if resp.status_code != 200:
            return f"Error: {data.get('detail', 'Unknown error')}"

        explanation = data["explanation"]
        plain = explanation["plain_english"]
        drivers = explanation["top_drivers"]

        breakdown = "\n".join([
            f"  - {d['label']}: {d['direction']} forecast (SHAP: {d['shap_impact']:+.3f})"
            for d in drivers[:5]
        ])

        return f"{plain}\n\nTop feature breakdown:\n{breakdown}"
    except Exception as e:
        return f"Tool error: {str(e)}"


def query_sales_history(store_id: int, item_id: int) -> str:
    """
    Pulls recent sales context from the feature lookup table via the predict
    endpoint (which loads the last known state). Returns a human-readable summary.
    """
    try:
        resp = requests.post(f"{API_BASE}/predict", json={
            "store_id": store_id,
            "item_id": item_id
        })
        data = resp.json()

        if resp.status_code != 200:
            return f"Error: {data.get('detail', 'Unknown error')}"

        return (
            f"Historical context for Store {store_id}, Item {item_id} "
            f"(Family: {data['family']}): "
            f"Most recent forecast baseline is {data['predicted_unit_sales']} units. "
            f"This is based on the last known sales patterns including lag and "
            f"rolling average features from the feature store."
        )
    except Exception as e:
        return f"Tool error: {str(e)}"


def recommend_action(
    store_id: int,
    item_id: int,
    onpromotion: int = 0,
    date: Optional[str] = None
) -> str:
    try:
        resp = requests.post(f"{API_BASE}/forecast", json={
            "store_id": store_id,
            "item_id": item_id,
            "onpromotion": onpromotion,
            "date": date
        })
        data = resp.json()

        if resp.status_code != 200:
            return f"Error: {data.get('detail', 'Unknown error')}"

        return (
            f"RECOMMENDATION: {data['recommendation']}\n"
            f"Reason: {data['recommendation_reason']}\n"
            f"Forecast: {data['predicted_unit_sales']} units\n"
            f"Drivers: {data['explanation']}\n"
            f"Store: {store_id} | Item: {item_id} | Family: {data['family']}"
        )
    except Exception as e:
        return f"Tool error: {str(e)}"


# ── Tool Router ──────────────────────────────────────────────────────────────
def execute_tool(tool_name: str, tool_input: dict) -> str:
    tools = {
        "forecast_demand": forecast_demand,
        "explain_forecast": explain_forecast,
        "query_sales_history": query_sales_history,
        "recommend_action": recommend_action,
    }

    if tool_name not in tools:
        return f"Unknown tool: {tool_name}"

    return tools[tool_name](**tool_input)