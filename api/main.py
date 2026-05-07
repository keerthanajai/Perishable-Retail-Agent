from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List
from api.predictor import predict
from api.explainer import explain
from agent.agent import DemandAgent
import os

app = FastAPI(
    title="Demand Intelligence Agent API",
    description="XGBoost demand forecasting with SHAP explainability — built for perishable retail.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_agent = DemandAgent()


# --- Request / Response Schemas ---

class PredictRequest(BaseModel):
    store_id: int
    item_id: int
    onpromotion: Optional[int] = 0
    date: Optional[str] = None  

class ExplainRequest(BaseModel):
    store_id: int
    item_id: int
    onpromotion: Optional[int] = 0
    date: Optional[str] = None
    top_n: Optional[int] = 5

class ForecastWithExplanationRequest(BaseModel):
    store_id: int
    item_id: int
    onpromotion: Optional[int] = 0
    date: Optional[str] = None


# --- Routes ---

@app.get("/")
def root():
    return {
        "message": "Demand Intelligence Agent API is running.",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_endpoint(req: PredictRequest):
    """
    Returns predicted unit_sales for a given store + item.
    """
    try:
        result = predict(
            store_id=req.store_id,
            item_id=req.item_id,
            onpromotion=req.onpromotion,
            date=req.date
        )
        result.pop("feature_row", None)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/explain")
def explain_endpoint(req: ExplainRequest):
    """
    Returns SHAP-based explanation of what drove the forecast.
    """
    try:
        pred_result = predict(
            store_id=req.store_id,
            item_id=req.item_id,
            onpromotion=req.onpromotion,
            date=req.date
        )
        explanation = explain(
            feature_row=pred_result["feature_row"],
            top_n=req.top_n
        )
        return {
            "store_id": req.store_id,
            "item_id": req.item_id,
            "predicted_unit_sales": pred_result["predicted_unit_sales"],
            "explanation": explanation
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


@app.post("/forecast")
def forecast_with_explanation(req: ForecastWithExplanationRequest):
    """
    One-shot endpoint: returns forecast + explanation + recommendation.
    Used by the agent's recommend_action tool.
    """
    try:
        pred_result = predict(
            store_id=req.store_id,
            item_id=req.item_id,
            onpromotion=req.onpromotion,
            date=req.date
        )
        explanation = explain(feature_row=pred_result["feature_row"])

        forecast = pred_result["predicted_unit_sales"]
        rolling_mean = float(pred_result["feature_row"]["rolling_mean_7"].values[0])

        # Simple rule-based recommendation
        pct_change = ((forecast - rolling_mean) / (rolling_mean + 1e-8)) * 100
        if pct_change > 15:
            action = "STOCK UP"
            reason = f"Forecast is {pct_change:.1f}% above recent average — increase order quantity."
        elif pct_change < -15:
            action = "REDUCE ORDER"
            reason = f"Forecast is {abs(pct_change):.1f}% below recent average — reduce order to avoid waste."
        else:
            action = "MAINTAIN"
            reason = f"Forecast is within normal range ({pct_change:+.1f}% vs recent average)."

        return {
            "store_id": req.store_id,
            "item_id": req.item_id,
            "family": pred_result["family"],
            "date": pred_result["date"],
            "predicted_unit_sales": forecast,
            "recommendation": action,
            "recommendation_reason": reason,
            "explanation": explanation["plain_english"],
            "top_drivers": explanation["top_drivers"]
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast failed: {str(e)}")


# --- Html UI chat ---

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[dict]] = []

@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    """
    Main chat endpoint for the HTML UI.
    Runs the full ReAct agent loop and returns a response.
    """
    try:
        response = _agent.chat(req.message)
        return {"response": response}
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")


@app.post("/reset")
def reset_endpoint():
    """Clears agent conversation memory."""
    _agent.reset()
    return {"status": "reset"}


# ── Serve HTML UI ─────────────────────────────────────────────────────────────

@app.get("/ui")
def serve_ui():
    ui_path = os.path.join(os.path.dirname(__file__), '..', 'ui', 'index.html')
    return FileResponse(os.path.abspath(ui_path))