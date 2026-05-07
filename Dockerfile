"""
Unit tests for Demand Intelligence Agent tools.
Run with: python3 -m pytest tests/ -v
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# ── Predictor Tests ───────────────────────────────────────────────────────────

class TestPredictor:

    def test_predict_returns_valid_structure(self):
        """Prediction result has all required keys."""
        from api.predictor import predict
        result = predict(store_id=51, item_id=1239986)

        assert "predicted_unit_sales" in result
        assert "store_id" in result
        assert "item_id" in result
        assert "family" in result
        assert "feature_row" in result

    def test_predict_returns_non_negative(self):
        """Predicted sales should never be negative."""
        from api.predictor import predict
        result = predict(store_id=51, item_id=1239986)

        assert result["predicted_unit_sales"] >= 0

    def test_predict_correct_store_item_echoed(self):
        """Result should echo back the requested store and item."""
        from api.predictor import predict
        result = predict(store_id=44, item_id=1503844)

        assert result["store_id"] == 44
        assert result["item_id"] == 1503844

    def test_predict_with_promotion(self):
        """Promotion flag should be accepted without error."""
        from api.predictor import predict
        result = predict(store_id=51, item_id=1239986, onpromotion=1)

        assert result["predicted_unit_sales"] >= 0

    def test_predict_with_date(self):
        """Date override should be accepted without error."""
        from api.predictor import predict
        result = predict(store_id=51, item_id=1239986, date="2017-03-15")

        assert result["date"] == "2017-03-15"

    def test_predict_invalid_store_raises(self):
        """Unknown store+item combo should raise ValueError."""
        from api.predictor import predict
        with pytest.raises(ValueError):
            predict(store_id=9999, item_id=9999999)

    def test_predict_feature_row_has_correct_columns(self):
        """Feature row passed to explainer should match the model's feature list."""
        from api.predictor import predict, FEATURES
        result = predict(store_id=51, item_id=1239986)
        feature_row = result["feature_row"]

        assert list(feature_row.columns) == FEATURES


# ── Explainer Tests ───────────────────────────────────────────────────────────

class TestExplainer:

    def test_explain_returns_valid_structure(self):
        """Explanation should have top_drivers and plain_english keys."""
        from api.predictor import predict
        from api.explainer import explain

        result = predict(store_id=51, item_id=1239986)
        explanation = explain(result["feature_row"])

        assert "top_drivers" in explanation
        assert "plain_english" in explanation

    def test_explain_returns_five_drivers_by_default(self):
        """Should return 5 top drivers by default."""
        from api.predictor import predict
        from api.explainer import explain

        result = predict(store_id=51, item_id=1239986)
        explanation = explain(result["feature_row"], top_n=5)

        assert len(explanation["top_drivers"]) == 5

    def test_explain_driver_has_required_fields(self):
        """Each driver should have feature, label, direction, shap_impact."""
        from api.predictor import predict
        from api.explainer import explain

        result = predict(store_id=44, item_id=1503844)
        explanation = explain(result["feature_row"])
        driver = explanation["top_drivers"][0]

        assert "feature" in driver
        assert "label" in driver
        assert "direction" in driver
        assert "shap_impact" in driver
        assert driver["direction"] in ("increased", "decreased")

    def test_explain_plain_english_is_string(self):
        """Plain English explanation should be a non-empty string."""
        from api.predictor import predict
        from api.explainer import explain

        result = predict(store_id=3, item_id=1503844)
        explanation = explain(result["feature_row"])

        assert isinstance(explanation["plain_english"], str)
        assert len(explanation["plain_english"]) > 10


# ── Memory Tests ──────────────────────────────────────────────────────────────

class TestMemory:

    def test_memory_adds_messages(self):
        """Memory should store user and assistant messages."""
        from agent.memory import ConversationMemory
        mem = ConversationMemory()

        mem.add_user("Hello")
        mem.add_assistant("Hi there")

        assert len(mem.get_history()) == 2

    def test_memory_roles_are_correct(self):
        """Messages should have correct roles."""
        from agent.memory import ConversationMemory
        mem = ConversationMemory()

        mem.add_user("Hello")
        mem.add_assistant("Hi")

        assert mem.get_history()[0]["role"] == "user"
        assert mem.get_history()[1]["role"] == "assistant"

    def test_memory_context_update(self):
        """Context should store and retrieve last used store/item."""
        from agent.memory import ConversationMemory
        mem = ConversationMemory()

        mem.update_context(store_id=51, item_id=1239986)

        assert mem.get_context()["store_id"] == 51
        assert mem.get_context()["item_id"] == 1239986

    def test_memory_clear(self):
        """Clear should reset history and context."""
        from agent.memory import ConversationMemory
        mem = ConversationMemory()

        mem.add_user("test")
        mem.update_context(store_id=1)
        mem.clear()

        assert len(mem.get_history()) == 0
        assert mem.get_context() == {}

    def test_memory_trims_on_overflow(self):
        """Memory should not exceed max_turns * 2 messages."""
        from agent.memory import ConversationMemory
        mem = ConversationMemory(max_turns=3)

        for i in range(10):
            mem.add_user(f"message {i}")
            mem.add_assistant(f"response {i}")

        assert len(mem.get_history()) <= 6


# ── Tool Router Tests ─────────────────────────────────────────────────────────

class TestToolRouter:

    def test_unknown_tool_returns_error_string(self):
        """Calling an unknown tool should return an error string, not raise."""
        from agent.tools import execute_tool
        result = execute_tool("nonexistent_tool", {})

        assert "Unknown tool" in result

    def test_forecast_tool_returns_string(self):
        """forecast_demand tool should return a string result."""
        from agent.tools import execute_tool
        result = execute_tool("forecast_demand", {
            "store_id": 51,
            "item_id": 1239986
        })

        assert isinstance(result, str)
        assert len(result) > 0