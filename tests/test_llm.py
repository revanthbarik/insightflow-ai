"""Unit tests for local Ollama prompt and availability helpers."""

from __future__ import annotations

import requests

from src import llm


def test_check_ollama_available_handles_connection_error(monkeypatch):
    def _raise(*args, **kwargs):
        raise requests.RequestException("connection refused")

    monkeypatch.setattr(llm.requests, "request", _raise)
    assert llm.check_ollama_available() is False


def test_build_rag_prompt_contains_safety_instructions():
    prompt = llm.build_rag_prompt(
        user_question="What does forecast quality mean?",
        retrieved_context="Forecast quality compares weak forecast against total value.",
        computed_context="Current weak forecast ratio is already computed by Pandas.",
    )
    assert "Do not calculate new numbers." in prompt
    assert "Use only the provided local context." in prompt
    assert "Do not invent numbers." in prompt
    assert "Do not invent metrics." in prompt


def test_build_computed_prompt_contains_grounding_and_aed_rules():
    prompt = llm.build_computed_prompt(
        user_question="Explain the monthly pipeline trend",
        computed_context={
            "answer_title": "Monthly Pipeline Trend by Closing Date",
            "answer_source": "time trend chart",
            "highest_month": "Mar 2025",
            "highest_amount": 2760000.0,
            "currency": "AED",
            "table_preview": [
                {"Closing_Month": "Jan 2025", "Total_Amount": 1200000.0},
                {"Closing_Month": "Mar 2025", "Total_Amount": 2760000.0},
            ],
            "recommended_action": "Review high-closing months closely.",
            "computed_answer": "This chart shows pipeline grouped by month.",
            "metric_type": "monthly_pipeline_trend",
        },
    )
    assert "Pandas has already computed the result." in prompt
    assert "Do not calculate." in prompt
    assert "Use AED only." in prompt
    assert "Do not infer values from charts." in prompt
    assert '"highest_month":"Mar 2025"' in prompt
    assert "computed_context" in prompt


def test_synthesize_computed_answer_returns_safe_message_when_context_is_incomplete():
    response = llm.synthesize_computed_answer(
        user_question="Explain this result",
        computed_context={"title": "Incomplete"},
    )
    assert response["ok"] is True
    assert "not enough structured context" in response["content"].lower()
