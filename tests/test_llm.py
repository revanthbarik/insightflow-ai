"""Unit tests for local Ollama prompt and availability helpers."""

from __future__ import annotations

import requests
from types import SimpleNamespace

from src import llm
from src import config


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


def test_fast_explanation_preserves_structured_response_without_sentence_trimming(monkeypatch):
    structured_response = (
        "**What this indicates**\nA.\n\n"
        "**Likely causes to investigate**\nB.\n\n"
        "**Practical next steps**\nC.\n\n"
        "**Recommended follow-up analysis**\nD."
    )
    captured: dict[str, object] = {}

    def _fake_request(method, endpoint, payload=None):
        captured["payload"] = payload
        return {"response": structured_response}, None

    monkeypatch.setattr(llm, "_safe_json_request", _fake_request)
    response = llm.generate_fast_explanation("Prompt")

    assert response["content"] == structured_response
    assert captured["payload"]["options"]["num_predict"] == 240


def test_default_provider_uses_ollama_without_openai(monkeypatch):
    monkeypatch.setattr(llm, "LLM_PROVIDER", "ollama")
    monkeypatch.setattr(llm, "_generate_with_ollama", lambda prompt, model=None: {"ok": True})
    monkeypatch.setattr(
        llm,
        "_generate_with_openai",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("OpenAI called")),
    )

    assert llm.generate_local_response("Grounded prompt") == {"ok": True}


def test_openai_response_is_normalized_without_real_request(monkeypatch):
    captured: dict[str, object] = {}

    class FakeResponses:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(output_text="Grounded qualitative explanation")

    monkeypatch.setattr(llm, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(llm, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(config, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        llm,
        "_get_openai_client",
        lambda: SimpleNamespace(responses=FakeResponses()),
    )
    monkeypatch.setattr(
        llm,
        "_generate_with_ollama",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Ollama called")),
    )

    response = llm.generate_local_response("Use only supplied numbers.")

    assert response["ok"] is True
    assert response["content"] == "Grounded qualitative explanation"
    assert response["provider"] == "openai"
    assert captured["model"] == llm.OPENAI_TEXT_MODEL
    assert "Use only" in captured["input"]


def test_openai_embedding_returns_chroma_compatible_vector(monkeypatch):
    class FakeEmbeddings:
        def create(self, **kwargs):
            assert kwargs["model"] == llm.OPENAI_EMBED_MODEL
            return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2])])

    monkeypatch.setattr(llm, "EMBEDDING_PROVIDER", "openai")
    monkeypatch.setattr(llm, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(config, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        llm,
        "_get_openai_client",
        lambda: SimpleNamespace(embeddings=FakeEmbeddings()),
    )
    monkeypatch.setattr(
        llm,
        "_embed_with_ollama",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Ollama called")),
    )

    assert llm.get_local_embedding("compact context") == [0.1, 0.2]


def test_openai_without_key_returns_safe_controlled_error(monkeypatch):
    monkeypatch.setattr(llm, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(llm, "OPENAI_API_KEY", "")
    monkeypatch.setattr(config, "OPENAI_API_KEY", "")

    response = llm.generate_local_response("Grounded prompt")

    assert response["ok"] is False
    assert response["error"] == "OpenAI provider is selected but OPENAI_API_KEY is not configured."
