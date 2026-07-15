"""Local Ollama wrappers for private explanation and embedding tasks."""

from __future__ import annotations

import json
import time
from typing import Any

import requests

from src.config import (
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_MODEL,
    OLLAMA_RETRY_ATTEMPTS,
    OLLAMA_RETRY_DELAY_SECONDS,
    OLLAMA_TEXT_MODEL,
    OLLAMA_TIMEOUT_SECONDS,
)

SAFETY_INSTRUCTIONS = (
    "You are InsightFlow AI, a private revenue operations assistant.\n"
    "Use only the provided local context.\n"
    "Do not calculate new numbers.\n"
    "Do not invent numbers.\n"
    "Do not invent metrics.\n"
    "Do not use outside knowledge.\n"
    "Do not mention data that is not in the context.\n"
    "If the context is insufficient, say what information is missing.\n"
    "Keep the answer concise, clear, and business-friendly.\n"
    "When useful, provide recommended actions.\n"
    "All numeric calculations have already been computed by Pandas."
)
FOLLOWUP_GENERATION_OPTIONS = {
    "temperature": 0.1,
    "num_predict": 240,
    "top_p": 0.8,
    "num_ctx": 2048,
}


def _safe_json_request(
    method: str,
    endpoint: str,
    payload: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Call the Ollama HTTP API safely and return JSON plus an optional error."""
    url = f"{OLLAMA_BASE_URL.rstrip('/')}{endpoint}"
    last_error: requests.RequestException | None = None
    for attempt in range(OLLAMA_RETRY_ATTEMPTS):
        try:
            response = requests.request(
                method=method,
                url=url,
                json=payload,
                timeout=OLLAMA_TIMEOUT_SECONDS,
            )
            break
        except requests.RequestException as error:
            last_error = error
            if attempt < OLLAMA_RETRY_ATTEMPTS - 1:
                time.sleep(OLLAMA_RETRY_DELAY_SECONDS)
    else:
        return None, f"Ollama request failed after {OLLAMA_RETRY_ATTEMPTS} attempt(s): {last_error}"

    if response.status_code >= 400:
        try:
            error_payload = response.json()
            detail = error_payload.get("error", response.text)
        except ValueError:
            detail = response.text
        return None, f"Ollama returned HTTP {response.status_code}: {detail}"

    try:
        return response.json(), None
    except ValueError as error:
        return None, f"Ollama returned invalid JSON: {error}"


def check_ollama_available() -> bool:
    """Return True when the local Ollama server is reachable."""
    payload, error = _safe_json_request("GET", "/api/tags")
    return error is None and payload is not None


def check_ollama_model_available(model_name: str) -> bool:
    """Return True when the given Ollama model is available locally."""
    payload, error = _safe_json_request("GET", "/api/tags")
    if error or payload is None:
        return False

    for model in payload.get("models", []):
        name = model.get("name", "")
        if name == model_name or name.split(":")[0] == model_name.split(":")[0]:
            return True
    return False


def generate_local_response(prompt: str, model: str | None = None) -> dict[str, Any]:
    """Generate a local Ollama response safely."""
    selected_model = model or OLLAMA_TEXT_MODEL
    payload, error = _safe_json_request(
        "POST",
        "/api/generate",
        payload={
            "model": selected_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 120,
            },
        },
    )
    if error:
        return {"ok": False, "content": None, "error": error, "model": selected_model}

    return {
        "ok": True,
        "content": payload.get("response", "").strip(),
        "error": None,
        "model": selected_model,
    }


def get_local_embedding(text: str, model: str | None = None) -> list[float] | None:
    """Return an embedding from local Ollama, or None when unavailable."""
    if not text.strip():
        return None

    selected_model = model or OLLAMA_EMBED_MODEL
    payload, error = _safe_json_request(
        "POST",
        "/api/embeddings",
        payload={"model": selected_model, "prompt": text},
    )
    if error or payload is None:
        return None

    embedding = payload.get("embedding")
    if isinstance(embedding, list) and embedding:
        return embedding
    return None


def build_rag_prompt(
    user_question: str,
    retrieved_context: str,
    computed_context: str | None = None,
) -> str:
    """Build the safe RAG synthesis prompt."""
    computed_section = computed_context or "None"
    return (
        f"{SAFETY_INSTRUCTIONS}\n\n"
        f"User question:\n{user_question}\n\n"
        f"Retrieved local context:\n{retrieved_context}\n\n"
        f"Optional computed summaries:\n{computed_section}\n"
    )


def build_computed_prompt(
    user_question: str,
    computed_context: dict[str, Any],
) -> str:
    """Build the safe computed-answer synthesis prompt."""
    serialized_context = json.dumps(
        computed_context,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
        sort_keys=True,
    )
    return (
        "You are a business explainer for InsightFlow AI.\n"
        "Pandas has already computed the result.\n"
        "Use only the provided computed_context.\n"
        "Do not calculate.\n"
        "Do not create new totals, averages, percentages, rankings, comparisons, or trends.\n"
        "Do not infer values from charts.\n"
        "Do not invent numbers.\n"
        "Do not change currency.\n"
        "Use AED only.\n"
        "If the user asks for a new metric/calculation/chart/ranking/comparison, say it must be routed to Pandas analytics.\n"
        "Keep the answer under 4 sentences.\n"
        "Give practical management next steps when useful.\n\n"
        f"User question:\n{user_question}\n\n"
        f"computed_context:\n{serialized_context}\n"
    )


def synthesize_rag_answer(
    user_question: str,
    retrieved_context: str,
    computed_context: str | None = None,
) -> dict[str, Any]:
    """Synthesize a qualitative answer from retrieved local context only."""
    prompt = build_rag_prompt(user_question, retrieved_context, computed_context)
    return generate_local_response(prompt, model=OLLAMA_TEXT_MODEL)


def synthesize_computed_answer(
    user_question: str,
    computed_context: dict[str, Any],
) -> dict[str, Any]:
    """Synthesize an explanation for already-computed metrics only."""
    required_keys = {"title", "answer_source", "recommended_action"}
    if not required_keys.issubset(computed_context.keys()):
        return {
            "ok": True,
            "content": (
                "The computed result is available above, but there is not enough structured context for a safe explanation."
            ),
            "error": None,
            "model": OLLAMA_TEXT_MODEL,
        }
    prompt = build_computed_prompt(user_question, computed_context)
    return generate_local_response(prompt, model=OLLAMA_TEXT_MODEL)


def generate_fast_explanation(prompt: str, model: str | None = None) -> dict[str, Any]:
    """Generate a concise structured local explanation via Ollama."""
    selected_model = model or OLLAMA_TEXT_MODEL
    payload, error = _safe_json_request(
        "POST",
        "/api/generate",
        payload={
            "model": selected_model,
            "prompt": prompt,
            "stream": False,
            "options": FOLLOWUP_GENERATION_OPTIONS,
        },
    )
    if error:
        lowered_error = error.lower()
        if "timed out" in lowered_error or "timeout" in lowered_error:
            return {
                "ok": False,
                "content": "Ollama is taking too long. The computed Pandas answer is still available.",
                "error": error,
                "model": selected_model,
            }
        return {
            "ok": False,
            "content": "Ollama is unavailable. Follow-up explanation is disabled, but computed Pandas answers still work.",
            "error": error,
            "model": selected_model,
        }

    raw_response_text = payload.get("response", "")
    response_text = raw_response_text.strip()
    return {
        "ok": True,
        "content": response_text,
        "raw_content": raw_response_text,
        "post_processing": "strip_only",
        "error": None,
        "model": selected_model,
    }


def explain_with_ollama(context: dict[str, Any], model: str | None = None) -> str | None:
    """Backward-compatible helper for optional computed-result explanations."""
    response = synthesize_computed_answer(
        user_question=str(context.get("title", "Explain the computed result")),
        computed_context=context,
    )
    if not response["ok"]:
        return None
    return response["content"] or None
