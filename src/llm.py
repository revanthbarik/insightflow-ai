"""Provider-backed qualitative explanation and embedding helpers."""

from __future__ import annotations

import json
import time
from typing import Any

import requests

from src.config import (
    EMBEDDING_PROVIDER,
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_MODEL,
    OLLAMA_RETRY_ATTEMPTS,
    OLLAMA_RETRY_DELAY_SECONDS,
    OLLAMA_TEXT_MODEL,
    OLLAMA_TIMEOUT_SECONDS,
    OPENAI_API_KEY,
    OPENAI_EMBED_MODEL,
    OPENAI_MAX_OUTPUT_TOKENS,
    OPENAI_RETRY_ATTEMPTS,
    OPENAI_RETRY_DELAY_SECONDS,
    OPENAI_TEXT_MODEL,
    OPENAI_TIMEOUT_SECONDS,
    embedding_provider_identity,
    llm_provider_identity,
    provider_configuration_error,
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


def _get_openai_client():
    """Create the official OpenAI client only when OpenAI mode is selected."""
    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError("OpenAI SDK is not installed.") from error
    return OpenAI(
        api_key=OPENAI_API_KEY,
        timeout=OPENAI_TIMEOUT_SECONDS,
        max_retries=0,
    )


def _run_openai_with_retry(operation):
    """Retry transient OpenAI calls without exposing request details or secrets."""
    last_error: Exception | None = None
    for attempt in range(OPENAI_RETRY_ATTEMPTS):
        try:
            return operation(), None
        except Exception as error:  # SDK exception classes vary by installed version.
            last_error = error
            transient_error = type(error).__name__ in {
                "APIConnectionError",
                "APITimeoutError",
                "InternalServerError",
                "RateLimitError",
            }
            if transient_error and attempt < OPENAI_RETRY_ATTEMPTS - 1:
                time.sleep(OPENAI_RETRY_DELAY_SECONDS)
                continue
            break
    return None, f"OpenAI request failed: {type(last_error).__name__}"


def check_text_provider_available() -> bool:
    """Return whether the configured text provider can be used safely."""
    if provider_configuration_error(LLM_PROVIDER):
        return False
    if LLM_PROVIDER == "ollama":
        return check_ollama_available() and check_ollama_model_available(OLLAMA_TEXT_MODEL)
    return True


def check_embedding_provider_available() -> bool:
    """Return whether the configured embedding provider can be used safely."""
    if provider_configuration_error(EMBEDDING_PROVIDER):
        return False
    if EMBEDDING_PROVIDER == "ollama":
        return check_ollama_available() and check_ollama_model_available(OLLAMA_EMBED_MODEL)
    return True


def provider_status_message(provider: str) -> str | None:
    """Return a concise safe provider configuration message, if any."""
    return provider_configuration_error(provider)


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


def _generate_with_ollama(prompt: str, model: str | None = None) -> dict[str, Any]:
    """Generate text with the local Ollama API."""
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
        return {
            "ok": False,
            "content": None,
            "error": error,
            "model": selected_model,
            "provider": "ollama",
        }

    return {
        "ok": True,
        "content": payload.get("response", "").strip(),
        "error": None,
        "model": selected_model,
        "provider": "ollama",
    }


def _generate_with_openai(prompt: str, model: str | None = None) -> dict[str, Any]:
    """Generate a grounded qualitative response with OpenAI's Responses API."""
    configuration_error = provider_configuration_error("openai")
    selected_model = model or OPENAI_TEXT_MODEL
    if configuration_error:
        return {
            "ok": False,
            "content": None,
            "error": configuration_error,
            "model": selected_model,
            "provider": "openai",
        }

    try:
        client = _get_openai_client()
    except Exception as error:
        return {
            "ok": False,
            "content": None,
            "error": f"OpenAI client unavailable: {type(error).__name__}",
            "model": selected_model,
            "provider": "openai",
        }
    response, error = _run_openai_with_retry(
        lambda: client.responses.create(
            model=selected_model,
            input=prompt,
            max_output_tokens=OPENAI_MAX_OUTPUT_TOKENS,
        )
    )
    if error or response is None:
        return {
            "ok": False,
            "content": None,
            "error": error,
            "model": selected_model,
            "provider": "openai",
        }
    return {
        "ok": True,
        "content": str(getattr(response, "output_text", "")).strip(),
        "error": None,
        "model": selected_model,
        "provider": "openai",
    }


def generate_local_response(prompt: str, model: str | None = None) -> dict[str, Any]:
    """Generate a qualitative response through the configured text provider."""
    if LLM_PROVIDER == "openai":
        return _generate_with_openai(prompt, model=model)
    if LLM_PROVIDER == "ollama":
        return _generate_with_ollama(prompt, model=model)
    return {
        "ok": False,
        "content": None,
        "error": provider_configuration_error(LLM_PROVIDER),
        "model": None,
        "provider": LLM_PROVIDER,
    }


def _embed_with_ollama(text: str, model: str | None = None) -> list[float] | None:
    """Return an embedding from the local Ollama API."""
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


def _embed_with_openai(text: str, model: str | None = None) -> list[float] | None:
    """Return an embedding from OpenAI in Chroma-compatible list form."""
    if provider_configuration_error("openai"):
        return None
    selected_model = model or OPENAI_EMBED_MODEL
    try:
        client = _get_openai_client()
    except Exception:
        return None
    response, error = _run_openai_with_retry(
        lambda: client.embeddings.create(model=selected_model, input=text)
    )
    if error or response is None or not getattr(response, "data", None):
        return None
    embedding = getattr(response.data[0], "embedding", None)
    return embedding if isinstance(embedding, list) and embedding else None


def get_local_embedding(text: str, model: str | None = None) -> list[float] | None:
    """Return an embedding from the configured provider in list form."""
    if not text.strip():
        return None
    if EMBEDDING_PROVIDER == "openai":
        return _embed_with_openai(text, model=model)
    if EMBEDDING_PROVIDER == "ollama":
        return _embed_with_ollama(text, model=model)
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
    return generate_local_response(prompt)


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
            "model": llm_provider_identity()[1],
            "provider": llm_provider_identity()[0],
        }
    prompt = build_computed_prompt(user_question, computed_context)
    return generate_local_response(prompt)


def generate_fast_explanation(prompt: str, model: str | None = None) -> dict[str, Any]:
    """Generate a concise structured explanation through the active provider."""
    if LLM_PROVIDER == "openai":
        response = _generate_with_openai(prompt, model=model)
        if not response["ok"]:
            return {
                "ok": False,
                "content": "OpenAI explanation is unavailable. The computed Pandas answer is still available.",
                "error": response["error"],
                "model": response["model"],
                "provider": "openai",
            }
        content = str(response["content"])
        return {
            "ok": True,
            "content": content,
            "raw_content": content,
            "post_processing": "strip_only",
            "error": None,
            "model": response["model"],
            "provider": "openai",
        }
    if LLM_PROVIDER != "ollama":
        return {
            "ok": False,
            "content": "Configured explanation provider is unavailable. The computed Pandas answer is still available.",
            "error": provider_configuration_error(LLM_PROVIDER),
            "model": None,
            "provider": LLM_PROVIDER,
        }

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
                "provider": "ollama",
            }
        return {
            "ok": False,
            "content": "Ollama is unavailable. Follow-up explanation is disabled, but computed Pandas answers still work.",
            "error": error,
            "model": selected_model,
            "provider": "ollama",
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
        "provider": "ollama",
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
