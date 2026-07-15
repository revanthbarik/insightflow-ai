"""Project paths and local-only configuration."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
SUPPORTED_PROVIDERS = {"ollama", "openai"}

DATA_DIR = PROJECT_ROOT / "data"
LEADS_CSV = DATA_DIR / "leads.csv"
DEALS_CSV = DATA_DIR / "deals.csv"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHARTS_DIR = OUTPUTS_DIR / "charts"
CHROMA_DB_DIR = Path(
    os.getenv("CHROMA_PERSIST_DIR", str(PROJECT_ROOT / "chroma_db"))
).expanduser()
CHROMA_DIR = CHROMA_DB_DIR

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_TEXT_MODEL = os.getenv("OLLAMA_TEXT_MODEL", "llama3.1")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "20"))
OLLAMA_RETRY_ATTEMPTS = max(1, int(os.getenv("OLLAMA_RETRY_ATTEMPTS", "2")))
OLLAMA_RETRY_DELAY_SECONDS = max(
    0.0,
    float(os.getenv("OLLAMA_RETRY_DELAY_SECONDS", "0.5")),
)

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "ollama").strip().lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4o-mini")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
OPENAI_TIMEOUT_SECONDS = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "30"))
OPENAI_MAX_OUTPUT_TOKENS = max(1, int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "240")))
OPENAI_RETRY_ATTEMPTS = max(1, int(os.getenv("OPENAI_RETRY_ATTEMPTS", "2")))
OPENAI_RETRY_DELAY_SECONDS = max(
    0.0,
    float(os.getenv("OPENAI_RETRY_DELAY_SECONDS", "0.5")),
)
RAG_COLLECTION_NAME = os.getenv("RAG_COLLECTION_NAME", "insightflow_context")


def provider_configuration_error(provider: str) -> str | None:
    """Return a safe configuration error without exposing secrets."""
    if provider not in SUPPORTED_PROVIDERS:
        return (
            f"Unsupported provider '{provider}'. "
            f"Choose one of: {', '.join(sorted(SUPPORTED_PROVIDERS))}."
        )
    if provider == "openai" and not OPENAI_API_KEY:
        return "OpenAI provider is selected but OPENAI_API_KEY is not configured."
    return None


def llm_provider_identity() -> tuple[str, str]:
    """Return the active text provider and model for cache isolation."""
    return (
        LLM_PROVIDER,
        OPENAI_TEXT_MODEL if LLM_PROVIDER == "openai" else OLLAMA_TEXT_MODEL,
    )


def embedding_provider_identity() -> tuple[str, str]:
    """Return the active embedding provider and model for collection isolation."""
    return (
        EMBEDDING_PROVIDER,
        OPENAI_EMBED_MODEL if EMBEDDING_PROVIDER == "openai" else OLLAMA_EMBED_MODEL,
    )
