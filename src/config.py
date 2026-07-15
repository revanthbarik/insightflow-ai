"""Project paths and local-only configuration."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

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
RAG_COLLECTION_NAME = os.getenv("RAG_COLLECTION_NAME", "insightflow_context")
