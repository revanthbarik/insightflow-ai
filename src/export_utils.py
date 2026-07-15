"""Small, in-memory helpers for exporting InsightFlow results."""

from __future__ import annotations

import re

import pandas as pd


def make_safe_filename(label: str, suffix: str = "csv") -> str:
    """Return a predictable download filename from a user-facing label."""
    slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return f"{slug or 'insightflow_export'}.{suffix.lstrip('.')}"


def dataframe_to_csv_bytes(dataframe: pd.DataFrame) -> bytes:
    """Serialize a dataframe without creating a temporary file."""
    return dataframe.to_csv(index=False).encode("utf-8")


def insights_to_dataframe(insights: list[dict]) -> pd.DataFrame:
    """Convert existing Auto Insight cards into an exportable table."""
    return pd.DataFrame(
        [
            {
                "title": insight.get("title", ""),
                "metric": insight.get("metric", ""),
                "explanation": insight.get("explanation", ""),
                "recommended_action": insight.get("recommended_action", ""),
            }
            for insight in insights
        ]
    )


def ask_result_to_dataframe(result: dict) -> pd.DataFrame:
    """Export the current structured Ask result or safely package a metric."""
    data = result.get("data")
    if isinstance(data, pd.DataFrame) and not data.empty:
        return data.copy()
    return pd.DataFrame(
        [
            {
                "answer_title": result.get("title", ""),
                "answer": result.get("answer", ""),
                "answer_source": result.get("answer_source", ""),
            }
        ]
    )
