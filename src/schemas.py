"""Shared result schemas for the InsightFlow semantic BI layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class AnswerResult:
    """Structured business-answer payload compatible with the Streamlit app."""

    answer_type: str
    title: str
    answer: str
    recommended_action: str
    data: pd.DataFrame | None = None
    answer_source: str = "semantic bi query engine"
    chart_key: str | None = None
    chart_type: str | None = None
    metric_type: str | None = None
    metrics: list[str] = field(default_factory=list)
    business_object: str | None = None
    currency: str = "AED"
    limitations: list[str] = field(default_factory=list)
    source: str | None = None
    x_axis: str | None = None
    y_axis: str | None = None
    chart_title: str | None = None
    table_preview: list[dict[str, Any]] = field(default_factory=list)
    key_observations: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    query_debug: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a dict with both legacy and semantic-friendly field names."""
        payload = {
            "answer_type": self.answer_type,
            "title": self.title,
            "answer": self.answer,
            "recommended_action": self.recommended_action,
            "data": self.data,
            "answer_source": self.answer_source,
            "currency": self.currency,
            "metric_type": self.metric_type,
            "metrics": self.metrics,
            "business_object": self.business_object,
            "limitations": self.limitations,
            "errors": self.errors,
            "answer_title": self.title,
            "computed_answer": self.answer,
        }
        if self.chart_key is not None:
            payload["chart_key"] = self.chart_key
        if self.chart_type is not None:
            payload["chart_type"] = self.chart_type
        if self.source is not None:
            payload["source"] = self.source
        if self.x_axis is not None:
            payload["x_axis"] = self.x_axis
        if self.y_axis is not None:
            payload["y_axis"] = self.y_axis
        if self.chart_title is not None:
            payload["chart_title"] = self.chart_title
        if self.table_preview:
            payload["table_preview"] = self.table_preview
        if self.key_observations:
            payload["key_observations"] = self.key_observations
        if self.query_debug is not None:
            payload["query_debug"] = self.query_debug
        return payload
