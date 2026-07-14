"""Deterministic routing helpers for quantitative and qualitative Ask intents."""

from __future__ import annotations

from src.dynamic_charting import detect_chart_intent

QUALITATIVE_PHRASES = (
    "what does forecast quality mean",
    "explain forecast quality",
    "what does at-risk pipeline mean",
    "how should management interpret at-risk deals",
    "how should management interpret at-risk pipeline",
    "what should management focus on this week",
    "explain the sales pipeline health",
    "what does the dashboard tell us",
    "summarize the current pipeline",
    "summarise the current pipeline",
    "summarize this month's sales pipeline",
    "summarise this month's sales pipeline",
    "explain the forecast categories",
    "what does commit mean",
    "what does best case mean",
    "what does pipeline mean",
    "what does omitted mean",
    "what does po received mean",
    "what does closed won mean",
    "what does closed lost mean",
    "why is pipeline high but closed revenue low",
    "what risks should management review",
    "what should the sales team prioritize",
    "what are the main revenue risks",
    "how should i interpret the auto insights",
    "what does lead conversion mean",
    "what does win rate mean",
    "how should we use this report",
    "how should we use this dashboard",
)

QUANTITATIVE_BLOCKERS = (
    "what is the total pipeline value",
    "show revenue by region",
    "create a chart of deals by stage",
    "which product category has weakest forecast quality",
    "which product category has the weakest forecast quality",
    "show monthly pipeline trend by closing date",
    "show monthly pipeline trend based on closing date",
    "which lead source has high value but low conversion",
    "which lead source generates the highest estimated value but low conversion into deals",
    "what is the win rate",
    "what is the average deal size",
    "which sales rep has strongest pipeline",
    "how many leads",
    "how many deals",
)


def detect_qualitative_intent(question: str) -> bool:
    """Return True for qualitative local-context questions that should use RAG."""
    question_lower = question.lower().strip()
    if not question_lower:
        return False

    if detect_chart_intent(question_lower):
        return False

    if any(phrase in question_lower for phrase in QUANTITATIVE_BLOCKERS):
        return False

    return any(phrase in question_lower for phrase in QUALITATIVE_PHRASES) or (
        any(
            token in question_lower
            for token in (
                "explain",
                "interpret",
                "summarize",
                "summarise",
                "what does",
                "how should",
                "why is",
            )
        )
        and not any(
            token in question_lower
            for token in (
                "total",
                "chart",
                "graph",
                "plot",
                "revenue by",
                "amount by",
                "average deal size",
                "win rate",
            )
        )
    )
