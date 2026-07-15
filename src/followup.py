"""Controlled follow-up chat helpers for InsightFlow Ask-tab results."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src.query_engine import answer_business_question
from src.llm import (
    FOLLOWUP_GENERATION_OPTIONS,
    check_ollama_available,
    check_ollama_model_available,
    check_text_provider_available,
    generate_fast_explanation,
)
from src.config import LLM_PROVIDER, OLLAMA_TEXT_MODEL, llm_provider_identity

ANALYTICS_VERBS = (
    "calculate",
    "compare",
    "rank",
    "show",
    "create",
    "plot",
    "filter",
    "group",
    "break down",
    "list",
    "sort",
)
ANALYTICS_METRICS = (
    "total",
    "sum",
    "count",
    "average",
    "avg",
    "top",
    "bottom",
    "highest",
    "lowest",
    "best",
    "worst",
    "trend",
    "monthly",
    "yearly",
    "conversion",
    "win rate",
    "forecast quality",
    "sales rep",
    "salesman",
    "salesperson",
    "region",
    "product category",
    "pipeline",
    "deals",
    "amount",
    "margin",
)
NEW_CHART_PHRASES = (
    "show this as a chart",
    "show this in a chart",
    "plot this as",
    "plot this",
    "create a chart",
    "create chart",
    "make a chart",
    "build a chart",
    "graph this",
)
EXPLANATION_PHRASES = (
    "what does this chart tell us",
    "what does this chart actually tell us",
    "what does this mean",
    "explain this",
    "interpret this",
    "why does this matter",
    "is this good or bad",
    "what is driving this",
)
EXPLANATION_KEYWORDS = (
    "why",
    "explain",
    "meaning",
    "interpret",
    "tell us",
    "good or bad",
    "summary",
)
CHART_IMPROVEMENT_PHRASES = (
    "what changes can i make on this chart",
    "what are some changes i can make on this chart",
    "how can this chart be improved",
    "how should i present this chart",
    "make this visual better",
    "improve this chart",
    "improve this visual",
)
BUSINESS_RECOMMENDATION_PHRASES = (
    "what should management do",
    "what should sales focus on",
    "what are the risks",
    "what opportunities do we have",
    "what should we do next",
    "what should management focus on",
)
BUSINESS_RECOMMENDATION_KEYWORDS = (
    "management",
    "sales focus",
    "focus on",
    "risks",
    "risk",
    "opportunities",
    "next steps",
    "recommend",
    "action",
    "address",
    "reduce",
    "improve",
    "prioritize",
)
CLARIFICATION_PHRASES = (
    "what is forecast quality",
    "what does commit mean",
    "what does po received mean",
    "what does pipeline mean",
    "what does omitted mean",
)
CLARIFICATION_PREFIXES = (
    "what is ",
    "what does ",
    "define ",
    "meaning of ",
)
UNRELATED_PHRASES = (
    "tell me a joke",
    "who won the world cup",
    "write python code",
    "write code",
)
CHART_NOUNS = ("chart", "graph", "visual")
CONTEXTUAL_ACTION_MARKERS = (
    "how should",
    "how can",
    "what should",
    "what can",
    "next",
    "address",
    "reduce",
    "improve",
    "focus",
    "action",
)
QUALITATIVE_ACTION_PATTERNS = (
    r"\bhow\s+(?:can|should|do)\b",
    r"\bwhat\s+(?:can|should)\b.*\b(?:do|take|address|reduce|improve|focus)\b",
    r"\bwhat\s+.*\b(?:steps?|actions?|recommendations?)\b",
    r"\b(?:steps?|actions?)\s+(?:to|for)\b",
    r"\bwhy\s+(?:might|is|are|does|do)\b",
    r"\bwhat\s+does\s+(?:this|that|the)\b.*\bmean\b",
)
DOMAIN_TERMS = {
    "stage",
    "stages",
    "region",
    "source",
    "lead",
    "leads",
    "deal",
    "deals",
    "sales",
    "rep",
    "forecast",
    "product",
    "category",
    "pipeline",
    "revenue",
    "amount",
    "margin",
    "conversion",
    "win",
    "lost",
    "won",
    "risk",
}
TOKEN_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "can",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "of",
    "on",
    "or",
    "should",
    "that",
    "the",
    "this",
    "to",
    "us",
    "we",
    "what",
    "why",
    "with",
}

EXPLANATION_CACHE: dict[str, dict[str, Any]] = {}
FOLLOWUP_PROMPT_VERSION = "FOLLOWUP_ACTION_PROMPT_V2"


def _followup_debug_enabled() -> bool:
    """Return whether terminal-only follow-up tracing is enabled."""
    return os.getenv("INSIGHTFLOW_DEBUG_FOLLOWUP", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _followup_trace_context(last_answer_context: dict[str, Any] | None) -> dict[str, Any]:
    """Build a compact, non-tabular trace of the session context used for routing."""
    context = last_answer_context or {}
    preview_entities: list[str] = []
    for row in context.get("table_preview", []) or []:
        for value in row.values():
            if isinstance(value, str) and value not in preview_entities:
                preview_entities.append(value)

    return {
        "original_user_question": context.get("original_user_question"),
        "prior_answer_title": context.get("answer_title"),
        "prior_answer_source": context.get("answer_source"),
        "has_computed_answer": bool(context.get("computed_answer")),
        "has_chart_data": bool(context.get("chart_type") or context.get("table_preview")),
        "has_rag_context": bool(context.get("retrieved_rag_context")),
        "resolved_topic": context.get("business_object") or context.get("main_entity"),
        "extracted_entities": preview_entities[:5],
    }


def _emit_followup_debug_trace(trace: dict[str, Any]) -> None:
    """Print one structured terminal trace when follow-up debug is enabled."""
    if not _followup_debug_enabled():
        return
    print(
        "\n=== FOLLOW-UP DEBUG TRACE ===\n"
        + json.dumps(trace, default=str, indent=2, sort_keys=True)
        + "\n=== END FOLLOW-UP DEBUG TRACE ===",
        flush=True,
    )


def _is_action_or_interpretation_followup(question: str) -> bool:
    """Identify non-calculating business action and interpretation requests."""
    question_lower = question.lower().strip()
    return any(
        re.search(pattern, question_lower) for pattern in QUALITATIVE_ACTION_PATTERNS
    ) or any(
        marker in question_lower
        for marker in (
            "management",
            "next steps",
            "suitable steps",
            "recommend",
            "interpret",
            "meaning",
            "explain",
        )
    )


def _is_quantitative_ranking_followup(question: str) -> bool:
    """Keep explicit requests to identify a top or bottom entity deterministic."""
    return bool(
        re.search(
            r"\bwhich\s+(?:sales\s+rep|customer\s+segment|region|source|product\s+category)"
            r"\b.*\b(?:highest|lowest|best|worst|priority)\b",
            question.lower(),
        )
    )


def classify_followup_intent(
    question: str,
    last_answer_context: dict[str, Any] | None = None,
) -> str:
    """Classify follow-up intent without letting generic chart words hijack routing."""
    question_lower = question.lower().strip()
    if not question_lower:
        return "ambiguous"

    if any(phrase in question_lower for phrase in UNRELATED_PHRASES):
        return "unrelated"

    if _is_quantitative_ranking_followup(question_lower):
        return "analytics"

    # Wording that asks for action or interpretation takes precedence over
    # mentioned entities such as "deals" or "stage"; entities establish relevance,
    # not the calculation route.
    if _is_action_or_interpretation_followup(question_lower):
        return (
            "explanation"
            if any(keyword in question_lower for keyword in EXPLANATION_KEYWORDS)
            else "business_recommendation"
        )

    analytics_override_tokens = (
        "compare",
        "rank",
        "bottom",
        "top",
        "worst",
        "weakest",
        "underperforming",
        "give me",
        "show it by",
        "show this by",
        "filter to",
        "which sales reps",
        "which customer segment",
        "product categories",
    )
    if any(token in question_lower for token in analytics_override_tokens):
        return "analytics"

    if any(phrase in question_lower for phrase in CHART_IMPROVEMENT_PHRASES):
        return "chart_improvement"

    if any(phrase in question_lower for phrase in BUSINESS_RECOMMENDATION_PHRASES):
        return "business_recommendation"

    if any(phrase in question_lower for phrase in CLARIFICATION_PHRASES):
        return "clarification"

    if any(phrase in question_lower for phrase in EXPLANATION_PHRASES):
        return "explanation"

    if any(phrase in question_lower for phrase in NEW_CHART_PHRASES):
        return "analytics"

    if any(
        phrase in question_lower
        for phrase in ("total ", "sum ", "count ", "average ", "avg ", "total amount")
    ):
        return "analytics"

    if any(question_lower.startswith(prefix) for prefix in CLARIFICATION_PREFIXES):
        return "clarification"

    if any(keyword in question_lower for keyword in BUSINESS_RECOMMENDATION_KEYWORDS):
        return "business_recommendation"

    if any(keyword in question_lower for keyword in EXPLANATION_KEYWORDS):
        return "explanation"

    if any(marker in question_lower for marker in CONTEXTUAL_ACTION_MARKERS):
        return "business_recommendation"

    if any(verb in question_lower for verb in ANALYTICS_VERBS) and (
        any(metric in question_lower for metric in ANALYTICS_METRICS)
        or any(noun in question_lower for noun in CHART_NOUNS)
    ):
        return "analytics"

    if any(metric in question_lower for metric in ANALYTICS_METRICS):
        return "analytics"

    if question_lower in CHART_NOUNS:
        return "ambiguous"

    if "chart" in question_lower and ("improve" in question_lower or "present" in question_lower):
        return "chart_improvement"

    context_title = (last_answer_context or {}).get("answer_title", "").lower()
    if context_title and any(token in question_lower for token in ("this", "it", "that")):
        return "explanation"

    return "ambiguous"


def _tokenize(text: object) -> set[str]:
    """Return normalized meaningful tokens for a lightweight relevance score."""
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(text).lower())
        if len(token) > 1 and token not in TOKEN_STOP_WORDS
    }


def _has_deictic_reference(text: str) -> bool:
    tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
    if tokens.intersection({"this", "these", "it", "here", "there"}):
        return True
    return bool(re.search(r"\bthat\s+(?:result|chart|stage|source|region|category)\b", text.lower()))


def _context_text(last_answer_context: dict[str, Any]) -> str:
    """Collect only stored result context that may safely establish continuity."""
    context_values = [
        last_answer_context.get("original_user_question", ""),
        last_answer_context.get("answer_title", ""),
        last_answer_context.get("answer_source", ""),
        last_answer_context.get("computed_answer", ""),
        last_answer_context.get("recommended_action", ""),
        last_answer_context.get("metric_type", ""),
        last_answer_context.get("business_object", ""),
        last_answer_context.get("main_entity", ""),
        last_answer_context.get("x_axis", ""),
        last_answer_context.get("y_axis", ""),
    ]
    for observation in last_answer_context.get("key_observations", []) or []:
        context_values.append(observation)
    for row in last_answer_context.get("table_preview", []) or []:
        context_values.extend([*row.keys(), *row.values()])
    return " ".join(str(value) for value in context_values if value)


def score_followup_relevance(
    followup_question: str,
    last_answer_context: dict[str, Any] | None,
) -> int:
    """Score whether a follow-up has a deterministic link to the current result."""
    if not last_answer_context:
        return 0

    question_lower = followup_question.lower().strip()
    if any(phrase in question_lower for phrase in UNRELATED_PHRASES):
        return -3

    question_tokens = _tokenize(question_lower)
    context_tokens = _tokenize(_context_text(last_answer_context))
    overlap = question_tokens.intersection(context_tokens)
    domain_overlap = question_tokens.intersection(DOMAIN_TERMS).intersection(context_tokens)
    has_result = bool(
        last_answer_context.get("answer_title")
        and (
            last_answer_context.get("computed_answer")
            or last_answer_context.get("table_preview")
            or last_answer_context.get("key_observations")
        )
    )
    score = 0
    if _has_deictic_reference(question_lower) and has_result:
        score += 2
    if domain_overlap:
        score += 2
    elif overlap:
        score += 1
    if (
        ("closed lost" in question_lower or "lost deal" in question_lower)
        and ("stage" in context_tokens or "deal_stage" in context_tokens)
    ):
        score += 2
    if has_result and any(
        marker in question_lower
        for marker in ("management", "focus", "next", "address", "reduce", "improve", "action")
    ):
        score += 2
    return score


def is_grounded_qualitative_followup(
    followup_question: str,
    last_answer_context: dict[str, Any] | None,
) -> bool:
    """Allow Ollama only when a qualitative request is tied to stored result facts."""
    if not last_answer_context or score_followup_relevance(followup_question, last_answer_context) < 2:
        return False
    return bool(
        last_answer_context.get("computed_answer")
        or last_answer_context.get("table_preview")
        or last_answer_context.get("key_observations")
        or last_answer_context.get("retrieved_rag_context")
    )


def classify_followup_intent_bucket(intent: str) -> str:
    """Collapse UI-oriented intent labels into the three routing buckets."""
    if intent == "analytics":
        return "quantitative"
    if intent in {
        "explanation",
        "chart_improvement",
        "business_recommendation",
        "clarification",
    }:
        return "qualitative_contextual"
    return "unrelated"


def resolve_followup_entities_from_context(
    followup_question: str,
    last_answer_context: dict[str, Any],
) -> str:
    """Conservatively resolve only unambiguous stage references from current context."""
    question_lower = followup_question.lower()
    context_text = _context_text(last_answer_context).lower()
    answer_title = last_answer_context.get("answer_title", "the current result")
    has_stage_context = "stage" in context_text or "deal_stage" in context_text

    if has_stage_context and "lost deal" in question_lower and "closed lost" in context_text:
        return f"{followup_question} for the Closed Lost deal stage"
    if has_stage_context and "this stage" in question_lower:
        return f"{followup_question} about {answer_title}"
    if _has_deictic_reference(question_lower):
        return f"{followup_question} about {answer_title}"
    return followup_question


def build_followup_suggestions_from_context(last_answer_context: dict[str, Any]) -> list[str]:
    """Return a few relevant next prompts without opening general chat."""
    context_text = _context_text(last_answer_context).lower()
    if "stage" in context_text:
        return [
            "Compare this stage by region",
            "Show only Closed Lost deals",
            "Why might this stage be weak?",
        ]
    if "lead source" in context_text or "source" in context_text:
        return [
            "Compare this source by sales rep",
            "What does this mean for lead quality?",
            "What should management focus on next?",
        ]
    if "region" in context_text:
        return [
            "Compare this by sales rep",
            "Show the trend by month",
            "What should management do here?",
        ]
    if "forecast" in context_text or "pipeline" in context_text:
        return [
            "Break this down by sales rep",
            "Show the trend by month",
            "How should we address this?",
        ]
    return [
        "What does this mean?",
        "Compare this by region",
        "What should management focus on next?",
    ]


def _unsupported_followup_response(
    last_answer_context: dict[str, Any],
    title: str = "Follow-up Out of Scope",
) -> dict[str, Any]:
    suggestions = build_followup_suggestions_from_context(last_answer_context)
    return {
        "answer_type": "unsupported",
        "title": title,
        "answer": (
            "This follow-up is not clearly related to the current InsightFlow result. "
            "Follow-up chat only supports questions related to the current InsightFlow result."
        ),
        "recommended_action": "Try: " + "; ".join(suggestions) + ".",
        "answer_source": "unsupported",
    }


def _build_followup_grounding_context(last_answer_context: dict[str, Any]) -> dict[str, Any]:
    """Select compact, computed-only context for a useful follow-up explanation."""
    rag_context = last_answer_context.get("retrieved_rag_context") or []
    return {
        "original_user_question": last_answer_context.get("original_user_question"),
        "answer_title": last_answer_context.get("answer_title"),
        "answer_source": last_answer_context.get("answer_source"),
        "computed_answer": last_answer_context.get("computed_answer"),
        "result_topic": last_answer_context.get("business_object"),
        "primary_entity": last_answer_context.get("main_entity"),
        "primary_value": last_answer_context.get("main_value"),
        "chart_or_table": {
            "chart_type": last_answer_context.get("chart_type"),
            "x_axis": last_answer_context.get("x_axis"),
            "y_axis": last_answer_context.get("y_axis"),
            "categories_and_values": last_answer_context.get("table_preview", [])[:5],
        },
        "computed_observations": last_answer_context.get("key_observations", [])[:5],
        "existing_recommended_action": last_answer_context.get("recommended_action"),
        "retrieved_rag_context": rag_context[:3],
    }


def build_followup_prompt(
    followup_question: str,
    last_answer_context: dict[str, Any],
    intent: str | None = None,
) -> str:
    """Build a strict, intent-aware prompt for grounded follow-up explanations."""
    serialized_context = json.dumps(
        _build_followup_grounding_context(last_answer_context),
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
        sort_keys=True,
    )
    resolved_intent = intent or classify_followup_intent(
        followup_question,
        last_answer_context,
    )
    if resolved_intent == "business_recommendation":
        response_format = (
            "Use exactly these four Markdown headings, with one or two concise sentences under each:\n"
            "**What this indicates**\n"
            "**Likely causes to investigate**\n"
            "**Practical next steps**\n"
            "**Recommended follow-up analysis**\n"
        )
    elif resolved_intent in {"explanation", "clarification"}:
        response_format = (
            "Use exactly these three Markdown headings, with one or two concise sentences under each:\n"
            "**What this means**\n"
            "**Why it matters**\n"
            "**Recommended next step**\n"
        )
    elif resolved_intent == "chart_improvement":
        response_format = (
            "Use exactly these three Markdown headings, with one or two concise sentences under each:\n"
            "**What the visual should emphasize**\n"
            "**Practical presentation improvements**\n"
            "**Recommended management takeaway**\n"
        )
    else:
        response_format = "Use short Markdown headings and give a concise, practical answer.\n"

    return (
        f"{FOLLOWUP_PROMPT_VERSION}\n"
        "You are a business explainer for InsightFlow AI.\n"
        "Pandas has already computed the result.\n"
        "Use only the provided deterministic_result_context.\n"
        "Do not calculate.\n"
        "Do not create new totals, averages, percentages, rankings, comparisons, or trends.\n"
        "Do not infer values from charts.\n"
        "Do not invent numbers.\n"
        "Do not alter the supplied categories, values, or currency.\n"
        "Use AED only.\n"
        "If the user asks for a new metric/calculation/chart/ranking/comparison, say it must be routed to Pandas analytics.\n"
        "Do not merely restate the chart or table.\n"
        "Treat causes as hypotheses to investigate, never as facts that are not in the result.\n"
        "For practical next steps, name concrete operational actions such as capturing loss reasons, assigning an owner, checking qualification criteria, or running a win-loss review when relevant.\n"
        "Do not use filler such as 'review the sales process', 'review deal details', or 'improve the sales strategy' unless you also name the specific review, owner, criterion, or action required.\n"
        "When recommending further analysis, propose one focused deterministic Pandas drill-down using only dimensions, categories, or filters named in the deterministic result context; do not invent a new dimension such as industry, product type, or salesperson.\n"
        "Tie every interpretation and recommendation to the current deterministic result context.\n"
        "Keep the response concise but complete, typically 100 to 170 words.\n\n"
        f"RESPONSE FORMAT:\n{response_format}\n"
        f"FOLLOW-UP QUESTION:\n{followup_question}\n\n"
        f"DETERMINISTIC RESULT CONTEXT:\n{serialized_context}"
    )


def _to_python_value(value: Any) -> Any:
    """Convert Pandas/Numpy scalars into plain Python values safely."""
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def build_computed_context(
    result: dict[str, Any],
    user_question: str,
    leads_df: pd.DataFrame | None = None,
    deals_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Build a compact structured computed context for safe Ollama grounding."""
    answer_title = result.get("title", "Computed Result")
    answer_source = result.get("answer_source", "metric calculation")
    computed_answer = result.get("answer", "")
    recommended_action = result.get("recommended_action", "")
    metric_type = "general_revenue_insight"
    business_object = "revenue operations"
    main_entity = answer_title
    main_value = computed_answer
    time_period = None
    chart_type = None
    chart_title = answer_title
    x_axis = None
    y_axis = None
    key_observations: list[str] = []
    known_limitations: list[str] = []
    data_df = result.get("data") if isinstance(result.get("data"), pd.DataFrame) else None
    table_preview: list[dict[str, Any]] = []

    title_lower = answer_title.lower()
    question_lower = user_question.lower().strip()

    if "monthly pipeline trend" in title_lower or result.get("chart_key") == "monthly_pipeline_trend":
        metric_type = "monthly_pipeline_trend"
        business_object = "pipeline trend"
        main_entity = "closing month"
        chart_type = "line_chart"
        x_axis = "Closing Month"
        y_axis = "Total Amount (AED)"
        time_period = "monthly"
        key_observations.append("Monthly trend is grouped by Closing_Date month.")
        key_observations.append("Missing months are filled with AED 0.")
        if data_df is not None and not data_df.empty and {"Closing_Month", "Total_Amount"}.issubset(data_df.columns):
            table_preview = [
                {
                    "Closing_Month": _to_python_value(row["Closing_Month"]),
                    "Total_Amount": _to_python_value(row["Total_Amount"]),
                }
                for row in data_df[["Closing_Month", "Total_Amount"]].head(5).to_dict(orient="records")
            ]
            top_row = data_df.sort_values("Total_Amount", ascending=False).iloc[0]
            main_value = f"{top_row['Closing_Month']} at AED {_to_python_value(top_row['Total_Amount']):,.0f}"
            key_observations.append(f"Highest month is {top_row['Closing_Month']}.")

    elif "forecast quality" in title_lower or answer_source == "forecast quality analysis":
        metric_type = "forecast_quality"
        business_object = "forecast quality"
        main_entity = "product category"
        key_observations.append("Weak forecast = At Risk + Omitted + Pipeline.")
        key_observations.append("Strong forecast = Commit + Closed + Best Case.")
        if data_df is not None and not data_df.empty:
            safe_columns = [
                column
                for column in [
                    "Product_Category",
                    "Weak_Forecast_Amount",
                    "Strong_Forecast_Amount",
                    "Weak_Forecast_Ratio",
                ]
                if column in data_df.columns
            ]
            table_preview = [
                {key: _to_python_value(value) for key, value in row.items()}
                for row in data_df[safe_columns].head(5).to_dict(orient="records")
            ]

    elif "lead source" in title_lower and "conversion" in title_lower or answer_source == "lead conversion analysis":
        metric_type = "lead_source_conversion"
        business_object = "lead conversion"
        main_entity = "lead source"
        key_observations.append("Conversion uses Lead_ID match between leads and deals.")
        if data_df is not None and not data_df.empty:
            safe_columns = [
                column
                for column in [
                    "Lead_Source",
                    "Total_Leads",
                    "Converted_Leads",
                    "Conversion_Rate",
                    "Estimated_Value",
                ]
                if column in data_df.columns
            ]
            table_preview = [
                {key: _to_python_value(value) for key, value in row.items()}
                for row in data_df[safe_columns].head(5).to_dict(orient="records")
            ]
            key_observations.append("Lead sources are sorted by Estimated Value descending, then Conversion Rate ascending.")

    elif (
        "stage" in title_lower
        or (data_df is not None and "Deal_Stage" in data_df.columns)
    ):
        metric_type = "deal_stage_breakdown"
        business_object = "deal stage performance"
        main_entity = "deal stage"
        key_observations.append("Deal stages are grouped from the Deal_Stage field.")
        if data_df is not None and not data_df.empty:
            safe_columns = [
                column
                for column in ["Deal_Stage", "Deal_Count", "Total_Amount"]
                if column in data_df.columns
            ]
            table_preview = [
                {key: _to_python_value(value) for key, value in row.items()}
                for row in data_df[safe_columns].head(5).to_dict(orient="records")
            ]
    elif "sales rep" in title_lower:
        metric_type = "sales_rep_performance"
        business_object = "sales performance"
        main_entity = "sales rep"
        key_observations.append("Sales rep performance is grouped by Sales_Rep and summed by Amount.")
        if data_df is not None and not data_df.empty:
            safe_columns = [
                column
                for column in ["Sales_Rep", "Total_Deal_Amount", "Deal_Count"]
                if column in data_df.columns
            ]
            table_preview = [
                {key: _to_python_value(value) for key, value in row.items()}
                for row in data_df[safe_columns].head(5).to_dict(orient="records")
            ]

    elif "total pipeline value" in title_lower:
        metric_type = "total_pipeline_value"
        business_object = "pipeline value"
        main_entity = "pipeline"
        key_observations.append("Pipeline totals come from the Amount column.")
    elif data_df is not None and not data_df.empty:
        safe_columns = [
            column
            for column in data_df.columns
            if "id" not in column.lower() and "company" not in column.lower()
        ][:5]
        table_preview = [
            {key: _to_python_value(value) for key, value in row.items()}
            for row in data_df[safe_columns].head(5).to_dict(orient="records")
        ]
        if len(safe_columns) >= 2:
            x_axis = safe_columns[0].replace("_", " ")
            y_axis = safe_columns[1].replace("_", " ")

    if not table_preview:
        known_limitations.append("No supporting table preview is available for this result.")
    if result.get("answer_type") == "metric":
        chart_type = chart_type or "no_chart"
    elif result.get("answer_type") == "chart":
        chart_type = chart_type or result.get("chart_type", "chart")
    elif result.get("answer_type") == "dynamic_chart":
        chart_type = chart_type or result.get("chart_spec", {}).get("chart_type", "chart")

    chart_spec = result.get("chart_spec", {}) if isinstance(result.get("chart_spec"), dict) else {}
    if not x_axis and chart_spec.get("x_column"):
        x_axis = str(chart_spec["x_column"]).replace("_", " ")
    if not y_axis:
        if chart_spec.get("y_column"):
            y_axis = str(chart_spec["y_column"]).replace("_", " ")
        elif chart_type in {"line", "line_chart", "bar"}:
            y_axis = "Value"

    return {
        "original_user_question": user_question,
        "answer_title": answer_title,
        "answer_source": answer_source,
        "computed_answer": computed_answer,
        "recommended_action": recommended_action,
        "currency": "AED",
        "metric_type": metric_type,
        "business_object": business_object,
        "main_entity": main_entity,
        "main_value": main_value,
        "time_period": time_period,
        "chart_type": chart_type,
        "chart_title": chart_title,
        "x_axis": x_axis,
        "y_axis": y_axis,
        "table_preview": table_preview[:5],
        "key_observations": key_observations[:5],
        "known_limitations": known_limitations[:5],
    }


def build_last_answer_context(
    original_user_question: str,
    result: dict[str, Any],
    leads_df: pd.DataFrame | None = None,
    deals_df: pd.DataFrame | None = None,
    data_signature: tuple | None = None,
) -> dict[str, Any]:
    """Create the structured session-state context for follow-up chat."""
    context = build_computed_context(
        result=result,
        user_question=original_user_question,
        leads_df=leads_df,
        deals_df=deals_df,
    )

    if result.get("retrieved_results"):
        context["retrieved_rag_context"] = [
            {
                "text": item.get("text"),
                "metadata": item.get("metadata", {}),
            }
            for item in result["retrieved_results"][:5]
        ]
    else:
        context["retrieved_rag_context"] = []

    context["data_version"] = data_signature
    context["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    return context


def make_explanation_cache_key(
    followup_question: str,
    last_answer_context: dict[str, Any],
    intent: str | None = None,
) -> str:
    """Return a stable cache key for follow-up explanations."""
    payload = {
        "followup_prompt_version": FOLLOWUP_PROMPT_VERSION,
        "llm_provider": llm_provider_identity()[0],
        "llm_model": llm_provider_identity()[1],
        "followup_question": followup_question,
        "answer_title": last_answer_context.get("answer_title"),
        "computed_answer": last_answer_context.get("computed_answer"),
        "recommended_action": last_answer_context.get("recommended_action"),
        "metric_type": last_answer_context.get("metric_type"),
        "answer_source": last_answer_context.get("answer_source"),
        "intent": intent,
    }
    serialized = json.dumps(
        payload,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
        sort_keys=True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _rewrite_followup_for_analytics(
    followup_question: str,
    last_answer_context: dict[str, Any],
) -> str:
    """Resolve only conservative context carryover before Pandas routing."""
    return resolve_followup_entities_from_context(followup_question, last_answer_context)


def _explanation_provider_is_available() -> bool:
    """Keep local Ollama checks patchable while routing hosted mode correctly."""
    if LLM_PROVIDER == "ollama":
        return (
            check_ollama_available()
            and check_ollama_model_available(OLLAMA_TEXT_MODEL)
        )
    return check_text_provider_available()


def answer_followup_question(
    followup_question: str,
    last_answer_context: dict[str, Any] | None,
    leads_df: pd.DataFrame,
    deals_df: pd.DataFrame,
    active_business_context: dict[str, Any] | None = None,
    extra_tables: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """Answer a controlled follow-up using either Pandas routing or Ollama explanation."""
    trace: dict[str, Any] = {
        "followup_question": followup_question,
        **_followup_trace_context(last_answer_context),
        "relevance_score": None,
        "relevance_decision": "not_evaluated",
        "intent": None,
        "intent_bucket": None,
        "grounding_decision": "not_evaluated",
        "final_route": None,
        "ollama_called": False,
        "ollama_cache_hit": False,
        "pandas_called": False,
        "unsupported_reason": None,
    }

    def finish(
        response: dict[str, Any],
        *,
        final_route: str,
        unsupported_reason: str | None = None,
    ) -> dict[str, Any]:
        trace["final_route"] = final_route
        trace["unsupported_reason"] = unsupported_reason
        trace["response_answer_type"] = response.get("answer_type")
        trace["response_answer_source"] = response.get("answer_source")
        _emit_followup_debug_trace(trace)
        return response

    if not last_answer_context:
        return finish(
            {
                "answer_type": "unsupported",
                "title": "Follow-up Unavailable",
                "answer": "Please ask a main question first before using follow-up chat.",
                "recommended_action": "Ask a main InsightFlow question to create a result for follow-up discussion.",
                "answer_source": "unsupported",
            },
            final_route="unsupported",
            unsupported_reason="missing_last_answer_context",
        )

    try:
        intent = classify_followup_intent(followup_question, last_answer_context)
        intent_bucket = classify_followup_intent_bucket(intent)
        relevance_score = score_followup_relevance(followup_question, last_answer_context)
        if intent_bucket == "quantitative" and active_business_context:
            # The active semantic context is structured metadata from the current result,
            # so it can safely establish continuity for a deterministic Pandas follow-up.
            relevance_score = max(relevance_score, 2)

        trace["intent"] = intent
        trace["intent_bucket"] = intent_bucket
        trace["relevance_score"] = relevance_score
        trace["relevance_decision"] = "relevant" if relevance_score >= 2 else "not_relevant"

        if intent_bucket == "unrelated" or relevance_score < 2:
            if intent == "ambiguous":
                suggestions = build_followup_suggestions_from_context(last_answer_context)
                return finish(
                    {
                        "answer_type": "unsupported",
                        "title": "Clarify Follow-up",
                        "answer": (
                            "I can help with this result, but I need one direction: explanation, "
                            "business action, chart improvement, or a new calculation. This follow-up "
                            "is not clearly related to the current result yet."
                        ),
                        "recommended_action": "Try: " + "; ".join(suggestions) + ".",
                        "answer_source": "unsupported",
                    },
                    final_route="unsupported",
                    unsupported_reason="ambiguous_or_low_relevance",
                )
            return finish(
                _unsupported_followup_response(last_answer_context),
                final_route="unsupported",
                unsupported_reason=(
                    "classified_unrelated"
                    if intent_bucket == "unrelated"
                    else "low_relevance"
                ),
            )

        if intent_bucket == "quantitative":
            trace["grounding_decision"] = "not_required_deterministic_pandas"
            trace["pandas_called"] = True
            rewritten_question = _rewrite_followup_for_analytics(
                followup_question,
                last_answer_context,
            )
            business_question_kwargs: dict[str, Any] = {
                "leads_df": leads_df,
                "deals_df": deals_df,
                "active_context": active_business_context,
            }
            if extra_tables is not None:
                business_question_kwargs["extra_tables"] = extra_tables
            quantitative_result = answer_business_question(
                rewritten_question,
                **business_question_kwargs,
            )

            if "answer_source" not in quantitative_result:
                if quantitative_result["answer_type"] == "chart":
                    quantitative_result["answer_source"] = (
                        "time trend chart"
                        if quantitative_result.get("chart_key") == "monthly_pipeline_trend"
                        else "semantic BI calculation"
                    )
                elif quantitative_result["answer_type"] == "table":
                    quantitative_result["answer_source"] = "table aggregation"
                elif quantitative_result["answer_type"] == "metric":
                    quantitative_result["answer_source"] = quantitative_result.get(
                        "source",
                        "metric calculation",
                    )
                else:
                    quantitative_result["answer_source"] = "unsupported"

            if quantitative_result["answer_source"] == "forecast_quality_analysis":
                quantitative_result["answer_source"] = "forecast quality analysis"
            if quantitative_result["answer_source"] == "lead_conversion_analysis":
                quantitative_result["answer_source"] = "lead conversion analysis"
            quantitative_result["followup_intent"] = intent
            quantitative_result["followup_intent_bucket"] = intent_bucket
            quantitative_result["followup_relevance_score"] = relevance_score
            if (
                quantitative_result.get("answer_type") == "unsupported"
                and _is_action_or_interpretation_followup(followup_question)
                and is_grounded_qualitative_followup(
                    followup_question,
                    last_answer_context,
                )
            ):
                # A result entity can occasionally trigger the deterministic parser
                # despite action wording. Retry once as a grounded explanation only.
                trace["pandas_result_answer_type"] = quantitative_result.get("answer_type")
                trace["pandas_result_answer_source"] = quantitative_result.get("answer_source")
                trace["pandas_fallback_to_qualitative"] = True
                trace["intent"] = "business_recommendation"
                trace["intent_bucket"] = "qualitative_contextual"
                trace["grounding_decision"] = "grounded_after_pandas_unsupported"
                intent = "business_recommendation"
                intent_bucket = "qualitative_contextual"
            else:
                return finish(quantitative_result, final_route="pandas")

        if intent_bucket == "qualitative_contextual":
            if not is_grounded_qualitative_followup(followup_question, last_answer_context):
                trace["grounding_decision"] = "not_grounded"
                return finish(
                    _unsupported_followup_response(last_answer_context),
                    final_route="unsupported",
                    unsupported_reason="qualitative_followup_not_grounded",
                )
            trace["grounding_decision"] = "grounded"
            if not _explanation_provider_is_available():
                return finish(
                    {
                        "answer_type": "unsupported",
                        "title": "Follow-up Explanation Unavailable",
                        "answer": (
                            "The configured explanation provider is unavailable. "
                            "Computed Pandas answers still work."
                        ),
                        "recommended_action": "Ask a calculation or chart follow-up, or configure the explanation provider to enable qualitative follow-ups.",
                        "answer_source": "unsupported",
                    },
                    final_route="unsupported",
                    unsupported_reason="ollama_or_model_unavailable",
                )

            prompt = build_followup_prompt(
                followup_question,
                last_answer_context,
                intent=intent,
            )
            trace["prompt_marker"] = FOLLOWUP_PROMPT_VERSION
            trace["prompt_marker_present"] = FOLLOWUP_PROMPT_VERSION in prompt
            trace["prompt_preview"] = prompt[:2400]
            trace["grounding_context_keys"] = list(
                _build_followup_grounding_context(last_answer_context).keys()
            )
            trace["generation_options"] = dict(FOLLOWUP_GENERATION_OPTIONS)
            cache_key = make_explanation_cache_key(
                followup_question,
                last_answer_context,
                intent=intent,
            )
            if cache_key in EXPLANATION_CACHE:
                cached = dict(EXPLANATION_CACHE[cache_key])
                cached["from_cache"] = True
                trace["ollama_cache_hit"] = True
                cached_content = str(cached.get("answer", ""))
                trace["cached_response_length"] = len(cached_content)
                trace["cached_response_preview"] = cached_content[:800]
                return finish(cached, final_route="ollama")

            start_time = time.perf_counter()
            try:
                trace["ollama_called"] = True
                response = generate_fast_explanation(prompt)
            except Exception as error:
                trace["ollama_error"] = str(error)
                return finish(
                    {
                        "answer_type": "unsupported",
                        "title": "Follow-up Explanation Unavailable",
                        "answer": (
                            "Ollama is unavailable. Follow-up explanation is disabled, but computed Pandas answers still work."
                        ),
                        "recommended_action": "Ask a calculation or chart follow-up, or start Ollama to enable explanation follow-ups.",
                        "answer_source": "unsupported",
                    },
                    final_route="unsupported",
                    unsupported_reason="ollama_call_exception",
                )

            if not response["ok"] or not response["content"]:
                return finish(
                    {
                        "answer_type": "unsupported",
                        "title": "Follow-up Explanation Unavailable",
                        "answer": response["content"]
                        or "Ollama is unavailable. Follow-up explanation is disabled, but computed Pandas answers still work.",
                        "recommended_action": "Ask a calculation or chart follow-up, or start Ollama to enable explanation follow-ups.",
                        "answer_source": "unsupported",
                    },
                    final_route="unsupported",
                    unsupported_reason=f"ollama_response_not_ok:{response.get('error') or 'empty_content'}",
                )

            raw_response = str(response.get("raw_content", response["content"]))
            final_response = str(response["content"])
            trace["raw_response_length"] = len(raw_response)
            trace["raw_response_preview"] = raw_response[:1200]
            trace["post_processing"] = response.get("post_processing", "unknown")
            trace["final_response_length"] = len(final_response)
            trace["final_response_preview"] = final_response[:1200]
            result = {
                "answer_type": "metric",
                "title": "Follow-up Explanation",
                "answer": response["content"],
                "recommended_action": last_answer_context.get("recommended_action"),
                "answer_source": "Ollama follow-up explanation",
                "followup_intent": intent,
                "followup_intent_bucket": intent_bucket,
                "followup_relevance_score": relevance_score,
                "latency_seconds": round(time.perf_counter() - start_time, 2),
                "from_cache": False,
            }
            EXPLANATION_CACHE[cache_key] = dict(result)
            return finish(result, final_route="ollama")

        return finish(
            _unsupported_followup_response(last_answer_context),
            final_route="unsupported",
            unsupported_reason="no_matching_route",
        )
    except Exception as error:
        trace["router_error"] = str(error)
        return finish(
            {
                "answer_type": "unsupported",
                "title": "Follow-up Unavailable",
                "answer": "The follow-up could not be processed safely right now.",
                "recommended_action": "Try a simpler follow-up question or ask a new main question.",
                "answer_source": "unsupported",
            },
            final_route="unsupported",
            unsupported_reason="unexpected_router_exception",
        )
