"""Controlled follow-up chat helpers for InsightFlow Ask-tab results."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src.query_engine import answer_business_question
from src.llm import (
    check_ollama_available,
    check_ollama_model_available,
    generate_fast_explanation,
)
from src.config import OLLAMA_TEXT_MODEL

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

EXPLANATION_CACHE: dict[str, dict[str, Any]] = {}


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

    if any(question_lower.startswith(prefix) for prefix in CLARIFICATION_PREFIXES):
        return "clarification"

    if any(keyword in question_lower for keyword in BUSINESS_RECOMMENDATION_KEYWORDS):
        return "business_recommendation"

    if any(keyword in question_lower for keyword in EXPLANATION_KEYWORDS):
        return "explanation"

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


def build_followup_prompt(
    followup_question: str,
    last_answer_context: dict[str, Any],
) -> str:
    """Build a strict grounded prompt for follow-up explanations only."""
    serialized_context = json.dumps(
        last_answer_context,
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
        "Use AED only.\n"
        "If the user asks for a new metric/calculation/chart/ranking/comparison, say it must be routed to Pandas analytics.\n"
        "Keep the answer under 4 sentences.\n"
        "Give practical management next steps when useful.\n\n"
        "For chart improvement requests, give specific suggestions like sorting, clearer labels, AED value labels, percentage share, highlighting risks, or adding a management takeaway.\n\n"
        f"FOLLOW-UP QUESTION:\n{followup_question}\n\n"
        f"COMPUTED CONTEXT:\n{serialized_context}"
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
) -> str:
    """Return a stable cache key for follow-up explanations."""
    payload = {
        "followup_question": followup_question,
        "answer_title": last_answer_context.get("answer_title"),
        "computed_answer": last_answer_context.get("computed_answer"),
        "recommended_action": last_answer_context.get("recommended_action"),
        "metric_type": last_answer_context.get("metric_type"),
        "answer_source": last_answer_context.get("answer_source"),
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
    """Lightly ground deictic follow-ups in the last answer title."""
    question_lower = followup_question.lower().strip()
    answer_title = last_answer_context.get("answer_title") or "the last result"
    if any(token in question_lower for token in ("this", "it", "that result", "this result")):
        return f"{followup_question} about {answer_title}"
    return followup_question


def answer_followup_question(
    followup_question: str,
    last_answer_context: dict[str, Any] | None,
    leads_df: pd.DataFrame,
    deals_df: pd.DataFrame,
    active_business_context: dict[str, Any] | None = None,
    extra_tables: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """Answer a controlled follow-up using either Pandas routing or Ollama explanation."""
    if not last_answer_context:
        return {
            "answer_type": "unsupported",
            "title": "Follow-up Unavailable",
            "answer": "Please ask a main question first before using follow-up chat.",
            "recommended_action": "Ask a main InsightFlow question to create a result for follow-up discussion.",
            "answer_source": "unsupported",
        }

    try:
        intent = classify_followup_intent(followup_question, last_answer_context)

        if intent == "analytics":
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
            return quantitative_result

        if intent == "unrelated":
            return {
                "answer_type": "unsupported",
                "title": "Follow-up Out of Scope",
                "answer": (
                    "Follow-up chat only supports questions related to the current InsightFlow result."
                ),
                "recommended_action": (
                    "Ask for an explanation, business action, chart improvement, or a new calculation based on the current result."
                ),
                "answer_source": "unsupported",
            }

        if intent == "ambiguous":
            return {
                "answer_type": "unsupported",
                "title": "Clarify Follow-up",
                "answer": (
                    "I can help with this result, but I need one direction: explanation, business action, chart improvement, or a new calculation."
                ),
                "recommended_action": (
                    "Try: 'Explain this', 'What should management do?', 'How can this chart be improved?', or 'Create a chart by region'."
                ),
                "answer_source": "unsupported",
            }

        if intent in {
            "explanation",
            "chart_improvement",
            "business_recommendation",
            "clarification",
        }:
            if not (
                check_ollama_available()
                and check_ollama_model_available(OLLAMA_TEXT_MODEL)
            ):
                return {
                    "answer_type": "unsupported",
                    "title": "Follow-up Explanation Unavailable",
                    "answer": (
                        "Ollama is unavailable. Follow-up explanation is disabled, but computed Pandas answers still work."
                    ),
                    "recommended_action": "Ask a calculation or chart follow-up, or start Ollama to enable explanation follow-ups.",
                    "answer_source": "unsupported",
                }

            prompt = build_followup_prompt(followup_question, last_answer_context)
            cache_key = make_explanation_cache_key(followup_question, last_answer_context)
            if cache_key in EXPLANATION_CACHE:
                cached = dict(EXPLANATION_CACHE[cache_key])
                cached["from_cache"] = True
                return cached

            start_time = time.perf_counter()
            try:
                response = generate_fast_explanation(prompt, model=OLLAMA_TEXT_MODEL)
            except Exception:
                return {
                    "answer_type": "unsupported",
                    "title": "Follow-up Explanation Unavailable",
                    "answer": (
                        "Ollama is unavailable. Follow-up explanation is disabled, but computed Pandas answers still work."
                    ),
                    "recommended_action": "Ask a calculation or chart follow-up, or start Ollama to enable explanation follow-ups.",
                    "answer_source": "unsupported",
                }

            if not response["ok"] or not response["content"]:
                return {
                    "answer_type": "unsupported",
                    "title": "Follow-up Explanation Unavailable",
                    "answer": response["content"]
                    or "Ollama is unavailable. Follow-up explanation is disabled, but computed Pandas answers still work.",
                    "recommended_action": "Ask a calculation or chart follow-up, or start Ollama to enable explanation follow-ups.",
                    "answer_source": "unsupported",
                }

            result = {
                "answer_type": "metric",
                "title": "Follow-up Explanation",
                "answer": response["content"],
                "recommended_action": last_answer_context.get("recommended_action"),
                "answer_source": "Ollama follow-up explanation",
                "followup_intent": intent,
                "latency_seconds": round(time.perf_counter() - start_time, 2),
                "from_cache": False,
            }
            EXPLANATION_CACHE[cache_key] = dict(result)
            return result

        return {
            "answer_type": "unsupported",
            "title": "Follow-up Unsupported",
            "answer": (
                "Please ask an explanation, business action, chart improvement, or new calculation follow-up about the current result."
            ),
            "recommended_action": "Try asking why the result matters, how to improve the chart, or ask for a new chart or metric.",
            "answer_source": "unsupported",
        }
    except Exception:
        return {
            "answer_type": "unsupported",
            "title": "Follow-up Unavailable",
            "answer": "The follow-up could not be processed safely right now.",
            "recommended_action": "Try a simpler follow-up question or ask a new main question.",
            "answer_source": "unsupported",
        }
