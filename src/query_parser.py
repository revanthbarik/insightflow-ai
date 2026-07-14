"""Natural-language parsing for InsightFlow semantic BI questions."""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.semantic_layer import DIMENSION_SYNONYM_TO_NAME, METRIC_SYNONYM_TO_NAME

FORECAST_QUALITY_PHRASES = (
    "forecast quality",
    "weakest forecast",
    "weak forecast",
    "at-risk forecast",
    "at risk forecast",
    "poor forecast",
    "weak pipeline",
    "highest at-risk",
    "highest at risk",
    "most at-risk",
    "most at risk",
    "weakest category",
)
LEAD_CONVERSION_PHRASES = (
    "low conversion",
    "poor conversion",
    "highest estimated value but low conversion",
    "high lead value but poor conversion",
    "expensive but weak opportunities",
    "weak opportunities",
)
TIME_TREND_PHRASES = (
    "monthly pipeline trend",
    "pipeline by month",
    "revenue trend by closing date",
    "monthly deal amount trend",
    "revenue trend by month",
    "trend based on closing date",
)
CHART_KEYWORDS = ("chart", "graph", "plot", "visualize", "visualise")

METRIC_KEYWORDS = {
    "revenue": (
        "revenue",
        "sales",
        "total revenue",
        "total sales",
        "amount",
        "deal value",
        "value",
    ),
    "average_deal_size": (
        "average deal size",
        "avg deal size",
        "average value",
        "avg value",
    ),
    "pipeline": ("pipeline", "pipeline value", "open pipeline"),
    "margin": ("margin", "expected margin"),
    "deal_count": ("count", "number of deals", "deal count"),
    "win_rate": ("win rate", "conversion rate"),
}

DIMENSION_HINTS = {
    "Customer_Type": ("customer type", "customer segment", "segment"),
    "Account_Name": ("account name", "customer name", "account", "customer"),
    "Product_Name": ("product name", "item name", "sku"),
}


def _make_parsed_query(
    question: str,
    normalized: str,
    *,
    intent: str,
    metric_name: str | None,
    metrics: list[str],
    dataset: str | None,
    dimension: str | None,
    dimensions: list[str],
    limit: int | None,
    sort_order: str | None,
    aggregation: str | None,
    aggregation_map: dict[str, str],
    wants_chart: bool,
    chart_type: str | None,
    time_grain: str | None,
    intent_type: str,
    filters: dict[str, str] | None = None,
    filter_pipeline_only: bool = False,
    percentage: bool = False,
    include_full_distribution: bool = False,
    comparison: bool = False,
    recommendation_requested: bool = False,
    requires_join: bool = False,
    join_key: str | None = None,
    inherited_from_context: bool = False,
) -> ParsedQuery:
    """Create ParsedQuery objects with keyword arguments to avoid positional drift."""
    return ParsedQuery(
        raw_question=question,
        normalized_question=normalized,
        intent=intent,
        metric_name=metric_name,
        metrics=metrics,
        dataset=dataset,
        dimension=dimension,
        dimensions=dimensions,
        limit=limit,
        sort_order=sort_order,
        aggregation=aggregation,
        aggregation_map=aggregation_map,
        wants_chart=wants_chart,
        chart_type=chart_type,
        time_grain=time_grain,
        intent_type=intent_type,
        filters=filters,
        filter_pipeline_only=filter_pipeline_only,
        percentage=percentage,
        include_full_distribution=include_full_distribution,
        comparison=comparison,
        recommendation_requested=recommendation_requested,
        requires_join=requires_join,
        join_key=join_key,
        inherited_from_context=inherited_from_context,
    )


@dataclass
class ParsedQuery:
    """Structured parsed business intent."""

    raw_question: str
    normalized_question: str
    intent: str
    metric_name: str | None
    metrics: list[str]
    dataset: str | None
    dimension: str | None
    dimensions: list[str]
    limit: int | None
    sort_order: str | None
    aggregation: str | None
    aggregation_map: dict[str, str]
    wants_chart: bool
    chart_type: str | None
    time_grain: str | None
    intent_type: str
    filters: dict[str, str] | None = None
    filter_pipeline_only: bool = False
    percentage: bool = False
    include_full_distribution: bool = False
    comparison: bool = False
    recommendation_requested: bool = False
    requires_join: bool = False
    join_key: str | None = None
    inherited_from_context: bool = False


def _match_longest_phrase(question_lower: str, mapping: dict[str, str]) -> str | None:
    matches = [phrase for phrase in mapping if phrase in question_lower]
    if not matches:
        return None
    return max(matches, key=len)


def _extract_metrics(normalized: str) -> list[str]:
    metrics: list[str] = []
    for metric_name, keywords in METRIC_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            metrics.append(metric_name)
    return metrics


def _extract_dimensions(normalized: str) -> list[str]:
    dimensions: list[str] = []
    for dimension_name, synonyms in DIMENSION_HINTS.items():
        if any(keyword in normalized for keyword in synonyms):
            dimensions.append(dimension_name)
    return dimensions


def parse_business_question(
    question: str,
    active_context: dict | None = None,
) -> ParsedQuery:
    """Parse a business question into a structured BI query."""
    normalized = " ".join(question.lower().strip().split())
    active_context = active_context or {}
    inherited_from_context = False

    limit = None
    for pattern in (
        r"\b(?:top|bottom)\s+(\d+)\b",
        r"\bgive me\s+(\d+)\b",
        r"\bshow\s+(\d+)\b",
        r"\bfirst\s+(\d+)\b",
    ):
        limit_match = re.search(pattern, normalized)
        if limit_match:
            limit = int(limit_match.group(1))
            break

    sort_order = None
    if any(token in normalized for token in ("top", "best", "highest", "strongest", "most", "leading")):
        sort_order = "desc"
    if any(token in normalized for token in ("bottom", "worst", "lowest", "weakest", "underperforming", "least")):
        sort_order = "asc"

    if any(phrase in normalized for phrase in LEAD_CONVERSION_PHRASES) and "source" in normalized:
        return _make_parsed_query(
            question,
            normalized,
            intent="analytics",
            metric_name="lead_source_conversion",
            metrics=["lead_source_conversion"],
            dataset="leads",
            dimension="Lead_Source",
            dimensions=["Lead_Source"],
            limit=1,
            sort_order="desc",
            aggregation="comparison",
            aggregation_map={"lead_source_conversion": "comparison"},
            wants_chart=False,
            chart_type=None,
            time_grain=None,
            intent_type="advanced",
        )

    if any(phrase in normalized for phrase in FORECAST_QUALITY_PHRASES) and any(
        token in normalized for token in ("product", "category")
    ):
        return _make_parsed_query(
            question,
            normalized,
            intent="analytics",
            metric_name="forecast_quality",
            metrics=["forecast_quality"],
            dataset="deals",
            dimension="Product_Category",
            dimensions=["Product_Category"],
            limit=1,
            sort_order="desc",
            aggregation="ratio",
            aggregation_map={"forecast_quality": "ratio"},
            wants_chart=False,
            chart_type=None,
            time_grain=None,
            intent_type="advanced",
        )

    if any(phrase in normalized for phrase in TIME_TREND_PHRASES) or (
        "trend" in normalized and ("month" in normalized or "closing date" in normalized or "close date" in normalized)
    ):
        metric_name = "monthly_pipeline_trend" if "pipeline" in normalized else "revenue"
        return _make_parsed_query(
            question,
            normalized,
            intent="analytics",
            metric_name=metric_name,
            metrics=[metric_name],
            dataset="deals",
            dimension="Closing_Date",
            dimensions=["Closing_Date"],
            limit=None,
            sort_order=None,
            aggregation="sum",
            aggregation_map={metric_name: "sum"},
            wants_chart=True,
            chart_type="line",
            time_grain="month",
            intent_type="time_trend",
            filter_pipeline_only="pipeline" in normalized,
        )

    metric_phrase = _match_longest_phrase(normalized, METRIC_SYNONYM_TO_NAME)
    metric_name = METRIC_SYNONYM_TO_NAME.get(metric_phrase) if metric_phrase else None
    dimension_phrase = _match_longest_phrase(normalized, DIMENSION_SYNONYM_TO_NAME)
    dimension = DIMENSION_SYNONYM_TO_NAME.get(dimension_phrase) if dimension_phrase else None
    metrics = _extract_metrics(normalized)
    dimensions = _extract_dimensions(normalized)

    if "average deal size" in normalized or "avg deal size" in normalized:
        metric_name = "average_deal_size"
    elif "win rate" in normalized:
        metric_name = "win_rate"
    elif "margin" in normalized:
        metric_name = "margin"
    elif "deal count" in normalized or "number of deals" in normalized:
        metric_name = "deal_count"
    elif "pipeline" in normalized:
        metric_name = "pipeline"
    elif any(token in normalized for token in ("revenue", "sales", "amount", "value")):
        metric_name = "revenue"

    if metric_name and metric_name not in metrics:
        metrics.insert(0, metric_name)

    if metric_name is None and (active_context.get("metric") or active_context.get("metrics")) and (
        dimension is not None
        or sort_order is not None
        or limit is not None
        or any(token in normalized for token in ("compare", "show it", "show this", "filter", "rank", "priority", "recommend"))
    ):
        metric_name = str(active_context.get("metric") or active_context["metrics"][0])
        inherited_from_context = True
        if metric_name not in metrics:
            metrics.insert(0, metric_name)

    if "best salesman" in normalized or "best salesperson" in normalized or "best sales rep" in normalized:
        metric_name = "pipeline"
        dimension = "Sales_Rep"
        limit = 1
        return _make_parsed_query(
            question,
            normalized,
            intent="analytics",
            metric_name=metric_name,
            metrics=[metric_name],
            dataset="deals",
            dimension=dimension,
            dimensions=[dimension],
            limit=limit,
            sort_order="desc",
            aggregation="sum",
            aggregation_map={metric_name: "sum"},
            wants_chart=False,
            chart_type=None,
            time_grain=None,
            intent_type="rank",
        )

    if dimension is None and active_context.get("dimension") and any(
        token in normalized for token in ("show it by", "compare this by", "now show", "rank", "filter", "show this as", "which")
    ):
        dimension = str(active_context["dimension"])
        inherited_from_context = True

    if "product name" in normalized or "item name" in normalized or "sku" in normalized:
        dimension = "Product_Name"
    elif "product category" in normalized or "products" in normalized or "product" in normalized:
        dimension = "Product_Category"
    elif (
        ("account" in normalized or "customer" in normalized)
        and "customer type" not in normalized
        and "customer segment" not in normalized
    ):
        dimension = "Account_Name"
    elif "regions" in normalized or "region" in normalized or "dubai" in normalized:
        if "by region" in normalized or "regions" in normalized or "region" in normalized:
            dimension = "Region"
    elif "sales rep" in normalized or "sales reps" in normalized or "owner" in normalized:
        dimension = "Sales_Rep"
    elif "customer type" in normalized or "customer segment" in normalized or "segment" in normalized:
        dimension = "Customer_Type"

    if dimension and dimension not in dimensions:
        dimensions.insert(0, dimension)

    if "highest revenue" in normalized or "most revenue" in normalized or "strongest pipeline" in normalized:
        if dimension is not None:
            limit = limit or 1

    explicit_chart = any(keyword in normalized for keyword in CHART_KEYWORDS)
    wants_chart = explicit_chart
    if " by " in normalized and metric_name in {"revenue", "pipeline", "average_deal_size", "deal_count", "margin", "win_rate"}:
        wants_chart = wants_chart or normalized.startswith(("show ", "revenue by", "pipeline by", "average deal size by", "deal count by", "margin by", "win rate by"))
    if metric_name == "deal_count" and dimension == "Deal_Stage" and not explicit_chart:
        wants_chart = False

    percentage = any(token in normalized for token in ("percentage", "percent", "share"))
    include_full_distribution = (
        "distribution" in normalized
        or "all regions" in normalized
        or "across all" in normalized
        or "of each" in normalized
    )
    comparison = any(token in normalized for token in ("compare", "comparison", "performance of each"))
    recommendation_requested = any(
        token in normalized for token in ("management priority", "management focus", "recommend", "recommendation", "priority")
    )

    if metric_name == "deal_count" and dimension == "Deal_Stage" and not wants_chart:
        intent_type = "table"
    elif metric_name in {"revenue", "pipeline", "average_deal_size", "deal_count", "margin", "win_rate"} and dimension is not None and wants_chart:
        intent_type = "grouped_chart"
    elif metric_name in {"revenue", "pipeline", "average_deal_size", "deal_count", "margin", "win_rate"} and dimension is not None:
        intent_type = "grouped_metric"
    elif wants_chart and metric_name is not None:
        intent_type = "chart"
    else:
        intent_type = "metric"

    if "which " in normalized or "who " in normalized:
        limit = limit or 1
        if metric_name in {"revenue", "pipeline", "average_deal_size", "deal_count", "margin", "win_rate"} and dimension is not None:
            intent_type = "grouped_metric"
            if wants_chart or include_full_distribution:
                intent_type = "grouped_chart"

    if "worst" in normalized and limit is None:
        limit = 3

    filters: dict[str, str] = {}
    if "filter to dubai" in normalized or normalized.endswith("to dubai"):
        filters["Region"] = "Dubai"
    elif "to dubai" in normalized and "filter" in normalized:
        filters["Region"] = "Dubai"

    if not filters and active_context.get("filters") and "filter" not in normalized:
        filters = dict(active_context["filters"])

    chart_type = "line" if "trend" in normalized or "month" in normalized else ("bar" if wants_chart else None)
    if "pie chart" in normalized:
        chart_type = "pie"
    time_grain = "month" if "month" in normalized or "monthly" in normalized else None
    filter_pipeline_only = metric_name == "pipeline" and dimension != "Forecast_Category"
    dataset = str(active_context.get("dataset", "deals")) if active_context.get("dataset") else ("leads" if metric_name == "lead_source_conversion" else "deals")
    requires_join = False
    join_key = None
    if "Customer_Type" in dimensions:
        dataset = "deals_joined_leads"
        requires_join = True
        join_key = "Lead_ID"
    aggregation = "sum"
    if metric_name == "average_deal_size":
        aggregation = "mean"
    elif metric_name == "deal_count":
        aggregation = "count"
    elif metric_name == "win_rate":
        aggregation = "ratio"
    elif metric_name == "forecast_quality":
        aggregation = "ratio"
    elif metric_name == "lead_source_conversion":
        aggregation = "comparison"
    aggregation_map = {metric: ("mean" if metric == "average_deal_size" else "count" if metric == "deal_count" else "ratio" if metric in {"win_rate", "forecast_quality"} else "comparison" if metric == "lead_source_conversion" else "sum") for metric in metrics}

    if not metrics and active_context.get("metrics"):
        metrics = list(active_context["metrics"])
        if metrics:
            metric_name = metrics[0]
            inherited_from_context = True

    if not dimensions and dimension:
        dimensions = [dimension]

    intent = "analytics" if metric_name or metrics else "unsupported"

    return _make_parsed_query(
        question,
        normalized,
        intent=intent,
        metric_name=metric_name,
        metrics=metrics,
        dataset=dataset,
        dimension=dimension,
        dimensions=dimensions,
        limit=limit,
        sort_order=sort_order,
        aggregation=aggregation,
        aggregation_map=aggregation_map,
        wants_chart=wants_chart,
        chart_type=chart_type,
        time_grain=time_grain,
        intent_type=intent_type,
        filters=filters or None,
        filter_pipeline_only=filter_pipeline_only,
        percentage=percentage,
        include_full_distribution=include_full_distribution,
        comparison=comparison,
        recommendation_requested=recommendation_requested,
        requires_join=requires_join,
        join_key=join_key,
        inherited_from_context=inherited_from_context,
    )
