"""Deterministic chart inference and generation for safe Ask-tab visualizations."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from src.analytics import format_aed_compact

CHART_INTENT_KEYWORDS = (
    "chart",
    "graph",
    "plot",
    "visualize",
    "visualise",
)

VALUE_PHRASE_TO_COLUMN = {
    "revenue": "Amount",
    "amount": "Amount",
    "deal value": "Amount",
    "pipeline value": "Amount",
    "opportunity value": "Amount",
    "pipeline": "Amount",
    "estimated value": "Estimated_Value",
    "lead value": "Estimated_Value",
    "deal size": "Amount",
}

CATEGORY_PHRASES = {
    "region": {"leads": "Region", "deals": "Region"},
    "location": {"leads": "Region", "deals": "Region"},
    "market": {"leads": "Region", "deals": "Region"},
    "product category": {"leads": "Product_Interest", "deals": "Product_Category"},
    "product": {"leads": "Product_Interest", "deals": "Product_Category"},
    "source": {"leads": "Lead_Source", "deals": None},
    "lead source": {"leads": "Lead_Source", "deals": None},
    "sales rep": {"leads": "Sales_Rep", "deals": "Sales_Rep"},
    "owner": {"leads": "Sales_Rep", "deals": "Sales_Rep"},
    "salesperson": {"leads": "Sales_Rep", "deals": "Sales_Rep"},
    "stage": {"leads": None, "deals": "Deal_Stage"},
    "deal stage": {"leads": None, "deals": "Deal_Stage"},
    "forecast": {"leads": None, "deals": "Forecast_Category"},
    "forecast category": {"leads": None, "deals": "Forecast_Category"},
    "industry": {"leads": "Industry", "deals": None},
    "segment": {"leads": "Industry", "deals": None},
    "customer type": {"leads": "Customer_Type", "deals": None},
}

DATE_PHRASES = {
    "closing date": ("deals", "Closing_Date"),
    "close date": ("deals", "Closing_Date"),
    "month": ("deals", "Closing_Date"),
    "created date": ("leads", "Created_Date"),
    "lead date": ("leads", "Created_Date"),
}

AGGREGATION_KEYWORDS = {
    "mean": "mean",
    "average": "mean",
    "avg": "mean",
    "count": "count",
    "number of": "count",
    "how many": "count",
    "total": "sum",
    "sum": "sum",
}

SUPPORTED_DYNAMIC_EXAMPLES = [
    "Show revenue by region",
    "Create a chart of amount by product category",
    "Plot leads by source",
    "Show count of deals by forecast category",
    "Show pipeline by sales rep",
    "Show average deal size by region",
    "Show monthly pipeline by closing date",
]


def detect_chart_intent(question: str) -> bool:
    """Return True when the question likely asks for a visualization."""
    question_lower = question.lower().strip()
    if not question_lower:
        return False

    if any(keyword in question_lower for keyword in CHART_INTENT_KEYWORDS):
        return True

    if any(
        pattern in question_lower
        for pattern in (
            "revenue by ",
            "amount by ",
            "pipeline by ",
            "estimated value by ",
            "lead value by ",
            "average deal size by ",
            "avg deal size by ",
            "count of deals by ",
            "count of leads by ",
            "monthly pipeline by ",
        )
    ):
        return True

    if re.search(r"\b(show|compare|display)\b.+\b(by)\b", question_lower) and any(
        token in question_lower
        for token in ("revenue", "amount", "pipeline", "value", "average", "count", "monthly")
    ):
        return True

    return False


def get_supported_chart_fields(leads_df: pd.DataFrame, deals_df: pd.DataFrame) -> dict[str, list[str]]:
    """Return available schema fields that can be charted safely."""
    lead_fields = [
        column
        for column in [
            "Lead_Source",
            "Estimated_Value",
            "Sales_Rep",
            "Industry",
            "Customer_Type",
            "Product_Interest",
            "Region",
            "Created_Date",
        ]
        if column in leads_df.columns
    ]
    deal_fields = [
        column
        for column in [
            "Amount",
            "Deal_Stage",
            "Forecast_Category",
            "Sales_Rep",
            "Product_Category",
            "Region",
            "Closing_Date",
        ]
        if column in deals_df.columns
    ]
    return {"leads": lead_fields, "deals": deal_fields}


def _infer_dataset(question_lower: str) -> str:
    """Infer whether the question should use leads or deals."""
    if any(
        token in question_lower
        for token in (
            "leads",
            "lead source",
            "estimated value",
            "lead value",
            "created date",
            "industry",
            "customer type",
        )
    ):
        return "leads"

    if any(
        token in question_lower
        for token in (
            "deals",
            "revenue",
            "amount",
            "pipeline",
            "stage",
            "forecast",
            "closed won",
            "closing date",
            "close date",
            "deal size",
        )
    ):
        return "deals"

    if "source" in question_lower:
        return "leads"

    return "deals"


def _infer_aggregation(question_lower: str, y_column: str | None) -> str:
    """Infer aggregation from keywords with safe defaults."""
    for phrase, aggregation in AGGREGATION_KEYWORDS.items():
        if phrase in question_lower:
            return aggregation

    if y_column in {"Amount", "Estimated_Value"}:
        return "sum"
    return "count"


def _infer_category_column(question_lower: str, dataset: str) -> str | None:
    """Infer the grouping column from supported category phrases."""
    for phrase, mapping in CATEGORY_PHRASES.items():
        if phrase in question_lower:
            return mapping.get(dataset)

    if "by " in question_lower:
        by_target = question_lower.split("by ", 1)[1]
        for phrase, mapping in CATEGORY_PHRASES.items():
            if phrase in by_target:
                return mapping.get(dataset)

    return None


def _infer_value_column(question_lower: str, dataset: str) -> str | None:
    """Infer the numeric measure column for the chart."""
    for phrase, column in VALUE_PHRASE_TO_COLUMN.items():
        if phrase in question_lower:
            if dataset == "leads" and column == "Amount":
                continue
            return column

    if "leads by" in question_lower and dataset == "leads":
        return "Lead_ID"
    if "deals by" in question_lower and dataset == "deals":
        return "Deal_ID"
    return None


def _build_chart_title(
    aggregation: str,
    dataset: str,
    x_column: str,
    y_column: str,
    time_grain: str | None = None,
    filters: dict[str, Any] | None = None,
) -> str:
    """Create a readable title from a chart specification."""
    x_label = x_column.replace("_", " ")
    if time_grain == "month":
        x_label = f"Month of {x_label}"

    if aggregation == "count":
        base = "Lead Count" if dataset == "leads" else "Deal Count"
    elif aggregation == "mean":
        base = "Average Deal Size" if y_column == "Amount" else f"Average {y_column.replace('_', ' ')}"
    else:
        if y_column == "Amount":
            base = "Revenue" if not filters or filters.get("Forecast_Category") != "Pipeline" else "Pipeline"
        elif y_column == "Estimated_Value":
            base = "Estimated Lead Value"
        else:
            base = y_column.replace("_", " ")

    return f"{base} by {x_label}"


def infer_chart_spec(
    question: str,
    leads_df: pd.DataFrame,
    deals_df: pd.DataFrame,
) -> dict[str, Any] | None:
    """Infer a safe chart specification from a supported question."""
    question_lower = question.lower().strip()
    if not question_lower:
        return None

    dataset = _infer_dataset(question_lower)
    x_column = _infer_category_column(question_lower, dataset)
    time_grain = None
    filters: dict[str, Any] = {}

    if x_column is None:
        for phrase, (date_dataset, date_column) in DATE_PHRASES.items():
            if phrase in question_lower:
                dataset = date_dataset
                x_column = date_column
                time_grain = "month" if "month" in question_lower else None
                break

    y_column = _infer_value_column(question_lower, dataset)
    aggregation = _infer_aggregation(question_lower, y_column)
    chart_type = "line" if time_grain == "month" else "bar"

    if "pipeline" in question_lower and dataset == "deals":
        filters["Forecast_Category"] = "Pipeline"

    if aggregation == "count" and dataset == "leads":
        y_column = "Lead_ID"
    elif aggregation == "count":
        y_column = "Deal_ID"

    if dataset == "deals" and x_column == "Sales_Rep" and "Sales_Rep" not in deals_df.columns:
        if {"Lead_ID", "Sales_Rep"}.issubset(leads_df.columns):
            filters["join_sales_rep_from_leads"] = True

    if x_column is None or y_column is None:
        return None

    title = _build_chart_title(
        aggregation=aggregation,
        dataset=dataset,
        x_column=x_column,
        y_column=y_column,
        time_grain=time_grain,
        filters=filters or None,
    )
    return {
        "dataset": dataset,
        "x_column": x_column,
        "y_column": y_column,
        "aggregation": aggregation,
        "chart_type": chart_type,
        "title": title,
        "time_grain": time_grain,
        "filters": filters,
    }


def _prepare_dataset(
    chart_spec: dict[str, Any],
    leads_df: pd.DataFrame,
    deals_df: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare a DataFrame for deterministic aggregation."""
    dataset_name = chart_spec["dataset"]
    df = leads_df.copy() if dataset_name == "leads" else deals_df.copy()

    if df.empty:
        return df

    filters = chart_spec.get("filters", {})

    if dataset_name == "deals" and filters.get("join_sales_rep_from_leads"):
        if {"Lead_ID", "Sales_Rep"}.issubset(leads_df.columns) and "Lead_ID" in df.columns:
            df = df.merge(
                leads_df[["Lead_ID", "Sales_Rep"]],
                on="Lead_ID",
                how="left",
            )

    for column, value in filters.items():
        if column == "join_sales_rep_from_leads":
            continue
        if column not in df.columns:
            return pd.DataFrame()
        df = df[df[column] == value]

    return df


def _aggregate_chart_data(chart_spec: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the chart data according to the spec."""
    if df.empty:
        return pd.DataFrame()

    x_column = chart_spec["x_column"]
    y_column = chart_spec["y_column"]
    aggregation = chart_spec["aggregation"]
    time_grain = chart_spec.get("time_grain")

    required_columns = {x_column, y_column}
    if not required_columns.issubset(df.columns):
        return pd.DataFrame()

    working_df = df.copy()

    if time_grain == "month":
        date_series = pd.to_datetime(working_df[x_column], errors="coerce")
        working_df = working_df.assign(_month=date_series.dt.to_period("M").dt.to_timestamp())
        working_df = working_df.dropna(subset=["_month"])
        x_column = "_month"

    if aggregation == "count":
        summary = (
            working_df.groupby(x_column, as_index=False)
            .agg(Value=(y_column, "count"))
            .sort_values("Value", ascending=False if time_grain is None else True)
        )
    elif aggregation == "mean":
        summary = (
            working_df.groupby(x_column, as_index=False)
            .agg(Value=(y_column, "mean"))
            .sort_values("Value", ascending=False if time_grain is None else True)
        )
    else:
        summary = (
            working_df.groupby(x_column, as_index=False)
            .agg(Value=(y_column, "sum"))
            .sort_values("Value", ascending=False if time_grain is None else True)
        )

    if time_grain == "month" and "_month" in summary.columns:
        summary["_month"] = summary["_month"].dt.strftime("%Y-%m")

    return summary.rename(columns={x_column: chart_spec["x_column"]})


def _build_chart_explanation(chart_spec: dict[str, Any], chart_data: pd.DataFrame) -> tuple[str, str]:
    """Create a short explanation and recommended action from computed data."""
    if chart_data.empty:
        return (
            "No matching rows were available after applying the requested grouping.",
            "Try another supported field or upload data with the needed columns populated.",
        )

    top_row = chart_data.iloc[0]
    x_label = chart_spec["x_column"]
    aggregation = chart_spec["aggregation"]
    y_column = chart_spec["y_column"]
    is_currency = y_column in {"Amount", "Estimated_Value"}

    if aggregation == "count":
        top_value = f"{int(top_row['Value']):,}"
        explanation = (
            f"{top_row[x_label]} has the highest count at {top_value} "
            f"{'lead' if chart_spec['dataset'] == 'leads' else 'deal'}(s)."
        )
    else:
        top_value = format_aed_compact(top_row["Value"]) if is_currency else f"{top_row['Value']:,.1f}"
        explanation = f"{top_row[x_label]} is currently the leading segment at {top_value}."

    if chart_spec.get("time_grain") == "month":
        peak_value = (
            format_aed_compact(top_row["Value"])
            if is_currency
            else f"{top_row['Value']:,.1f}"
        )
        explanation = f"The monthly trend peaks in {top_row[x_label]} at {peak_value}."

    recommended_action = (
        f"Review what is driving performance in {top_row[x_label]} and decide whether to replicate it elsewhere."
    )
    return explanation, recommended_action


def build_unsupported_chart_response(
    leads_df: pd.DataFrame,
    deals_df: pd.DataFrame,
    reason: str | None = None,
) -> dict[str, Any]:
    """Return a helpful response for unsupported dynamic chart requests."""
    supported_fields = get_supported_chart_fields(leads_df, deals_df)
    return {
        "answer_type": "unsupported",
        "title": "Unsupported Chart Request",
        "answer": (
            reason
            or (
                "I can create safe dynamic charts only for supported InsightFlow schema fields. "
                "Arbitrary BI questions and code generation are intentionally disabled."
            )
        ),
        "recommended_action": (
            "Try one of these examples: "
            + "; ".join(SUPPORTED_DYNAMIC_EXAMPLES)
            + f". Available leads fields: {', '.join(supported_fields['leads']) or 'none'}."
            + f" Available deals fields: {', '.join(supported_fields['deals']) or 'none'}."
        ),
        "data": None,
    }


def generate_dynamic_chart(
    chart_spec: dict[str, Any],
    leads_df: pd.DataFrame,
    deals_df: pd.DataFrame,
) -> dict[str, Any]:
    """Generate a dynamic chart and computed explanation from a safe spec."""
    df = _prepare_dataset(chart_spec, leads_df, deals_df)
    required_columns = {chart_spec["x_column"], chart_spec["y_column"]}
    if not required_columns.issubset(df.columns):
        missing_columns = sorted(required_columns.difference(df.columns))
        return build_unsupported_chart_response(
            leads_df,
            deals_df,
            reason=(
                "This chart cannot be generated because the active dataset is missing: "
                + ", ".join(missing_columns)
                + "."
            ),
        )

    chart_data = _aggregate_chart_data(chart_spec, df)
    if chart_data.empty:
        return build_unsupported_chart_response(
            leads_df,
            deals_df,
            reason=(
                "This chart could not be generated because there are no matching rows "
                "after applying the requested grouping and filters."
            ),
        )

    explanation, recommended_action = _build_chart_explanation(chart_spec, chart_data)
    return {
        "answer_type": "dynamic_chart",
        "title": chart_spec["title"],
        "answer": "Showing a dynamically generated chart based on the active dataset.",
        "recommended_action": recommended_action,
        "data": chart_data,
        "chart_spec": chart_spec,
        "explanation": explanation,
    }
