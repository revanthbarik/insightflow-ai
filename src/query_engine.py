"""Unified semantic BI query engine for InsightFlow analytics questions."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.generic_analytics import (
    bottom_n,
    forecast_quality,
    group_average,
    group_count,
    group_sum,
    lead_source_conversion,
    time_trend,
    top_n,
    win_rate,
)
from src.query_parser import ParsedQuery, parse_business_question
from src.schemas import AnswerResult

UNSUPPORTED_ANSWER_MESSAGE = (
    "I can currently answer pipeline, forecast, lead source, sales rep, "
    "product category, region, stage, win rate, and chart questions."
)

SUPPORTED_QUESTION_EXAMPLES = (
    "What is the total pipeline value?",
    "Show me deals by stage",
    "Create a chart of deals by stage",
    "Which product category has the weakest forecast quality?",
    "Show monthly pipeline trend based on closing date.",
    "Which lead source generates the highest estimated value but low conversion into deals?",
    "Show revenue by region",
    "Average deal size by region",
)


def _format_aed_compact(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if pd.isna(numeric):
        return "N/A"
    absolute_value = abs(numeric)
    if absolute_value >= 1_000_000:
        return f"AED {numeric / 1_000_000:.2f}M"
    if absolute_value >= 1_000:
        return f"AED {numeric / 1_000:.2f}K"
    return f"AED {numeric:,.0f}"


def _available_field_summary(leads_df: pd.DataFrame, deals_df: pd.DataFrame) -> str:
    lead_fields = ", ".join(sorted(leads_df.columns.tolist())) or "none"
    deal_fields = ", ".join(sorted(deals_df.columns.tolist())) or "none"
    return f"Available Leads fields: {lead_fields}. Available Deals fields: {deal_fields}."


def _unsupported_answer(
    leads_df: pd.DataFrame,
    deals_df: pd.DataFrame,
    title: str = "Unsupported Question",
    message: str | None = None,
) -> dict[str, Any]:
    answer = message or (
        f"{UNSUPPORTED_ANSWER_MESSAGE} "
        f"Examples: {'; '.join(SUPPORTED_QUESTION_EXAMPLES)}. "
        f"{_available_field_summary(leads_df, deals_df)}"
    )
    return AnswerResult(
        answer_type="unsupported",
        title=title,
        answer=answer,
        recommended_action="Try one of the supported examples or ask using the available dataset fields.",
        answer_source="unsupported",
        currency="AED",
    ).to_dict()


def _unsupported_chart_answer(leads_df: pd.DataFrame, deals_df: pd.DataFrame) -> dict[str, Any]:
    return AnswerResult(
        answer_type="unsupported",
        title="Unsupported Chart Request",
        answer=(
            "I can currently create charts for leads by source, deals by stage, "
            "forecast summary, sales rep performance, product category revenue, "
            "region-wise pipeline, grouped BI metrics, and monthly trend charts when the needed columns exist. "
            "Examples: Create a chart of deals by stage; Show revenue by region; Show monthly pipeline trend based on closing date. "
            f"{_available_field_summary(leads_df, deals_df)}"
        ),
        recommended_action="Try a supported chart example using the available dataset fields.",
        answer_source="unsupported",
    ).to_dict()


def _missing_data_answer(
    title: str,
    missing_items: list[str],
    leads_df: pd.DataFrame,
    deals_df: pd.DataFrame,
) -> dict[str, Any]:
    return AnswerResult(
        answer_type="unsupported",
        title=title,
        answer=f"I cannot compute this because the required column/table is missing: {', '.join(missing_items)}.",
        recommended_action=f"Upload data with the required fields and try again. {_available_field_summary(leads_df, deals_df)}",
        answer_source="unsupported",
    ).to_dict()


def _join_dimension_from_leads(
    deals_df: pd.DataFrame,
    leads_df: pd.DataFrame,
    dimension: str,
) -> pd.DataFrame:
    if dimension in deals_df.columns:
        return deals_df.copy()
    if "Lead_ID" not in deals_df.columns or "Lead_ID" not in leads_df.columns or dimension not in leads_df.columns:
        return pd.DataFrame()
    return deals_df.merge(leads_df[["Lead_ID", dimension]], on="Lead_ID", how="left")


def _resolve_grouped_deals_dataset(
    deals_df: pd.DataFrame,
    leads_df: pd.DataFrame,
    dimension: str,
) -> pd.DataFrame:
    if dimension in deals_df.columns:
        return deals_df.copy()
    if dimension in {"Region", "Sales_Rep", "Lead_Source"}:
        return _join_dimension_from_leads(deals_df, leads_df, dimension)
    return pd.DataFrame()


def _build_chart_context(
    title: str,
    data: pd.DataFrame,
    x_axis: str,
    y_axis: str,
    chart_type: str,
) -> dict[str, Any]:
    preview = data.head(5).to_dict(orient="records") if isinstance(data, pd.DataFrame) else []
    key_observations: list[str] = []
    if isinstance(data, pd.DataFrame) and not data.empty and y_axis in data.columns:
        top_row = data.sort_values(y_axis, ascending=False).iloc[0]
        key_observations.append(f"Highest value is {top_row[x_axis]} at {_format_aed_compact(top_row[y_axis]) if 'Rate' not in y_axis and 'Count' not in y_axis else top_row[y_axis]}.")
    return {
        "chart_type": chart_type,
        "chart_title": title,
        "x_axis": x_axis,
        "y_axis": y_axis,
        "table_preview": preview,
        "key_observations": key_observations[:5],
        "currency": "AED",
    }


def _make_result(
    *,
    answer_type: str,
    title: str,
    answer: str,
    recommended_action: str,
    data: pd.DataFrame | None = None,
    answer_source: str,
    metric_type: str,
    metrics: list[str] | None = None,
    business_object: str,
    chart_key: str | None = None,
    chart_type: str | None = None,
    source: str | None = None,
    x_axis: str | None = None,
    y_axis: str | None = None,
    chart_title: str | None = None,
    key_observations: list[str] | None = None,
    query_debug: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = AnswerResult(
        answer_type=answer_type,
        title=title,
        answer=answer,
        recommended_action=recommended_action,
        data=data,
        answer_source=answer_source,
        chart_key=chart_key,
        chart_type=chart_type,
        metric_type=metric_type,
        metrics=metrics or ([metric_type] if metric_type else []),
        business_object=business_object,
        source=source,
        x_axis=x_axis,
        y_axis=y_axis,
        chart_title=chart_title,
        table_preview=data.head(5).to_dict(orient="records") if isinstance(data, pd.DataFrame) else [],
        key_observations=key_observations or [],
        query_debug=query_debug,
    ).to_dict()
    return result


def _active_context_from_parsed_result(
    parsed: ParsedQuery,
    result: dict[str, Any],
) -> dict[str, Any]:
    """Build reusable analytics context for follow-up business questions."""
    return {
        "metric": result.get("metric_type") or parsed.metric_name,
        "metrics": result.get("metrics") or parsed.metrics,
        "dataset": parsed.dataset,
        "dimension": parsed.dimension,
        "dimensions": parsed.dimensions,
        "aggregation": parsed.aggregation,
        "aggregation_map": parsed.aggregation_map,
        "filters": parsed.filters or {},
        "chart_type": result.get("chart_type") or parsed.chart_type,
        "answer_title": result.get("title"),
        "computed_answer": result.get("answer"),
    }


def _with_engine_metadata(
    result: dict[str, Any],
    parsed: ParsedQuery,
) -> dict[str, Any]:
    """Attach parsed-query metadata so app and follow-up can reuse the BI context."""
    result["parsed_query"] = {
        "metric": parsed.metric_name,
        "metrics": parsed.metrics,
        "dataset": parsed.dataset,
        "dimension": parsed.dimension,
        "dimensions": parsed.dimensions,
        "aggregation": parsed.aggregation,
        "aggregation_map": parsed.aggregation_map,
        "filters": parsed.filters or {},
        "limit": parsed.limit,
        "sort_order": parsed.sort_order,
        "chart_type": parsed.chart_type,
        "time_grain": parsed.time_grain,
        "percentage": parsed.percentage,
        "include_full_distribution": parsed.include_full_distribution,
        "comparison": parsed.comparison,
        "recommendation_requested": parsed.recommendation_requested,
        "requires_join": parsed.requires_join,
        "join_key": parsed.join_key,
        "inherited_from_context": parsed.inherited_from_context,
    }
    result["active_business_context"] = _active_context_from_parsed_result(parsed, result)
    return result


def _answer_forecast_quality(deals_df: pd.DataFrame, leads_df: pd.DataFrame) -> dict[str, Any]:
    required = {"Product_Category", "Forecast_Category", "Amount"}
    if not required.issubset(deals_df.columns):
        return _missing_data_answer("Forecast Quality by Product Category Unavailable", sorted(required.difference(deals_df.columns)), leads_df, deals_df)
    summary = forecast_quality(deals_df)
    if summary.empty:
        return _unsupported_answer(leads_df, deals_df, title="Forecast Quality by Product Category Unavailable", message="There is not enough deal data to evaluate forecast quality by product category.")
    weakest = summary.iloc[0]
    return _make_result(
        answer_type="metric",
        title="Weakest Forecast Quality by Product Category",
        answer=(
            f"{weakest['Product_Category']} has the weakest forecast quality: "
            f"{_format_aed_compact(weakest['Weak_Forecast_Amount'])} weak forecast value, "
            f"representing {weakest['Weak_Forecast_Ratio'] * 100:.1f}% of its total category pipeline."
        ),
        recommended_action="Review this category's At Risk, Omitted, and early Pipeline deals and move serious opportunities toward Commit or Closed.",
        data=summary,
        answer_source="forecast quality analysis",
        metric_type="forecast_quality",
        business_object="forecast quality",
        source="forecast_quality_analysis",
        key_observations=[
            "Weak forecast = At Risk + Omitted + Pipeline.",
            "Strong forecast = Commit + Closed + Best Case.",
        ],
    )


def _answer_lead_source_conversion(leads_df: pd.DataFrame, deals_df: pd.DataFrame) -> dict[str, Any]:
    lead_required = {"Lead_ID", "Lead_Source", "Estimated_Value"}
    deal_required = {"Lead_ID"}
    missing = sorted(lead_required.difference(leads_df.columns)) + sorted(deal_required.difference(deals_df.columns))
    if missing:
        return _missing_data_answer("Lead Source Conversion Analysis Unavailable", missing, leads_df, deals_df)
    summary = lead_source_conversion(leads_df, deals_df)
    if summary.empty:
        return _unsupported_answer(leads_df, deals_df, title="Lead Source Conversion Analysis Unavailable", message="There is not enough lead and deal data to compare value against conversion.")
    average_conversion_rate = summary["Conversion_Rate"].mean()
    below_average = summary[summary["Conversion_Rate"] < average_conversion_rate]
    target = below_average.iloc[0] if not below_average.empty else summary.iloc[0]
    return _make_result(
        answer_type="metric",
        title="High Value but Low Conversion Lead Source",
        answer=(
            f"{target['Lead_Source']} has {_format_aed_compact(target['Estimated_Value'])} estimated lead value "
            f"but a conversion rate of {target['Conversion_Rate'] * 100:.1f}%, "
            f"below the average conversion rate of {average_conversion_rate * 100:.1f}%."
        ),
        recommended_action="Review qualification quality for this lead source before increasing sales focus or campaign spend.",
        data=summary.head(5),
        answer_source="lead conversion analysis",
        metric_type="lead_source_conversion",
        business_object="lead conversion",
        source="lead_conversion_analysis",
        key_observations=[
            "Lead sources are ranked by Estimated Value descending, then Conversion Rate ascending.",
        ],
    )


def _answer_monthly_trend(parsed: ParsedQuery, leads_df: pd.DataFrame, deals_df: pd.DataFrame) -> dict[str, Any]:
    required = {"Closing_Date", "Amount"}
    if not required.issubset(deals_df.columns):
        return _missing_data_answer("Monthly Trend Unavailable", sorted(required.difference(deals_df.columns)), leads_df, deals_df)
    working_df = deals_df.copy()
    if parsed.filter_pipeline_only:
        if "Forecast_Category" not in working_df.columns:
            return _missing_data_answer("Monthly Trend Unavailable", ["Forecast_Category"], leads_df, deals_df)
        working_df = working_df[working_df["Forecast_Category"] == "Pipeline"]
    summary = time_trend(working_df, "Closing_Date", "Amount")
    if summary.empty:
        return _unsupported_answer(leads_df, deals_df, title="Monthly Trend Unavailable", message="There is not enough valid closing-date data to build the monthly trend.")
    title = "Monthly Pipeline Trend by Closing Date" if parsed.filter_pipeline_only else "Monthly Deal Amount Trend by Closing Date"
    answer = "This chart shows pipeline deal amount grouped by closing month." if parsed.filter_pipeline_only else "This chart shows total deal amount grouped by closing month."
    result = _make_result(
        answer_type="chart",
        title=title,
        answer=answer,
        recommended_action="Review months with high expected pipeline value and confirm deal readiness." if parsed.filter_pipeline_only else "Review months with high expected closing value and confirm deal readiness.",
        data=summary[["Closing_Month", "Total_Amount"]],
        answer_source="time trend chart",
        metric_type="monthly_pipeline_trend",
        business_object="pipeline trend" if parsed.filter_pipeline_only else "revenue trend",
        chart_key="monthly_pipeline_trend",
        chart_type="line",
        x_axis="Closing_Month",
        y_axis="Total_Amount",
        chart_title=title,
        key_observations=[
            "Monthly trend is grouped by Closing_Date month.",
            "Missing months are filled with AED 0.",
        ],
    )
    result.update(_build_chart_context(title, result["data"], "Closing_Month", "Total_Amount", "line"))
    return result


def _answer_total_pipeline(leads_df: pd.DataFrame, deals_df: pd.DataFrame) -> dict[str, Any]:
    required = {"Forecast_Category", "Amount"}
    if not required.issubset(deals_df.columns):
        return _missing_data_answer("Total Pipeline Value Unavailable", sorted(required.difference(deals_df.columns)), leads_df, deals_df)
    value = float(deals_df.loc[deals_df["Forecast_Category"] == "Pipeline", "Amount"].sum())
    return _make_result(
        answer_type="metric",
        title="Total Pipeline Value",
        answer=_format_aed_compact(value),
        recommended_action="Review open pipeline deals and confirm next steps." if value > 0 else "No pipeline-tagged deals are currently recorded.",
        answer_source="metric calculation",
        metric_type="total_pipeline_value",
        business_object="pipeline value",
    )


def _answer_total_leads(leads_df: pd.DataFrame) -> dict[str, Any]:
    return _make_result(
        answer_type="metric",
        title="Total Leads",
        answer=f"{len(leads_df):,}",
        recommended_action="Review lead volume alongside estimated lead value.",
        answer_source="metric calculation",
        metric_type="lead_count",
        business_object="lead volume",
    )


def _answer_total_deals(deals_df: pd.DataFrame) -> dict[str, Any]:
    return _make_result(
        answer_type="metric",
        title="Total Deals",
        answer=f"{len(deals_df):,}",
        recommended_action="Review deal volume alongside total deal amount.",
        answer_source="metric calculation",
        metric_type="deal_count",
        business_object="deal volume",
    )


def _answer_stage_table(deals_df: pd.DataFrame, leads_df: pd.DataFrame) -> dict[str, Any]:
    required = {"Deal_Stage", "Deal_ID", "Amount"}
    if not required.issubset(deals_df.columns):
        return _missing_data_answer("Deals by Stage Unavailable", sorted(required.difference(deals_df.columns)), leads_df, deals_df)
    summary = deals_df.groupby("Deal_Stage", as_index=False).agg(
        Deal_Count=("Deal_ID", "count"),
        Total_Amount=("Amount", "sum"),
    ).sort_values("Total_Amount", ascending=False)
    return _make_result(
        answer_type="table",
        title="Deals by Stage",
        answer="Deals grouped by stage with counts and total amount.",
        data=summary,
        recommended_action="Focus on stages with the highest total amount.",
        answer_source="table aggregation",
        metric_type="deal_count",
        business_object="deal stage",
    )


def _answer_leads_by_source(leads_df: pd.DataFrame, deals_df: pd.DataFrame) -> dict[str, Any]:
    required = {"Lead_Source", "Lead_ID", "Estimated_Value"}
    if not required.issubset(leads_df.columns):
        return _missing_data_answer("Leads by Source Unavailable", sorted(required.difference(leads_df.columns)), leads_df, deals_df)
    summary = leads_df.groupby("Lead_Source", as_index=False).agg(
        Lead_Count=("Lead_ID", "count"),
        Total_Estimated_Value=("Estimated_Value", "sum"),
    ).sort_values("Lead_Count", ascending=False)
    return _make_result(
        answer_type="table",
        title="Leads by Source",
        answer="Leads grouped by source with counts and total estimated value.",
        data=summary,
        recommended_action="Compare source quality using lead count and estimated value.",
        answer_source="table aggregation",
        metric_type="lead_count",
        business_object="lead source",
    )


def _answer_forecast_summary_chart(deals_df: pd.DataFrame, leads_df: pd.DataFrame) -> dict[str, Any]:
    required = {"Forecast_Category", "Deal_ID", "Amount"}
    if not required.issubset(deals_df.columns):
        return _missing_data_answer("Forecast Summary Chart Unavailable", sorted(required.difference(deals_df.columns)), leads_df, deals_df)
    summary = deals_df.groupby("Forecast_Category", as_index=False).agg(
        Deal_Count=("Deal_ID", "count"),
        Total_Amount=("Amount", "sum"),
    ).sort_values("Total_Amount", ascending=False)
    result = _make_result(
        answer_type="chart",
        title="Forecast Summary Chart",
        answer="Showing forecast summary based on the active dataset.",
        data=summary,
        recommended_action="Review forecast categories with the largest amounts.",
        answer_source="predefined chart",
        metric_type="pipeline",
        business_object="forecast summary",
        chart_key="forecast_summary",
        chart_type="bar",
        x_axis="Forecast_Category",
        y_axis="Total_Amount",
        chart_title="Forecast Summary Chart",
    )
    result.update(_build_chart_context("Forecast Summary Chart", summary, "Forecast_Category", "Total_Amount", "bar"))
    return result


def _grouped_dataset_for_dimension(
    parsed: ParsedQuery,
    leads_df: pd.DataFrame,
    deals_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    dimension = parsed.dimension
    if dimension is None:
        return pd.DataFrame(), ["dimension"]
    if parsed.metric_name in {"revenue", "pipeline", "deal_count", "average_deal_size", "margin", "win_rate"}:
        dataset = _resolve_grouped_deals_dataset(deals_df, leads_df, dimension)
        if dataset.empty and not deals_df.empty:
            missing = []
            if dimension not in deals_df.columns and dimension not in leads_df.columns:
                missing.append(dimension)
            elif dimension in {"Region", "Sales_Rep", "Lead_Source"} and "Lead_ID" not in deals_df.columns:
                missing.append("Lead_ID")
            return pd.DataFrame(), missing
        return dataset, []
    return pd.DataFrame(), []


def _prepare_dataset_for_query(
    parsed: ParsedQuery,
    leads_df: pd.DataFrame,
    deals_df: pd.DataFrame,
    extra_tables: dict[str, pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Prepare the correct dataset, including deterministic joins when required."""
    extra_tables = extra_tables or {}

    def merge_lookup_dimension(
        lookup_df: pd.DataFrame,
        join_key: str,
        lookup_columns: list[str],
    ) -> tuple[pd.DataFrame, list[str]]:
        missing_columns = [column for column in [join_key, *lookup_columns] if column not in lookup_df.columns]
        if join_key not in deals_df.columns:
            missing_columns.insert(0, join_key)
        if missing_columns:
            return pd.DataFrame(), sorted(set(missing_columns))
        merged = deals_df.merge(
            lookup_df[[join_key, *lookup_columns]].drop_duplicates(subset=[join_key]),
            on=join_key,
            how="left",
        )
        return merged, []

    if parsed.requires_join:
        missing: list[str] = []
        for frame_name, frame in (("leads", leads_df), ("deals", deals_df)):
            if parsed.join_key not in frame.columns:
                missing.append(f"{parsed.join_key} in {frame_name}")
        if missing:
            return pd.DataFrame(), missing
        join_columns = [parsed.join_key] + [dimension for dimension in parsed.dimensions if dimension in leads_df.columns]
        if len(join_columns) == 1:
            return pd.DataFrame(), parsed.dimensions
        merged = deals_df.merge(
            leads_df[join_columns].drop_duplicates(subset=[parsed.join_key]),
            on=parsed.join_key,
            how="left",
        )
        return merged, []

    if (
        parsed.dimension in {"Account_Name", "Customer_Type", "Region"}
        and parsed.dimension not in deals_df.columns
        and "Accounts" in extra_tables
    ):
        accounts_df = extra_tables["Accounts"]
        if parsed.dimension in accounts_df.columns:
            return merge_lookup_dimension(accounts_df, "Account_ID", [parsed.dimension])

    if (
        parsed.dimension in {"Product_Name", "Product_Category"}
        and parsed.dimension not in deals_df.columns
        and "Products" in extra_tables
    ):
        products_df = extra_tables["Products"]
        if parsed.dimension in products_df.columns:
            return merge_lookup_dimension(products_df, "Product_ID", [parsed.dimension])

    if parsed.dimension is not None:
        return _grouped_dataset_for_dimension(parsed, leads_df, deals_df)

    return deals_df.copy(), []


def _answer_compound_grouped_query(
    parsed: ParsedQuery,
    leads_df: pd.DataFrame,
    deals_df: pd.DataFrame,
    extra_tables: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """Handle multi-metric grouped analytics questions with Pandas only."""
    dataset, missing = _prepare_dataset_for_query(parsed, leads_df, deals_df, extra_tables=extra_tables)
    if missing:
        return _missing_data_answer("Compound Analysis Unavailable", missing, leads_df, deals_df)
    if dataset.empty:
        return _unsupported_answer(leads_df, deals_df, title="Compound Analysis Unavailable", message="There is no matching data available for this compound analysis.")

    dimension = parsed.dimensions[0] if parsed.dimensions else parsed.dimension
    if dimension is None or dimension not in dataset.columns:
        return _missing_data_answer("Compound Analysis Unavailable", [dimension or "dimension"], leads_df, deals_df)

    required_columns = {"Deal_ID", "Amount"}
    missing_columns = sorted(required_columns.difference(dataset.columns))
    if missing_columns:
        return _missing_data_answer("Compound Analysis Unavailable", missing_columns, leads_df, deals_df)

    grouped = (
        dataset.dropna(subset=[dimension])
        .groupby(dimension, as_index=False)
        .agg(
            Total_Revenue=("Amount", "sum"),
            Average_Deal_Size=("Amount", "mean"),
            Deal_Count=("Deal_ID", "count"),
        )
        .sort_values("Total_Revenue", ascending=False)
    )
    if grouped.empty:
        return _unsupported_answer(leads_df, deals_df, title="Compound Analysis Unavailable", message="There is no grouped data available after applying the requested comparison.")

    grouped["Revenue_Rank"] = grouped["Total_Revenue"].rank(method="dense", ascending=False)
    grouped["Average_Deal_Size_Rank"] = grouped["Average_Deal_Size"].rank(method="dense", ascending=False)
    grouped["Deal_Count_Rank"] = grouped["Deal_Count"].rank(method="dense", ascending=False)
    grouped["Priority_Score"] = (
        grouped["Revenue_Rank"] + grouped["Average_Deal_Size_Rank"] + grouped["Deal_Count_Rank"]
    )
    grouped["Priority_Rank"] = grouped["Priority_Score"].rank(method="dense", ascending=True)
    priority_row = grouped.sort_values(["Priority_Score", "Total_Revenue"], ascending=[True, False]).iloc[0]

    title = "Customer Type Performance" if dimension == "Customer_Type" else f"{dimension.replace('_', ' ')} Performance"
    answer = (
        f"{priority_row[dimension]} deserves the highest management priority: "
        f"{_format_aed_compact(priority_row['Total_Revenue'])} total revenue, "
        f"{_format_aed_compact(priority_row['Average_Deal_Size'])} average deal size, "
        f"and {int(priority_row['Deal_Count'])} deals."
    )
    recommended_action = (
        f"Prioritize {priority_row[dimension]} because it has the strongest revenue contribution "
        "and a strong average deal value."
    )

    result = _make_result(
        answer_type="chart" if parsed.wants_chart or parsed.comparison or parsed.include_full_distribution else "table",
        title=title,
        answer=answer,
        recommended_action=recommended_action,
        data=grouped.reset_index(drop=True),
        answer_source="semantic BI calculation",
        metric_type=parsed.metric_name or (parsed.metrics[0] if parsed.metrics else "compound_metric"),
        metrics=parsed.metrics or ["revenue", "average_deal_size", "deal_count"],
        business_object=dimension.replace("_", " ").lower(),
        chart_type=parsed.chart_type or "bar",
        x_axis=dimension,
        y_axis="Total_Revenue",
        chart_title=title,
        key_observations=[
            "Priority score is based on revenue rank, average deal size rank, and deal count rank.",
            f"Highest priority segment is {priority_row[dimension]}.",
        ],
        query_debug={
            "intent": parsed.intent,
            "metrics": parsed.metrics,
            "dimensions": parsed.dimensions,
            "comparison": parsed.comparison,
            "recommendation_requested": parsed.recommendation_requested,
            "dataset": parsed.dataset,
            "requires_join": parsed.requires_join,
            "join_key": parsed.join_key,
        },
    )
    if result["answer_type"] == "chart":
        result.update(_build_chart_context(title, result["data"], dimension, "Total_Revenue", result["chart_type"]))
    return _with_engine_metadata(result, parsed)


def _generic_grouped_answer(
    parsed: ParsedQuery,
    leads_df: pd.DataFrame,
    deals_df: pd.DataFrame,
    extra_tables: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    dataset, missing = _prepare_dataset_for_query(parsed, leads_df, deals_df, extra_tables=extra_tables)
    if missing:
        return _missing_data_answer("Grouped Analysis Unavailable", missing, leads_df, deals_df)
    if dataset.empty:
        return _unsupported_answer(leads_df, deals_df, title="Grouped Analysis Unavailable", message="There is no matching data available for this grouped analysis.")

    dimension = parsed.dimension
    assert dimension is not None

    if parsed.filter_pipeline_only and "Forecast_Category" in dataset.columns:
        dataset = dataset[dataset["Forecast_Category"] == "Pipeline"]
    if parsed.filters:
        for column, value in parsed.filters.items():
            if column not in dataset.columns:
                return _missing_data_answer("Grouped Analysis Unavailable", [column], leads_df, deals_df)
            dataset = dataset[dataset[column].astype(str).str.lower() == str(value).lower()]

    if parsed.metric_name == "revenue":
        summary = group_sum(dataset, "Amount", dimension)
        value_column = "Value"
        table_df = summary.rename(columns={"Value": "Total_Amount"})
        answer_source = "semantic BI calculation"
    elif parsed.metric_name == "pipeline":
        if dimension == "Forecast_Category":
            summary = group_sum(dataset, "Amount", dimension)
        else:
            summary = group_sum(dataset[dataset["Forecast_Category"] == "Pipeline"] if "Forecast_Category" in dataset.columns else dataset, "Amount", dimension)
        value_column = "Value"
        table_df = summary.rename(columns={"Value": "Total_Amount"})
        answer_source = "semantic BI calculation"
    elif parsed.metric_name == "average_deal_size":
        summary = group_average(dataset, "Amount", dimension)
        value_column = "Value"
        table_df = summary.rename(columns={"Value": "Average_Deal_Size"})
        answer_source = "semantic BI calculation"
    elif parsed.metric_name == "deal_count":
        summary = group_count(dataset, "Deal_ID", dimension)
        value_column = "Value"
        table_df = summary.rename(columns={"Value": "Deal_Count"})
        answer_source = "table aggregation" if parsed.intent_type == "table" else "semantic bi query engine"
    elif parsed.metric_name == "margin":
        if "Margin" not in dataset.columns:
            return _missing_data_answer("Margin Analysis Unavailable", ["Margin"], leads_df, deals_df)
        summary = group_sum(dataset, "Margin", dimension)
        value_column = "Value"
        table_df = summary.rename(columns={"Value": "Total_Margin"})
        answer_source = "semantic BI calculation"
    elif parsed.metric_name == "win_rate":
        summary = win_rate(dataset, dimension)
        if summary.empty:
            return _unsupported_answer(leads_df, deals_df, title="Win Rate Unavailable", message="There are no Closed Won or Closed Lost deals available for win-rate analysis.")
        summary = summary.rename(columns={"Win_Rate": "Value"})
        value_column = "Value"
        table_df = summary.rename(columns={"Value": "Win_Rate"})
        answer_source = "semantic BI calculation"
    else:
        return _unsupported_answer(leads_df, deals_df)

    if summary.empty:
        return _unsupported_answer(leads_df, deals_df, title="Grouped Analysis Unavailable", message="There is no matching data available for this grouped analysis.")

    total_value = float(summary[value_column].sum()) if value_column in summary.columns else 0.0
    if parsed.percentage and total_value > 0:
        table_df["Share_Percentage"] = (table_df.iloc[:, 1] / total_value) * 100

    if parsed.sort_order == "asc":
        summary = summary.sort_values(value_column, ascending=True)
    else:
        summary = summary.sort_values(value_column, ascending=False)
    table_df = table_df.loc[summary.index]
    full_summary = summary.copy()
    full_table_df = table_df.copy()

    if parsed.limit and not parsed.include_full_distribution:
        summary = summary.head(parsed.limit)
        table_df = table_df.head(parsed.limit)

    dimension_label = dimension.replace("_", " ")
    metric_label_map = {
        "revenue": "Revenue",
        "pipeline": "Pipeline",
        "average_deal_size": "Average Deal Size",
        "deal_count": "Deal Count",
        "margin": "Margin",
        "win_rate": "Win Rate",
    }
    metric_label = metric_label_map.get(parsed.metric_name or "", "Value")

    if parsed.intent_type == "table":
        return _make_result(
            answer_type="table",
            title="Deals by Stage",
            answer="Deals grouped by stage with counts and total amount.",
            data=deals_df.groupby("Deal_Stage", as_index=False).agg(Deal_Count=("Deal_ID", "count"), Total_Amount=("Amount", "sum")).sort_values("Total_Amount", ascending=False),
            recommended_action="Focus on stages with the highest total amount.",
            answer_source="table aggregation",
            metric_type="deal_count",
            business_object="deal stage",
        )

    if parsed.limit == 1 and parsed.intent_type == "grouped_metric" and not parsed.include_full_distribution:
        top_row = summary.iloc[0]
        if parsed.metric_name == "win_rate":
            answer_value = f"{top_row[value_column]:.1f}%"
        elif parsed.metric_name == "deal_count":
            answer_value = f"{int(top_row[value_column]):,}"
        else:
            answer_value = _format_aed_compact(top_row[value_column])
        title_map = {
            ("revenue", "Region"): "Top Region by Revenue",
            ("pipeline", "Sales_Rep"): "Top Sales Rep by Pipeline",
            ("revenue", "Product_Category"): "Top Product Category by Revenue",
            ("revenue", "Account_Name"): "Top Account by Revenue",
            ("revenue", "Product_Name"): "Top Product by Revenue",
        }
        title_prefix = "Bottom" if parsed.sort_order == "asc" else "Top"
        title = title_map.get((parsed.metric_name or "", dimension), f"{title_prefix} {dimension_label} by {metric_label}")
        answer_body = f"{top_row[dimension]} — {answer_value}"
        if parsed.percentage and total_value > 0:
            percentage_value = (float(top_row[value_column]) / total_value) * 100
            answer_body = (
                f"{top_row[dimension]} contributes the highest share of {metric_label.lower()}: "
                f"{answer_value}, representing {percentage_value:.1f}% of the total."
            )
        result = _make_result(
            answer_type="metric",
            title=title,
            answer=answer_body,
            data=table_df,
            recommended_action=f"Review performance drivers for {top_row[dimension]}.",
            answer_source="metric calculation",
            metric_type=parsed.metric_name or "metric",
            business_object=dimension_label.lower(),
        )
        return _with_engine_metadata(result, parsed)

    title = {
        ("revenue", "Region"): "Revenue by Region",
        ("revenue", "Product_Category"): "Revenue by Product Category",
        ("revenue", "Product_Name"): "Revenue by Product",
        ("revenue", "Account_Name"): "Revenue by Customer",
        ("pipeline", "Sales_Rep"): "Pipeline by Sales Rep",
        ("average_deal_size", "Region"): "Average Deal Size by Region",
        ("deal_count", "Deal_Stage"): "Deals by Stage Chart",
        ("margin", "Product_Category"): "Margin by Product Category",
        ("win_rate", "Lead_Source"): "Win Rate by Lead Source",
        ("pipeline", "Forecast_Category"): "Pipeline by Forecast Category",
    }.get((parsed.metric_name or "", dimension), f"{metric_label} by {dimension_label}")

    if parsed.sort_order == "asc" and parsed.limit:
        title = f"Bottom {parsed.limit} {dimension_label} by {metric_label}"
    elif parsed.sort_order == "desc" and parsed.limit and parsed.limit > 1:
        title = f"Top {parsed.limit} {dimension_label} by {metric_label}"

    chart_type = parsed.chart_type or "bar"
    answer_text = f"Showing {metric_label.lower()} grouped by {dimension_label.lower()}."
    if parsed.sort_order == "asc" and parsed.limit and not table_df.empty:
        lines = [
            f"{row[dimension]} — "
            f"{f'{row[table_df.columns[1]]:.1f}%' if 'Rate' in table_df.columns[1] else _format_aed_compact(row[table_df.columns[1]]) if table_df.columns[1] != 'Deal_Count' else int(row[table_df.columns[1]])}"
            for _, row in table_df.head(parsed.limit).iterrows()
        ]
        answer_text = "Bottom performers:\n" + "\n".join(lines)
    elif parsed.percentage and total_value > 0 and not table_df.empty:
        top_row = table_df.iloc[0]
        share_column = "Share_Percentage"
        answer_text = (
            f"{top_row[dimension]} contributes the highest share of {metric_label.lower()}: "
            f"{_format_aed_compact(top_row[table_df.columns[1]]) if table_df.columns[1] != 'Win_Rate' else f'{top_row[table_df.columns[1]]:.1f}%'} "
            f"representing {top_row[share_column]:.1f}% of the total."
        )
    answer_type = "chart" if parsed.wants_chart or parsed.intent_type == "grouped_chart" else "table"
    chart_key = "deals_by_stage" if parsed.metric_name == "deal_count" and dimension == "Deal_Stage" and answer_type == "chart" else None

    recommended_action = f"Review which {dimension_label.lower()} segments drive the strongest {metric_label.lower()}."
    if parsed.sort_order == "asc" and "Product_Category" in dimension:
        recommended_action = "Review demand, pricing, sales coverage, or forecast quality for these product categories."
    result = _make_result(
        answer_type=answer_type,
        title=title,
        answer=answer_text,
        data=(full_table_df if parsed.include_full_distribution else table_df).reset_index(drop=True),
        recommended_action=recommended_action,
        answer_source="semantic BI calculation" if answer_type == "chart" else "table aggregation",
        metric_type=parsed.metric_name or "metric",
        business_object=dimension_label.lower(),
        chart_key=chart_key,
        chart_type=chart_type,
        x_axis=dimension,
        y_axis=(full_table_df if parsed.include_full_distribution else table_df).columns[1],
        chart_title=title,
    )
    if answer_type == "chart":
        result.update(_build_chart_context(title, result["data"], dimension, result["y_axis"], chart_type))
    return _with_engine_metadata(result, parsed)


def _answer_simple_metric(parsed: ParsedQuery, leads_df: pd.DataFrame, deals_df: pd.DataFrame) -> dict[str, Any]:
    question = parsed.normalized_question
    if "how many lead" in question or "total lead" in question:
        return _with_engine_metadata(_answer_total_leads(leads_df), parsed)
    if "how many deal" in question or "total deal" in question:
        return _with_engine_metadata(_answer_total_deals(deals_df), parsed)
    if parsed.metric_name == "total_pipeline_value" or parsed.metric_name == "pipeline":
        return _with_engine_metadata(_answer_total_pipeline(leads_df, deals_df), parsed)
    if parsed.metric_name == "average_deal_size":
        if deals_df.empty or "Amount" not in deals_df.columns:
            return _missing_data_answer("Average Deal Size Unavailable", ["Amount"], leads_df, deals_df)
        return _with_engine_metadata(_make_result(
            answer_type="metric",
            title="Average Deal Size",
            answer=_format_aed_compact(deals_df["Amount"].mean()) if not deals_df.empty else "N/A",
            recommended_action="Use this benchmark when reviewing new quotations.",
            answer_source="metric calculation",
            metric_type="average_deal_size",
            business_object="deal size",
        ), parsed)
    if parsed.metric_name == "win_rate":
        summary = win_rate(deals_df)
        if summary.empty:
            return _unsupported_answer(leads_df, deals_df, title="Win Rate", message="Win rate cannot be calculated because no Closed Won or Closed Lost deals exist.")
        return _with_engine_metadata(_make_result(
            answer_type="metric",
            title="Win Rate",
            answer=f"{summary.iloc[0]['Win_Rate']:.1f}%",
            recommended_action="Review lost deals to identify recurring blockers.",
            answer_source="metric calculation",
            metric_type="win_rate",
            business_object="win rate",
        ), parsed)
    if "estimated value" in question or "lead value" in question:
        if "Estimated_Value" not in leads_df.columns:
            return _missing_data_answer("Total Estimated Lead Value Unavailable", ["Estimated_Value"], leads_df, deals_df)
        return _with_engine_metadata(_make_result(
            answer_type="metric",
            title="Total Estimated Lead Value",
            answer=_format_aed_compact(leads_df["Estimated_Value"].sum()),
            recommended_action="Use this total to review inbound lead potential.",
            answer_source="metric calculation",
            metric_type="estimated_lead_value",
            business_object="lead value",
        ), parsed)
    if "deal amount" in question or "total amount" in question or parsed.metric_name == "revenue":
        if "Amount" not in deals_df.columns:
            return _missing_data_answer("Total Deal Amount Unavailable", ["Amount"], leads_df, deals_df)
        return _with_engine_metadata(_make_result(
            answer_type="metric",
            title="Total Deal Amount",
            answer=_format_aed_compact(deals_df["Amount"].sum()),
            recommended_action="Use this total to review overall deal value.",
            answer_source="metric calculation",
            metric_type="revenue",
            business_object="deal value",
        ), parsed)
    return _unsupported_answer(leads_df, deals_df)


def answer_business_question(
    question: str,
    leads_df: pd.DataFrame | None = None,
    deals_df: pd.DataFrame | None = None,
    extra_tables: dict[str, pd.DataFrame] | None = None,
    active_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Answer an analytics question using one deterministic semantic BI engine."""
    leads_df = leads_df if leads_df is not None else pd.DataFrame()
    deals_df = deals_df if deals_df is not None else pd.DataFrame()

    parsed = parse_business_question(question, active_context=active_context)
    if not parsed.normalized_question:
        return _unsupported_answer(leads_df, deals_df)

    if len(parsed.metrics) > 1 and parsed.dimensions:
        return _answer_compound_grouped_query(parsed, leads_df, deals_df, extra_tables=extra_tables)

    if parsed.metric_name is None:
        question_lower = parsed.normalized_question
        if "leads by source" in question_lower:
            return _with_engine_metadata(_answer_leads_by_source(leads_df, deals_df), parsed)
        if "how many lead" in question_lower or "total lead" in question_lower:
            return _with_engine_metadata(_answer_total_leads(leads_df), parsed)
        if "visualise forecast summary" in question_lower or "visualize forecast summary" in question_lower:
            return _with_engine_metadata(_answer_forecast_summary_chart(deals_df, leads_df), parsed)
        if active_context and not parsed.metric_name:
            return {
                "answer_type": "unsupported",
                "title": "Metric Needed for Follow-up",
                "answer": "I need a metric such as revenue, pipeline, deal count, margin, or win rate to answer this.",
                "recommended_action": "Ask a follow-up like 'show it by region' or 'bottom 3 product categories by revenue'.",
                "answer_source": "unsupported",
            }
        wants_chart = any(keyword in question_lower for keyword in ("chart", "graph", "plot", "visualize", "visualise"))
        return _unsupported_chart_answer(leads_df, deals_df) if wants_chart else _unsupported_answer(leads_df, deals_df)

    question_lower = parsed.normalized_question

    if "region" in question_lower and "opportunity" in question_lower and "most" in question_lower:
        if "Region" not in deals_df.columns and "Region" not in leads_df.columns:
            return _make_result(
                answer_type="metric",
                title="Top Region by Opportunity Value",
                answer="N/A",
                recommended_action="Region data is not available in the current dataset.",
                answer_source="metric calculation",
                metric_type="revenue",
                metrics=["revenue"],
                business_object="region",
            )

    if parsed.metric_name == "revenue" and parsed.dimension == "Product_Category" and parsed.limit == 1:
        if "Product_Category" not in deals_df.columns:
            return _make_result(
                answer_type="metric",
                title="Top Product Category by Revenue",
                answer="N/A",
                recommended_action="Product category data is not available in the current dataset.",
                answer_source="metric calculation",
                metric_type="revenue",
                metrics=["revenue"],
                business_object="product category",
            )

    if parsed.metric_name == "revenue" and parsed.dimension == "Region" and parsed.limit == 1 and "opportunity" in question_lower:
        if "Region" not in deals_df.columns and "Region" not in leads_df.columns:
            return _make_result(
                answer_type="metric",
                title="Top Region by Opportunity Value",
                answer="N/A",
                recommended_action="Region data is not available in the current dataset.",
                answer_source="metric calculation",
                metric_type="revenue",
                metrics=["revenue"],
                business_object="region",
            )

    if "create a chart of product category revenue" in question_lower and "Product_Category" not in deals_df.columns:
        return AnswerResult(
            answer_type="unsupported",
            title="Product Category Revenue Chart Unavailable",
            answer="Product category revenue chart is unavailable because `Product_Category` data is missing or there are no deals to plot.",
            recommended_action="Upload deals with `Product_Category` values to enable this chart.",
            answer_source="unsupported",
        ).to_dict()

    if "region-wise pipeline chart" in question_lower or ("region" in question_lower and "pipeline" in question_lower and parsed.wants_chart):
        if "Region" not in deals_df.columns and "Region" not in leads_df.columns:
            return AnswerResult(
                answer_type="unsupported",
                title="Region Pipeline Chart Unavailable",
                answer="Region-wise pipeline chart is unavailable because `Region` data is missing or there are no pipeline deals to plot.",
                recommended_action="Upload leads or deals with `Region` values and pipeline deals.",
                answer_source="unsupported",
            ).to_dict()

    if parsed.metric_name == "forecast_quality":
        return _with_engine_metadata(_answer_forecast_quality(deals_df, leads_df), parsed)
    if parsed.metric_name == "lead_source_conversion":
        return _with_engine_metadata(_answer_lead_source_conversion(leads_df, deals_df), parsed)
    if parsed.intent_type == "time_trend":
        return _with_engine_metadata(_answer_monthly_trend(parsed, leads_df, deals_df), parsed)
    if parsed.metric_name == "deal_count" and parsed.dimension == "Deal_Stage" and not parsed.wants_chart and "show" in parsed.normalized_question:
        return _with_engine_metadata(_answer_stage_table(deals_df, leads_df), parsed)
    if parsed.metric_name == "deal_count" and parsed.dimension == "Deal_Stage" and parsed.wants_chart:
        result = _generic_grouped_answer(parsed, leads_df, deals_df, extra_tables=extra_tables)
        if result["answer_type"] == "chart":
            result["title"] = "Deals by Stage Chart"
            result["chart_key"] = "deals_by_stage"
        return result
    if parsed.dimension is not None:
        return _generic_grouped_answer(parsed, leads_df, deals_df, extra_tables=extra_tables)
    return _answer_simple_metric(parsed, leads_df, deals_df)
