"""Revenue operations calculations — all math performed by Pandas, never by the LLM."""

import pandas as pd

EMPTY_LEADS_BY_SOURCE_COLUMNS = [
    "Lead_Source",
    "Lead_Count",
    "Total_Estimated_Value",
]
EMPTY_DEALS_BY_STAGE_COLUMNS = ["Deal_Stage", "Deal_Count", "Total_Amount"]
EMPTY_FORECAST_COLUMNS = ["Forecast_Category", "Deal_Count", "Total_Amount"]
EMPTY_SALES_REP_COLUMNS = [
    "Sales_Rep",
    "Lead_Count",
    "Total_Estimated_Value",
    "Deal_Count",
    "Total_Deal_Amount",
]
EMPTY_PRODUCT_CATEGORY_COLUMNS = ["Product_Category", "Deal_Count", "Total_Amount"]
EMPTY_REGION_PIPELINE_COLUMNS = ["Region", "Deal_Count", "Total_Amount"]
EMPTY_MONTHLY_TREND_COLUMNS = ["Closing_Month", "Total_Amount"]
OPEN_PIPELINE_FORECAST_CATEGORIES = {"Pipeline", "Best Case", "Commit", "At Risk"}
OPEN_PIPELINE_STAGES = {
    "New Inquiry",
    "Technical Discussion",
    "Quotation Sent",
    "Negotiation",
    "PO Received",
    "Proposal",
}
OPEN_PIPELINE_EXCLUDED_STAGES = {"Closed Won", "Closed Lost"}

WEAK_FORECAST_CATEGORIES = {"At Risk", "Omitted", "Pipeline"}
STRONG_FORECAST_CATEGORIES = {"Commit", "Closed", "Best Case"}
FORECAST_QUALITY_INTENT_PHRASES = (
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
PRODUCT_CATEGORY_INTENT_PHRASES = (
    "product category",
    "product categories",
    "category",
    "categories",
)
SUPPORTED_QUESTION_EXAMPLES = (
    "What is the total pipeline value?",
    "Show me deals by stage",
    "Create a chart of deals by stage",
    "Which product category has the weakest forecast quality?",
    "Show monthly pipeline trend based on closing date.",
    "Which lead source generates the highest estimated value but low conversion into deals?",
)


def _empty_summary(
    columns: list[str],
    *,
    reason: str,
    required_columns: list[str] | None = None,
    available_columns: list[str] | None = None,
    applied_filters: list[str] | None = None,
) -> pd.DataFrame:
    """Return an empty dataframe with attached diagnostics for UI messaging."""
    df = pd.DataFrame(columns=columns)
    df.attrs["empty_reason"] = reason
    df.attrs["required_columns"] = required_columns or []
    df.attrs["available_columns"] = available_columns or []
    df.attrs["applied_filters"] = applied_filters or []
    return df


def get_empty_reason(df: pd.DataFrame, fallback: str) -> str:
    """Return a user-facing reason from dataframe attrs when available."""
    return str(df.attrs.get("empty_reason") or fallback)


def _filter_open_pipeline_deals(deals_df: pd.DataFrame) -> pd.DataFrame:
    """Return active/open pipeline deals using open stage or open forecast rules."""
    if deals_df.empty or "Amount" not in deals_df.columns:
        return deals_df.iloc[0:0].copy()

    working_df = deals_df.copy()
    forecast_mask = pd.Series(False, index=working_df.index)
    stage_mask = pd.Series(False, index=working_df.index)
    closed_mask = pd.Series(False, index=working_df.index)

    if "Forecast_Category" in working_df.columns:
        forecast_mask = working_df["Forecast_Category"].isin(OPEN_PIPELINE_FORECAST_CATEGORIES)
    if "Deal_Stage" in working_df.columns:
        stage_mask = working_df["Deal_Stage"].isin(OPEN_PIPELINE_STAGES)
        closed_mask = working_df["Deal_Stage"].isin(OPEN_PIPELINE_EXCLUDED_STAGES)

    if "Forecast_Category" in working_df.columns or "Deal_Stage" in working_df.columns:
        open_mask = (forecast_mask | stage_mask) & ~closed_mask
        return working_df.loc[open_mask].copy()

    return working_df.iloc[0:0].copy()


def get_total_leads(leads_df: pd.DataFrame) -> int:
    """Return the total number of leads."""
    return len(leads_df)


def get_total_deals(deals_df: pd.DataFrame) -> int:
    """Return the total number of deals."""
    return len(deals_df)


def get_total_estimated_lead_value(leads_df: pd.DataFrame) -> float:
    """Return the sum of Estimated_Value across all leads."""
    if leads_df.empty or "Estimated_Value" not in leads_df.columns:
        return 0.0
    return leads_df["Estimated_Value"].sum()


def get_total_deal_amount(deals_df: pd.DataFrame) -> float:
    """Return the sum of Amount across all deals."""
    if deals_df.empty or "Amount" not in deals_df.columns:
        return 0.0
    return deals_df["Amount"].sum()


def get_leads_by_source(leads_df: pd.DataFrame) -> pd.DataFrame:
    """Group leads by Lead_Source with counts and total estimated value."""
    if leads_df.empty or not {
        "Lead_Source",
        "Lead_ID",
        "Estimated_Value",
    }.issubset(leads_df.columns):
        return _empty_summary(
            EMPTY_LEADS_BY_SOURCE_COLUMNS,
            reason="Lead source analysis requires Lead_Source, Lead_ID, and Estimated_Value after column mapping.",
            required_columns=["Lead_Source", "Lead_ID", "Estimated_Value"],
            available_columns=leads_df.columns.tolist(),
        )

    summary = (
        leads_df.dropna(subset=["Lead_Source"]).groupby("Lead_Source", as_index=False)
        .agg(
            Lead_Count=("Lead_ID", "count"),
            Total_Estimated_Value=("Estimated_Value", "sum"),
        )
        .sort_values("Lead_Count", ascending=False)
    )
    if summary.empty:
        return _empty_summary(
            EMPTY_LEADS_BY_SOURCE_COLUMNS,
            reason="Lead_Source exists after mapping, but all lead source values are empty in the uploaded data.",
            required_columns=["Lead_Source"],
            available_columns=leads_df.columns.tolist(),
        )
    return summary


def get_deals_by_stage(deals_df: pd.DataFrame) -> pd.DataFrame:
    """Group deals by Deal_Stage with counts and total amount."""
    if deals_df.empty or not {"Deal_Stage", "Deal_ID", "Amount"}.issubset(
        deals_df.columns
    ):
        return pd.DataFrame(columns=EMPTY_DEALS_BY_STAGE_COLUMNS)

    summary = (
        deals_df.groupby("Deal_Stage", as_index=False)
        .agg(
            Deal_Count=("Deal_ID", "count"),
            Total_Amount=("Amount", "sum"),
        )
        .sort_values("Total_Amount", ascending=False)
    )
    return summary


def get_forecast_summary(deals_df: pd.DataFrame) -> pd.DataFrame:
    """Group deals by Forecast_Category with counts and total amount."""
    if deals_df.empty or not {"Forecast_Category", "Deal_ID", "Amount"}.issubset(
        deals_df.columns
    ):
        return pd.DataFrame(columns=EMPTY_FORECAST_COLUMNS)

    summary = (
        deals_df.groupby("Forecast_Category", as_index=False)
        .agg(
            Deal_Count=("Deal_ID", "count"),
            Total_Amount=("Amount", "sum"),
        )
        .sort_values("Total_Amount", ascending=False)
    )
    return summary


def get_sales_rep_performance(
    leads_df: pd.DataFrame, deals_df: pd.DataFrame
) -> pd.DataFrame:
    """Summarize lead and deal performance by sales rep."""
    lead_summary = pd.DataFrame(columns=["Sales_Rep", "Lead_Count", "Total_Estimated_Value"])
    if not leads_df.empty and {"Sales_Rep", "Lead_ID", "Estimated_Value"}.issubset(leads_df.columns):
        lead_summary = (
            leads_df.dropna(subset=["Sales_Rep"])
            .groupby("Sales_Rep", as_index=False)
            .agg(
                Lead_Count=("Lead_ID", "count"),
                Total_Estimated_Value=("Estimated_Value", "sum"),
            )
        )

    deal_summary = pd.DataFrame(columns=["Sales_Rep", "Deal_Count", "Total_Deal_Amount"])
    if not deals_df.empty and {"Deal_ID", "Amount"}.issubset(deals_df.columns):
        if "Sales_Rep" in deals_df.columns and deals_df["Sales_Rep"].notna().any():
            deals_with_rep = deals_df.copy()
        elif (
            "Lead_ID" in deals_df.columns
            and not leads_df.empty
            and {"Lead_ID", "Sales_Rep"}.issubset(leads_df.columns)
        ):
            deals_with_rep = deals_df.merge(
                leads_df[["Lead_ID", "Sales_Rep"]],
                on="Lead_ID",
                how="left",
            )
        else:
            deals_with_rep = pd.DataFrame()

        if not deals_with_rep.empty and "Sales_Rep" in deals_with_rep.columns:
            deal_summary = (
                deals_with_rep.dropna(subset=["Sales_Rep"])
                .groupby("Sales_Rep", as_index=False)
                .agg(
                    Deal_Count=("Deal_ID", "count"),
                    Total_Deal_Amount=("Amount", "sum"),
                )
            )

    if lead_summary.empty and deal_summary.empty:
        return _empty_summary(
            EMPTY_SALES_REP_COLUMNS,
            reason="Sales rep performance cannot be computed because Sales_Rep is missing or empty after column mapping.",
            required_columns=["Sales_Rep", "Deal_ID", "Amount"],
            available_columns=sorted(set(leads_df.columns.tolist() + deals_df.columns.tolist())),
        )

    if lead_summary.empty:
        performance = deal_summary.copy()
        performance["Lead_Count"] = 0
        performance["Total_Estimated_Value"] = 0.0
    elif deal_summary.empty:
        performance = lead_summary.copy()
        performance["Deal_Count"] = 0
        performance["Total_Deal_Amount"] = 0.0
    else:
        performance = lead_summary.merge(deal_summary, on="Sales_Rep", how="outer")
        performance["Lead_Count"] = performance["Lead_Count"].fillna(0).astype(int)
        performance["Total_Estimated_Value"] = performance["Total_Estimated_Value"].fillna(0.0)
        performance["Deal_Count"] = performance["Deal_Count"].fillna(0).astype(int)
        performance["Total_Deal_Amount"] = performance["Total_Deal_Amount"].fillna(0.0)

    return performance.sort_values(["Total_Deal_Amount", "Total_Estimated_Value"], ascending=False)


def get_total_pipeline_amount(deals_df: pd.DataFrame) -> float:
    """Return total deal amount in the Pipeline forecast category."""
    if deals_df.empty or not {"Forecast_Category", "Amount"}.issubset(deals_df.columns):
        return 0.0
    pipeline_deals = deals_df[deals_df["Forecast_Category"] == "Pipeline"]
    return pipeline_deals["Amount"].sum()


def get_product_category_revenue(deals_df: pd.DataFrame) -> pd.DataFrame:
    """Group deals by product category with counts and total amount."""
    if deals_df.empty or not {
        "Product_Category",
        "Deal_ID",
        "Amount",
    }.issubset(deals_df.columns):
        return pd.DataFrame(columns=EMPTY_PRODUCT_CATEGORY_COLUMNS)

    return (
        deals_df.groupby("Product_Category", as_index=False)
        .agg(
            Deal_Count=("Deal_ID", "count"),
            Total_Amount=("Amount", "sum"),
        )
        .sort_values("Total_Amount", ascending=False)
    )


def get_region_pipeline(leads_df: pd.DataFrame, deals_df: pd.DataFrame) -> pd.DataFrame:
    """Return pipeline deal summary by region from deals or lead-linked region data."""
    if deals_df.empty or not {"Deal_ID", "Amount"}.issubset(deals_df.columns):
        return _empty_summary(
            EMPTY_REGION_PIPELINE_COLUMNS,
            reason="Region pipeline requires Deals with Deal_ID and Amount after column mapping.",
            required_columns=["Deal_ID", "Amount"],
            available_columns=deals_df.columns.tolist(),
        )

    pipeline_deals = _filter_open_pipeline_deals(deals_df)
    if pipeline_deals.empty:
        return _empty_summary(
            EMPTY_REGION_PIPELINE_COLUMNS,
            reason="No open pipeline deals were found after excluding Closed Won and Closed Lost deals.",
            required_columns=["Forecast_Category or Deal_Stage"],
            available_columns=deals_df.columns.tolist(),
            applied_filters=[
                "Open forecast categories: Pipeline, Best Case, Commit, At Risk",
                "Open stages: New Inquiry, Technical Discussion, Quotation Sent, Negotiation, PO Received, Proposal",
            ],
        )

    if "Region" in pipeline_deals.columns:
        region_df = pipeline_deals
    elif "Region" in leads_df.columns and "Lead_ID" in leads_df.columns:
        region_df = pipeline_deals.merge(
            leads_df[["Lead_ID", "Region"]],
            on="Lead_ID",
            how="left",
        )
    else:
        return _empty_summary(
            EMPTY_REGION_PIPELINE_COLUMNS,
            reason="Region pipeline cannot be computed because Region is missing from both Deals and Lead-linked data after column mapping.",
            required_columns=["Region"],
            available_columns=sorted(set(leads_df.columns.tolist() + deals_df.columns.tolist())),
        )

    region_df = region_df.dropna(subset=["Region"])
    if region_df.empty:
        return _empty_summary(
            EMPTY_REGION_PIPELINE_COLUMNS,
            reason="Region exists after mapping, but all open pipeline rows have empty Region values.",
            required_columns=["Region"],
            available_columns=sorted(set(leads_df.columns.tolist() + deals_df.columns.tolist())),
        )

    return (
        region_df.groupby("Region", as_index=False)
        .agg(
            Deal_Count=("Deal_ID", "count"),
            Total_Amount=("Amount", "sum"),
        )
        .sort_values("Total_Amount", ascending=False)
    )


def get_monthly_amount_trend(
    deals_df: pd.DataFrame,
    forecast_filter: str | None = None,
) -> pd.DataFrame:
    """Group deals by closing month with safe datetime parsing."""
    required_columns = {"Closing_Date", "Amount"}
    if deals_df.empty or not required_columns.issubset(deals_df.columns):
        return pd.DataFrame(columns=EMPTY_MONTHLY_TREND_COLUMNS)

    working_df = deals_df.copy()
    working_df["Closing_Date"] = pd.to_datetime(
        working_df["Closing_Date"],
        errors="coerce",
        format="mixed",
    )
    working_df = working_df.dropna(subset=["Closing_Date"])
    if working_df.empty:
        return pd.DataFrame(columns=EMPTY_MONTHLY_TREND_COLUMNS)

    if forecast_filter is not None:
        if "Forecast_Category" not in working_df.columns:
            return pd.DataFrame(columns=EMPTY_MONTHLY_TREND_COLUMNS)
        working_df = working_df[working_df["Forecast_Category"] == forecast_filter]
        if working_df.empty:
            return pd.DataFrame(columns=EMPTY_MONTHLY_TREND_COLUMNS)

    working_df["Closing_Month_Start"] = (
        working_df["Closing_Date"].dt.to_period("M").dt.to_timestamp()
    )
    summary = (
        working_df.groupby("Closing_Month_Start", as_index=False)
        .agg(Total_Amount=("Amount", "sum"))
        .sort_values("Closing_Month_Start", ascending=True)
    )
    month_range = pd.date_range(
        start=summary["Closing_Month_Start"].min(),
        end=summary["Closing_Month_Start"].max(),
        freq="MS",
    )
    summary = (
        pd.DataFrame({"Closing_Month_Start": month_range})
        .merge(summary, on="Closing_Month_Start", how="left")
        .fillna({"Total_Amount": 0.0})
    )
    summary["Closing_Month"] = summary["Closing_Month_Start"].dt.strftime("%b %Y")
    return summary


def get_weakest_forecast_quality_by_product_category(
    deals_df: pd.DataFrame,
) -> pd.DataFrame:
    """Return product-category forecast quality ranked by weakest ratio first."""
    required_columns = {"Product_Category", "Forecast_Category", "Amount"}
    if deals_df.empty or not required_columns.issubset(deals_df.columns):
        return pd.DataFrame(
            columns=[
                "Product_Category",
                "Total_Amount",
                "Weak_Forecast_Amount",
                "Strong_Forecast_Amount",
                "Weak_Forecast_Ratio",
            ]
        )

    working_df = deals_df.dropna(subset=["Product_Category"]).copy()
    if working_df.empty:
        return pd.DataFrame(
            columns=[
                "Product_Category",
                "Total_Amount",
                "Weak_Forecast_Amount",
                "Strong_Forecast_Amount",
                "Weak_Forecast_Ratio",
            ]
        )

    working_df["Weak_Amount"] = working_df["Amount"].where(
        working_df["Forecast_Category"].isin(WEAK_FORECAST_CATEGORIES),
        0,
    )
    working_df["Strong_Amount"] = working_df["Amount"].where(
        working_df["Forecast_Category"].isin(STRONG_FORECAST_CATEGORIES),
        0,
    )

    summary = working_df.groupby("Product_Category", as_index=False).agg(
        Total_Amount=("Amount", "sum"),
        Weak_Forecast_Amount=("Weak_Amount", "sum"),
        Strong_Forecast_Amount=("Strong_Amount", "sum"),
    )
    non_zero_total = summary["Total_Amount"].replace(0, pd.NA)
    summary["Weak_Forecast_Ratio"] = (
        summary["Weak_Forecast_Amount"] / non_zero_total
    ).fillna(0.0)
    return summary.sort_values(
        ["Weak_Forecast_Ratio", "Weak_Forecast_Amount"],
        ascending=[False, False],
    )


def get_lead_source_value_vs_conversion(
    leads_df: pd.DataFrame,
    deals_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compare lead source value against conversion into deals."""
    lead_required = {"Lead_ID", "Lead_Source", "Estimated_Value"}
    deal_required = {"Lead_ID"}
    if leads_df.empty or deals_df.empty:
        return pd.DataFrame(
            columns=[
                "Lead_Source",
                "Total_Leads",
                "Converted_Leads",
                "Conversion_Rate",
                "Estimated_Value",
                "Avg_Estimated_Value",
            ]
        )
    if not lead_required.issubset(leads_df.columns) or not deal_required.issubset(
        deals_df.columns
    ):
        return pd.DataFrame(
            columns=[
                "Lead_Source",
                "Total_Leads",
                "Converted_Leads",
                "Conversion_Rate",
                "Estimated_Value",
                "Avg_Estimated_Value",
            ]
        )

    converted_lead_ids = deals_df["Lead_ID"].dropna().unique()
    leads_with_conversion = leads_df.copy()
    leads_with_conversion["Converted"] = leads_with_conversion["Lead_ID"].isin(
        converted_lead_ids
    )

    summary = leads_with_conversion.groupby("Lead_Source", as_index=False).agg(
        Total_Leads=("Lead_ID", "nunique"),
        Converted_Leads=("Converted", "sum"),
        Estimated_Value=("Estimated_Value", "sum"),
        Avg_Estimated_Value=("Estimated_Value", "mean"),
    )

    non_zero_leads = summary["Total_Leads"].replace(0, pd.NA)
    summary["Conversion_Rate"] = (
        summary["Converted_Leads"] / non_zero_leads
    ).fillna(0.0)
    return summary.sort_values(
        ["Estimated_Value", "Conversion_Rate"],
        ascending=[False, True],
    )


def format_aed_compact(value) -> str:
    """Format a numeric value as a compact AED currency string."""
    if value is None:
        return "N/A"

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "N/A"

    if numeric != numeric:
        return "N/A"

    absolute_value = abs(numeric)
    if absolute_value >= 1_000_000:
        return f"AED {numeric / 1_000_000:.2f}M"
    if absolute_value >= 1_000:
        return f"AED {numeric / 1_000:.2f}K"
    return f"AED {numeric:,.0f}"


def _sum_forecast_amount(deals_df: pd.DataFrame, category: str) -> float:
    """Return total deal amount for a forecast category."""
    if deals_df.empty or "Forecast_Category" not in deals_df.columns:
        return 0.0
    return deals_df.loc[deals_df["Forecast_Category"] == category, "Amount"].sum()


def _sum_stage_amount(deals_df: pd.DataFrame, stage: str) -> float:
    """Return total deal amount for a deal stage."""
    if deals_df.empty or "Deal_Stage" not in deals_df.columns:
        return 0.0
    return deals_df.loc[deals_df["Deal_Stage"] == stage, "Amount"].sum()


def _calculate_win_rate(deals_df: pd.DataFrame) -> float | None:
    """Return win rate percentage for closed deals, or None if unavailable."""
    if deals_df.empty or "Deal_Stage" not in deals_df.columns:
        return None

    closed_won_count = (deals_df["Deal_Stage"] == "Closed Won").sum()
    closed_lost_count = (deals_df["Deal_Stage"] == "Closed Lost").sum()
    total_closed = closed_won_count + closed_lost_count

    if total_closed == 0:
        return None

    return (closed_won_count / total_closed) * 100


def _make_insight(
    title: str,
    metric: str,
    explanation: str,
    recommended_action: str,
) -> dict:
    """Build a single auto insight dictionary."""
    return {
        "title": title,
        "metric": metric,
        "explanation": explanation,
        "recommended_action": recommended_action,
    }


def generate_auto_insights(
    leads_df: pd.DataFrame,
    deals_df: pd.DataFrame,
) -> list[dict]:
    """Generate management insights from Pandas calculations only."""
    insights: list[dict] = []

    pipeline_value = _sum_forecast_amount(deals_df, "Pipeline")
    commit_value = _sum_forecast_amount(deals_df, "Commit")
    closed_won_value = _sum_stage_amount(deals_df, "Closed Won")
    at_risk_value = _sum_forecast_amount(deals_df, "At Risk")

    insights.append(
        _make_insight(
            title="Total Open Pipeline Value",
            metric=format_aed_compact(pipeline_value),
            explanation=(
                f"{get_total_deals(deals_df[deals_df['Forecast_Category'] == 'Pipeline']) if not deals_df.empty else 0} "
                f"deal(s) are tagged as Pipeline with a combined amount of "
                f"{format_aed_compact(pipeline_value)}."
            ),
            recommended_action=(
                "Review open pipeline deals and confirm next steps with sales reps."
                if pipeline_value > 0
                else "No pipeline-tagged deals are currently recorded."
            ),
        )
    )

    insights.append(
        _make_insight(
            title="Committed Forecast Value",
            metric=format_aed_compact(commit_value),
            explanation=(
                f"Deals in the Commit forecast category total "
                f"{format_aed_compact(commit_value)}."
            ),
            recommended_action=(
                "Validate delivery capacity and stock for committed deals."
                if commit_value > 0
                else "No committed forecast deals are currently recorded."
            ),
        )
    )

    insights.append(
        _make_insight(
            title="Closed Won Revenue",
            metric=format_aed_compact(closed_won_value),
            explanation=(
                f"Closed Won deals contribute "
                f"{format_aed_compact(closed_won_value)} in recorded revenue."
            ),
            recommended_action=(
                "Track fulfillment and invoicing for won deals."
                if closed_won_value > 0
                else "No Closed Won revenue is recorded yet."
            ),
        )
    )

    insights.append(
        _make_insight(
            title="At-Risk Pipeline Value",
            metric=format_aed_compact(at_risk_value),
            explanation=(
                f"Deals marked At Risk total "
                f"{format_aed_compact(at_risk_value)}."
            ),
            recommended_action=(
                "Prioritize follow-up on at-risk deals before forecast slips."
                if at_risk_value > 0
                else "No at-risk deals are currently flagged."
            ),
        )
    )

    if deals_df.empty:
        average_deal_size = None
        average_metric = "N/A"
        average_explanation = "No deals are available to calculate an average deal size."
        average_action = "Add deal records to enable deal size analysis."
    else:
        average_deal_size = deals_df["Amount"].mean()
        average_metric = format_aed_compact(average_deal_size)
        average_explanation = (
            f"The average deal amount across {len(deals_df)} deal(s) is "
            f"{format_aed_compact(average_deal_size)}."
        )
        average_action = "Use average deal size to benchmark new quotations."

    insights.append(
        _make_insight(
            title="Average Deal Size",
            metric=average_metric,
            explanation=average_explanation,
            recommended_action=average_action,
        )
    )

    win_rate = _calculate_win_rate(deals_df)
    if win_rate is None:
        win_rate_metric = "N/A"
        win_rate_explanation = (
            "Win rate cannot be calculated because no Closed Won or Closed Lost deals exist."
        )
        win_rate_action = "Close more deals to establish a win rate baseline."
    else:
        win_rate_metric = f"{win_rate:.1f}%"
        closed_won_count = (deals_df["Deal_Stage"] == "Closed Won").sum()
        closed_lost_count = (deals_df["Deal_Stage"] == "Closed Lost").sum()
        win_rate_explanation = (
            f"{closed_won_count} deal(s) were Closed Won and "
            f"{closed_lost_count} deal(s) were Closed Lost."
        )
        win_rate_action = (
            "Review lost deals to identify recurring blockers."
            if win_rate < 50
            else "Maintain current sales practices that are converting deals."
        )

    insights.append(
        _make_insight(
            title="Win Rate",
            metric=win_rate_metric,
            explanation=win_rate_explanation,
            recommended_action=win_rate_action,
        )
    )

    if deals_df.empty:
        top_rep_name = "N/A"
        top_rep_metric = "N/A"
        top_rep_explanation = "No deals are available to rank sales reps."
        top_rep_action = "Add deal records to compare sales rep performance."
    else:
        rep_performance = get_sales_rep_performance(leads_df, deals_df)
        if rep_performance.empty:
            top_rep_name = "N/A"
            top_rep_metric = "N/A"
            top_rep_explanation = "No sales rep performance data is available to rank reps."
            top_rep_action = "Add lead ownership data to compare sales reps."
        else:
            top_rep = rep_performance.sort_values(
                "Total_Deal_Amount", ascending=False
            ).iloc[0]
            top_rep_name = top_rep["Sales_Rep"]
            top_rep_metric = format_aed_compact(top_rep["Total_Deal_Amount"])
            top_rep_explanation = (
                f"{top_rep_name} has the highest total deal amount at "
                f"{format_aed_compact(top_rep['Total_Deal_Amount'])} across "
                f"{int(top_rep['Deal_Count'])} deal(s)."
            )
            top_rep_action = (
                f"Review {top_rep_name}'s pipeline to replicate successful deal patterns."
            )

    insights.append(
        _make_insight(
            title="Top Sales Rep by Deal Amount",
            metric=top_rep_metric,
            explanation=top_rep_explanation,
            recommended_action=top_rep_action,
        )
    )

    if "Product_Category" in deals_df.columns and not deals_df.empty:
        product_summary = get_product_category_revenue(deals_df)
        if not product_summary.empty:
            top_product = product_summary.iloc[0]
            insights.append(
                _make_insight(
                    title="Top Product Category",
                    metric=format_aed_compact(top_product["Total_Amount"]),
                    explanation=(
                        f"{top_product['Product_Category']} leads product revenue with "
                        f"{format_aed_compact(top_product['Total_Amount'])} in deal amount."
                    ),
                    recommended_action=(
                        f"Ensure inventory and pricing support demand for "
                        f"{top_product['Product_Category']}."
                    ),
                )
            )

    if not leads_df.empty:
        source_summary = get_leads_by_source(leads_df)
        if not source_summary.empty:
            top_source = source_summary.iloc[0]
            insights.append(
                _make_insight(
                    title="Top Lead Source by Estimated Value",
                    metric=format_aed_compact(top_source["Total_Estimated_Value"]),
                    explanation=(
                        f"{top_source['Lead_Source']} generated "
                        f"{int(top_source['Lead_Count'])} lead(s) with estimated value of "
                        f"{format_aed_compact(top_source['Total_Estimated_Value'])}."
                    ),
                    recommended_action=(
                        f"Invest more in {top_source['Lead_Source']} if lead quality remains strong."
                    ),
                )
            )

    if "Region" in deals_df.columns and not deals_df.empty:
        region_summary = (
            deals_df.groupby("Region", as_index=False)
            .agg(Total_Amount=("Amount", "sum"))
            .sort_values("Total_Amount", ascending=False)
        )
        if not region_summary.empty:
            top_region = region_summary.iloc[0]
            insights.append(
                _make_insight(
                    title="Top Region by Deal Amount",
                    metric=format_aed_compact(top_region["Total_Amount"]),
                    explanation=(
                        f"{top_region['Region']} has the highest deal amount at "
                        f"{format_aed_compact(top_region['Total_Amount'])}."
                    ),
                    recommended_action=(
                        f"Align sales coverage and logistics for {top_region['Region']}."
                    ),
                )
            )

    if deals_df.empty or "Deal_Stage" not in deals_df.columns:
        stage_metric = "N/A"
        stage_explanation = "No deal stage data is available."
        stage_action = "Add deals to review stage distribution."
    else:
        stage_summary = get_deals_by_stage(deals_df)
        top_stage = stage_summary.iloc[0]
        stage_metric = (
            f"{top_stage['Deal_Stage']} ({int(top_stage['Deal_Count'])} deals)"
        )
        stage_explanation = (
            f"The largest stage by amount is {top_stage['Deal_Stage']} with "
            f"{format_aed_compact(top_stage['Total_Amount'])} across "
            f"{int(top_stage['Deal_Count'])} deal(s)."
        )
        stage_action = (
            f"Focus coaching on advancing deals out of {top_stage['Deal_Stage']}."
        )

    insights.append(
        _make_insight(
            title="Deals by Stage Summary",
            metric=stage_metric,
            explanation=stage_explanation,
            recommended_action=stage_action,
        )
    )

    focus_items = []
    if at_risk_value > 0:
        focus_items.append(
            ("at-risk pipeline", at_risk_value, "Review at-risk deals with account owners.")
        )
    if pipeline_value > 0:
        focus_items.append(
            ("open pipeline", pipeline_value, "Prioritize follow-ups on open pipeline deals.")
        )
    if commit_value > 0:
        focus_items.append(
            ("committed forecast", commit_value, "Confirm fulfillment plans for committed deals.")
        )
    if closed_won_value > 0:
        focus_items.append(
            ("closed won revenue", closed_won_value, "Ensure invoicing and delivery for won deals.")
        )

    if focus_items:
        focus_label, focus_value, focus_action = max(
            focus_items, key=lambda item: item[1]
        )
        management_metric = format_aed_compact(focus_value)
        management_explanation = (
            f"The largest current value area is {focus_label} at "
            f"{format_aed_compact(focus_value)}."
        )
        management_action = focus_action
    else:
        management_metric = "N/A"
        management_explanation = "No deal values are available for management prioritization."
        management_action = "Load deal data to generate management focus recommendations."

    insights.append(
        _make_insight(
            title="Management Focus Recommendation",
            metric=management_metric,
            explanation=management_explanation,
            recommended_action=management_action,
        )
    )

    return insights


UNSUPPORTED_ANSWER_MESSAGE = (
    "I can currently answer pipeline, forecast, lead source, sales rep, "
    "product category, region, stage, win rate, and chart questions."
)

CHART_KEYWORDS = (
    "chart",
    "graph",
    "plot",
    "visualize",
    "visualise",
    "bar chart",
)


def _wants_chart(question_lower: str) -> bool:
    """Return True if the question asks for a chart."""
    return any(keyword in question_lower for keyword in CHART_KEYWORDS)


def _average_deal_size(deals_df: pd.DataFrame) -> float | None:
    """Return average deal amount, or None if no deals exist."""
    if deals_df.empty:
        return None
    return deals_df["Amount"].mean()


def _get_top_lead_source(leads_df: pd.DataFrame) -> pd.Series | None:
    """Return the lead source row with the highest estimated value."""
    if leads_df.empty:
        return None
    source_summary = get_leads_by_source(leads_df).sort_values(
        "Total_Estimated_Value", ascending=False
    )
    if source_summary.empty:
        return None
    return source_summary.iloc[0]


def _get_top_sales_rep_pipeline(
    leads_df: pd.DataFrame, deals_df: pd.DataFrame
) -> pd.Series | None:
    """Return the sales rep with the highest pipeline deal amount."""
    if deals_df.empty or "Amount" not in deals_df.columns:
        return None

    if "Sales_Rep" in deals_df.columns and deals_df["Sales_Rep"].notna().any():
        deals_with_rep = deals_df.copy()
    elif "Lead_ID" in deals_df.columns and {"Lead_ID", "Sales_Rep"}.issubset(leads_df.columns):
        deals_with_rep = deals_df.merge(
            leads_df[["Lead_ID", "Sales_Rep"]],
            on="Lead_ID",
            how="left",
        )
    else:
        return None

    pipeline_deals = _filter_open_pipeline_deals(deals_with_rep)
    if pipeline_deals.empty:
        return None

    rep_summary = (
        pipeline_deals.groupby("Sales_Rep", as_index=False)
        .agg(Pipeline_Amount=("Amount", "sum"))
        .sort_values("Pipeline_Amount", ascending=False)
    )
    if rep_summary.empty:
        return None
    return rep_summary.iloc[0]


def _get_top_product_category(deals_df: pd.DataFrame) -> pd.Series | None:
    """Return the product category with the highest deal amount."""
    product_summary = get_product_category_revenue(deals_df)
    if product_summary.empty:
        return None
    return product_summary.iloc[0]


def _get_top_region(deals_df: pd.DataFrame, leads_df: pd.DataFrame) -> tuple[str, float] | None:
    """Return the top region name and opportunity value from deals or leads."""
    if "Region" in deals_df.columns and not deals_df.empty and "Amount" in deals_df.columns:
        region_summary = (
            deals_df.groupby("Region", as_index=False)
            .agg(Total_Amount=("Amount", "sum"))
            .sort_values("Total_Amount", ascending=False)
        )
        if region_summary.empty:
            return None
        top_region = region_summary.iloc[0]
        return top_region["Region"], float(top_region["Total_Amount"])

    if "Region" in leads_df.columns and not leads_df.empty:
        region_summary = (
            leads_df.groupby("Region", as_index=False)
            .agg(Total_Estimated_Value=("Estimated_Value", "sum"))
            .sort_values("Total_Estimated_Value", ascending=False)
        )
        if region_summary.empty:
            return None
        top_region = region_summary.iloc[0]
        return top_region["Region"], float(top_region["Total_Estimated_Value"])

    return None


def _make_answer(
    answer_type: str,
    title: str,
    answer: str,
    recommended_action: str,
    data: pd.DataFrame | None = None,
    chart_key: str | None = None,
    source: str | None = None,
    answer_source: str | None = None,
) -> dict:
    """Build a standardized question answer dictionary."""
    response = {
        "answer_type": answer_type,
        "title": title,
        "answer": answer,
        "data": data,
        "recommended_action": recommended_action,
    }
    if chart_key is not None:
        response["chart_key"] = chart_key
    if source is not None:
        response["source"] = source
    if answer_source is not None:
        response["answer_source"] = answer_source
    return response


def _available_field_summary(leads_df: pd.DataFrame, deals_df: pd.DataFrame) -> str:
    """Return a concise summary of available uploaded fields."""
    lead_fields = ", ".join(sorted(leads_df.columns.tolist())) or "none"
    deal_fields = ", ".join(sorted(deals_df.columns.tolist())) or "none"
    return f"Available Leads fields: {lead_fields}. Available Deals fields: {deal_fields}."


def _unsupported_answer(leads_df: pd.DataFrame, deals_df: pd.DataFrame) -> dict:
    """Return a standardized unsupported question response."""
    return _make_answer(
        answer_type="unsupported",
        title="Unsupported Question",
        answer=(
            f"{UNSUPPORTED_ANSWER_MESSAGE} "
            f"Examples: {'; '.join(SUPPORTED_QUESTION_EXAMPLES)}. "
            f"{_available_field_summary(leads_df, deals_df)}"
        ),
        recommended_action="Try one of the supported examples or ask using the available dataset fields.",
    )


def _unsupported_chart_answer(leads_df: pd.DataFrame, deals_df: pd.DataFrame) -> dict:
    """Return a standardized unsupported chart request response."""
    return _make_answer(
        answer_type="unsupported",
        title="Unsupported Chart Request",
        answer=(
            "I can currently create charts for leads by source, deals by stage, "
            "forecast summary, sales rep performance, product category revenue, "
            "region-wise pipeline, dynamic schema-based charts, and monthly trend charts when the needed columns exist. "
            f"Examples: Create a chart of deals by stage; Show revenue by region; Show monthly pipeline trend based on closing date. "
            f"{_available_field_summary(leads_df, deals_df)}"
        ),
        recommended_action=(
            "Try a supported chart example using the available dataset fields."
        ),
    )


def _missing_data_answer(title: str, message: str, action: str) -> dict:
    """Return a consistent unsupported response for missing required data."""
    return _make_answer(
        answer_type="unsupported",
        title=title,
        answer=message,
        recommended_action=action,
    )


def _is_cross_table_conversion_question(question_lower: str) -> bool:
    """Return True when the user asks about source value versus conversion."""
    return (
        "lead source" in question_lower or "source" in question_lower
    ) and any(
        phrase in question_lower
        for phrase in (
            "conversion",
            "converted",
            "into deals",
            "poor conversion",
            "low conversion",
            "weak opportunities",
            "expensive but weak",
        )
    )


def _is_forecast_quality_question(question_lower: str) -> bool:
    """Return True when the user asks about weak forecast quality by category."""
    return any(
        phrase in question_lower for phrase in PRODUCT_CATEGORY_INTENT_PHRASES
    ) and any(phrase in question_lower for phrase in FORECAST_QUALITY_INTENT_PHRASES)


def _is_time_trend_question(question_lower: str) -> bool:
    """Return True when the user asks for a monthly closing-date trend."""
    return any(
        phrase in question_lower
        for phrase in (
            "monthly pipeline trend",
            "pipeline by month",
            "revenue trend by closing date",
            "monthly deal amount trend",
            "monthly trend based on closing date",
            "trend based on closing date",
        )
    ) or (
        "trend" in question_lower
        and (
            "closing date" in question_lower
            or "close date" in question_lower
            or "month" in question_lower
        )
    )


def _answer_lead_source_value_vs_conversion(
    leads_df: pd.DataFrame,
    deals_df: pd.DataFrame,
) -> dict:
    """Answer cross-table lead-source conversion questions."""
    lead_required = {"Lead_ID", "Lead_Source", "Estimated_Value"}
    deal_required = {"Lead_ID"}
    if not lead_required.issubset(leads_df.columns) or not deal_required.issubset(
        deals_df.columns
    ):
        return _missing_data_answer(
            title="Lead Source Conversion Analysis Unavailable",
            message=(
                "Lead source value versus conversion analysis requires "
                "`Lead_ID`, `Lead_Source`, and `Estimated_Value` in leads data, plus "
                "`Lead_ID` in deals data."
            ),
            action="Upload leads and deals with lead IDs and lead source fields to enable this analysis.",
        )

    summary = get_lead_source_value_vs_conversion(leads_df, deals_df)
    if summary.empty:
        return _missing_data_answer(
            title="Lead Source Conversion Analysis Unavailable",
            message="There is not enough lead and deal data to compare value against conversion.",
            action="Load non-empty leads and deals data to compare source quality.",
        )

    average_conversion_rate = summary["Conversion_Rate"].mean()
    problematic_sources = summary[summary["Conversion_Rate"] < average_conversion_rate]
    if problematic_sources.empty:
        target_row = summary.iloc[0]
    else:
        target_row = problematic_sources.sort_values(
            ["Estimated_Value", "Conversion_Rate"],
            ascending=[False, True],
        ).iloc[0]

        return _make_answer(
            answer_type="metric",
            title="High Value but Low Conversion Lead Source",
            answer=(
                f"{target_row['Lead_Source']} has "
            f"{format_aed_compact(target_row['Estimated_Value'])} estimated lead value "
            f"but a conversion rate of {target_row['Conversion_Rate'] * 100:.1f}%, "
            f"below the average conversion rate of {average_conversion_rate * 100:.1f}%."
        ),
        data=summary.sort_values(
            ["Estimated_Value", "Conversion_Rate"],
            ascending=[False, True],
        ).head(5),
        recommended_action=(
            "Review qualification quality for this lead source before increasing sales "
            "focus or campaign spend."
        ),
        answer_source="lead conversion analysis",
    )


def analyze_product_forecast_quality(deals_df: pd.DataFrame) -> dict:
    """Analyze forecast-quality risk by product category using Pandas only."""
    required_columns = {"Product_Category", "Forecast_Category", "Amount"}
    if not required_columns.issubset(deals_df.columns):
        return _missing_data_answer(
            title="Forecast Quality by Product Category Unavailable",
            message=(
                "Forecast quality analysis requires `Product_Category`, "
                "`Forecast_Category`, and `Amount` in deals data."
            ),
            action="Upload deals with product category, forecast category, and amount fields to enable this analysis.",
        )

    summary = get_weakest_forecast_quality_by_product_category(deals_df)
    if summary.empty:
        return _missing_data_answer(
            title="Forecast Quality by Product Category Unavailable",
            message="There is not enough deal data to evaluate forecast quality by product category.",
            action="Load non-empty deals with product category and forecast data to enable this analysis.",
        )

    weakest_row = summary.iloc[0]
    return _make_answer(
        answer_type="metric",
        title="Weakest Forecast Quality by Product Category",
        answer=(
            f"{weakest_row['Product_Category']} has the weakest forecast quality: "
            f"{format_aed_compact(weakest_row['Weak_Forecast_Amount'])} weak forecast value, "
            f"representing {weakest_row['Weak_Forecast_Ratio'] * 100:.1f}% of its total category pipeline."
        ),
        data=summary,
        recommended_action=(
            "Review this category's At Risk, Omitted, and early Pipeline deals and move "
            "serious opportunities toward Commit or Closed."
        ),
        source="forecast_quality_analysis",
        answer_source="forecast quality analysis",
    )


def _answer_monthly_trend(question_lower: str, deals_df: pd.DataFrame) -> dict:
    """Answer monthly pipeline or revenue trend questions."""
    if not {"Closing_Date", "Amount"}.issubset(deals_df.columns):
        return _missing_data_answer(
            title="Monthly Trend Unavailable",
            message="Monthly trend analysis requires `Closing_Date` and `Amount` in deals data.",
            action="Upload deals with valid closing dates and amounts to enable monthly trend analysis.",
        )

    forecast_filter = "Pipeline" if "pipeline" in question_lower else None
    summary = get_monthly_amount_trend(deals_df, forecast_filter=forecast_filter)
    if summary.empty:
        return _missing_data_answer(
            title="Monthly Trend Unavailable",
            message="There is not enough valid closing-date data to build the monthly trend.",
            action="Check that `Closing_Date` values are valid dates and the dataset is not empty.",
        )

    if forecast_filter == "Pipeline":
        title = "Monthly Pipeline Trend by Closing Date"
        answer = "This chart shows pipeline deal amount grouped by closing month."
        action = "Review months with high expected pipeline value and confirm deal readiness."
    else:
        title = "Monthly Deal Amount Trend by Closing Date"
        answer = "This chart shows total deal amount grouped by closing month."
        action = "Review months with high expected closing value and confirm deal readiness."

    response = _make_answer(
        answer_type="chart",
        title=title,
        answer=answer,
        data=summary,
        recommended_action=action,
        chart_key="monthly_pipeline_trend",
        answer_source="time trend chart",
    )
    response["chart_type"] = "line"
    return response


def answer_quantitative_question(
    question: str,
    leads_df: pd.DataFrame,
    deals_df: pd.DataFrame,
) -> dict:
    """Compatibility wrapper around the unified semantic BI query engine."""
    from src.query_engine import answer_business_question

    return answer_business_question(question, leads_df=leads_df, deals_df=deals_df)
