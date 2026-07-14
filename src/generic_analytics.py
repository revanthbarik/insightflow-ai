"""Reusable Pandas analytics primitives for the semantic BI engine."""

from __future__ import annotations

import pandas as pd

WEAK_FORECAST_CATEGORIES = {"At Risk", "Omitted", "Pipeline"}
STRONG_FORECAST_CATEGORIES = {"Commit", "Closed", "Best Case"}


def group_sum(df: pd.DataFrame, value_col: str, dimension_col: str) -> pd.DataFrame:
    if df.empty or not {value_col, dimension_col}.issubset(df.columns):
        return pd.DataFrame(columns=[dimension_col, "Value"])
    grouped = (
        df.dropna(subset=[dimension_col])
        .groupby(dimension_col, as_index=False)
        .agg(Value=(value_col, "sum"))
        .sort_values("Value", ascending=False)
    )
    return grouped


def group_average(df: pd.DataFrame, value_col: str, dimension_col: str) -> pd.DataFrame:
    if df.empty or not {value_col, dimension_col}.issubset(df.columns):
        return pd.DataFrame(columns=[dimension_col, "Value"])
    grouped = (
        df.dropna(subset=[dimension_col])
        .groupby(dimension_col, as_index=False)
        .agg(Value=(value_col, "mean"))
        .sort_values("Value", ascending=False)
    )
    return grouped


def group_count(df: pd.DataFrame, id_col: str, dimension_col: str) -> pd.DataFrame:
    if df.empty or not {id_col, dimension_col}.issubset(df.columns):
        return pd.DataFrame(columns=[dimension_col, "Value"])
    grouped = (
        df.dropna(subset=[dimension_col])
        .groupby(dimension_col, as_index=False)
        .agg(Value=(id_col, "count"))
        .sort_values("Value", ascending=False)
    )
    return grouped


def top_n(df: pd.DataFrame, value_col: str = "Value", n: int = 5) -> pd.DataFrame:
    if df.empty or value_col not in df.columns:
        return df
    return df.sort_values(value_col, ascending=False).head(n)


def bottom_n(df: pd.DataFrame, value_col: str = "Value", n: int = 5) -> pd.DataFrame:
    if df.empty or value_col not in df.columns:
        return df
    return df.sort_values(value_col, ascending=True).head(n)


def time_trend(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    freq: str = "M",
) -> pd.DataFrame:
    if df.empty or not {date_col, value_col}.issubset(df.columns):
        return pd.DataFrame(columns=["Period_Start", "Closing_Month", "Total_Amount"])
    working_df = df.copy()
    working_df[date_col] = pd.to_datetime(working_df[date_col], errors="coerce", format="mixed")
    working_df = working_df.dropna(subset=[date_col])
    if working_df.empty:
        return pd.DataFrame(columns=["Period_Start", "Closing_Month", "Total_Amount"])
    working_df["Period_Start"] = working_df[date_col].dt.to_period(freq).dt.to_timestamp()
    summary = (
        working_df.groupby("Period_Start", as_index=False)
        .agg(Total_Amount=(value_col, "sum"))
        .sort_values("Period_Start")
    )
    full_range = pd.date_range(summary["Period_Start"].min(), summary["Period_Start"].max(), freq="MS")
    merged = (
        pd.DataFrame({"Period_Start": full_range})
        .merge(summary, on="Period_Start", how="left")
        .fillna({"Total_Amount": 0.0})
    )
    merged["Closing_Month"] = merged["Period_Start"].dt.strftime("%b %Y")
    return merged


def win_rate(df: pd.DataFrame, dimension_col: str | None = None) -> pd.DataFrame:
    required = {"Deal_Stage"}
    if df.empty or not required.issubset(df.columns):
        columns = [dimension_col, "Closed_Won", "Closed_Lost", "Win_Rate"] if dimension_col else ["Closed_Won", "Closed_Lost", "Win_Rate"]
        return pd.DataFrame(columns=[column for column in columns if column is not None])
    working_df = df[df["Deal_Stage"].isin(["Closed Won", "Closed Lost"])].copy()
    if working_df.empty:
        columns = [dimension_col, "Closed_Won", "Closed_Lost", "Win_Rate"] if dimension_col else ["Closed_Won", "Closed_Lost", "Win_Rate"]
        return pd.DataFrame(columns=[column for column in columns if column is not None])
    if dimension_col:
        grouped = (
            working_df.groupby(dimension_col, as_index=False)
            .agg(
                Closed_Won=("Deal_Stage", lambda values: (values == "Closed Won").sum()),
                Closed_Lost=("Deal_Stage", lambda values: (values == "Closed Lost").sum()),
            )
        )
    else:
        grouped = pd.DataFrame(
            [
                {
                    "Closed_Won": int((working_df["Deal_Stage"] == "Closed Won").sum()),
                    "Closed_Lost": int((working_df["Deal_Stage"] == "Closed Lost").sum()),
                }
            ]
        )
    total_closed = (grouped["Closed_Won"] + grouped["Closed_Lost"]).replace(0, pd.NA)
    grouped["Win_Rate"] = ((grouped["Closed_Won"] / total_closed) * 100).fillna(0.0)
    return grouped.sort_values("Win_Rate", ascending=False)


def forecast_quality(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"Product_Category", "Forecast_Category", "Amount"}
    if df.empty or not required_columns.issubset(df.columns):
        return pd.DataFrame(
            columns=[
                "Product_Category",
                "Total_Amount",
                "Weak_Forecast_Amount",
                "Strong_Forecast_Amount",
                "Weak_Forecast_Ratio",
            ]
        )
    working_df = df.dropna(subset=["Product_Category"]).copy()
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
    total_amount = summary["Total_Amount"].replace(0, pd.NA)
    summary["Weak_Forecast_Ratio"] = (summary["Weak_Forecast_Amount"] / total_amount).fillna(0.0)
    return summary.sort_values(["Weak_Forecast_Ratio", "Weak_Forecast_Amount"], ascending=[False, False])


def lead_source_conversion(leads_df: pd.DataFrame, deals_df: pd.DataFrame) -> pd.DataFrame:
    lead_required = {"Lead_ID", "Lead_Source", "Estimated_Value"}
    deal_required = {"Lead_ID"}
    if leads_df.empty or deals_df.empty or not lead_required.issubset(leads_df.columns) or not deal_required.issubset(deals_df.columns):
        return pd.DataFrame(
            columns=[
                "Lead_Source",
                "Total_Leads",
                "Converted_Leads",
                "Estimated_Value",
                "Avg_Estimated_Value",
                "Conversion_Rate",
            ]
        )
    converted_ids = deals_df["Lead_ID"].dropna().unique()
    working_df = leads_df.copy()
    working_df["Converted"] = working_df["Lead_ID"].isin(converted_ids)
    summary = working_df.groupby("Lead_Source", as_index=False).agg(
        Total_Leads=("Lead_ID", "nunique"),
        Converted_Leads=("Converted", "sum"),
        Estimated_Value=("Estimated_Value", "sum"),
        Avg_Estimated_Value=("Estimated_Value", "mean"),
    )
    total_leads = summary["Total_Leads"].replace(0, pd.NA)
    summary["Conversion_Rate"] = (summary["Converted_Leads"] / total_leads).fillna(0.0)
    return summary.sort_values(["Estimated_Value", "Conversion_Rate"], ascending=[False, True])
