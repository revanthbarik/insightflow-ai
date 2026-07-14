"""Unit tests for generate_auto_insights."""

import pandas as pd

from src.analytics import generate_auto_insights


def _sample_leads() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Lead_ID": ["L001", "L002"],
            "Company_Name": ["Alpha Solar", "Beta Solar"],
            "Lead_Source": ["Website Inquiry", "Partner Channel"],
            "Industry": ["Solar EPC Contractor", "Commercial & Industrial"],
            "Estimated_Value": [500_000, 300_000],
            "Status": ["Qualified", "Contacted"],
            "Created_Date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            "Sales_Rep": ["Aisha Khan", "Omar Ali"],
        }
    )


def _sample_deals() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Deal_ID": ["D001", "D002", "D003"],
            "Lead_ID": ["L001", "L001", "L002"],
            "Company_Name": ["Alpha Solar", "Alpha Solar", "Beta Solar"],
            "Deal_Stage": ["Closed Won", "Closed Lost", "Negotiation"],
            "Amount": [1_000_000, 500_000, 200_000],
            "Closing_Date": pd.to_datetime(["2024-06-01", "2024-07-01", "2024-08-01"]),
            "Forecast_Category": ["Closed", "Omitted", "Pipeline"],
        }
    )


def test_generate_auto_insights_returns_list():
    insights = generate_auto_insights(_sample_leads(), _sample_deals())
    assert isinstance(insights, list)
    assert len(insights) > 0
    assert all(
        {"title", "metric", "explanation", "recommended_action"} <= insight.keys()
        for insight in insights
    )


def test_generate_auto_insights_includes_core_metrics():
    insights = generate_auto_insights(_sample_leads(), _sample_deals())
    titles = [insight["title"] for insight in insights]
    assert "Total Open Pipeline Value" in titles
    assert "Average Deal Size" in titles


def test_generate_auto_insights_handles_missing_optional_columns():
    leads_df = _sample_leads()
    deals_df = _sample_deals()
    insights = generate_auto_insights(leads_df, deals_df)
    titles = [insight["title"] for insight in insights]
    assert "Top Product Category" not in titles
    assert "Top Region by Deal Amount" not in titles


def test_generate_auto_insights_handles_empty_deals_safely():
    leads_df = _sample_leads()
    deals_df = pd.DataFrame(
        columns=[
            "Deal_ID",
            "Lead_ID",
            "Company_Name",
            "Deal_Stage",
            "Amount",
            "Closing_Date",
            "Forecast_Category",
        ]
    )
    insights = generate_auto_insights(leads_df, deals_df)
    assert isinstance(insights, list)
    assert len(insights) > 0

    average_deal = next(
        insight for insight in insights if insight["title"] == "Average Deal Size"
    )
    assert average_deal["metric"] == "N/A"


def test_generate_auto_insights_handles_small_data_safely():
    leads_df = _sample_leads().head(1)
    deals_df = _sample_deals().head(1)
    insights = generate_auto_insights(leads_df, deals_df)
    assert isinstance(insights, list)
    assert len(insights) > 0


def test_generate_auto_insights_handles_empty_leads_safely():
    leads_df = pd.DataFrame(
        columns=[
            "Lead_ID",
            "Company_Name",
            "Lead_Source",
            "Industry",
            "Estimated_Value",
            "Status",
            "Created_Date",
            "Sales_Rep",
        ]
    )
    insights = generate_auto_insights(leads_df, _sample_deals())
    assert isinstance(insights, list)
    assert len(insights) > 0
