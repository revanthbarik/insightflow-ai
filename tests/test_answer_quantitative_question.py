"""Unit tests for answer_quantitative_question."""

import pandas as pd

from src.analytics import (
    UNSUPPORTED_ANSWER_MESSAGE,
    answer_quantitative_question,
)


def _sample_leads() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Lead_ID": ["L001", "L002", "L003", "L004"],
            "Company_Name": ["Alpha Solar", "Beta Solar", "Gamma Storage", "Delta Grid"],
            "Lead_Source": ["Website Inquiry", "Partner Channel", "Partner Channel", "Referral"],
            "Industry": [
                "Solar EPC Contractor",
                "Commercial & Industrial",
                "Battery Storage",
                "Utilities",
            ],
            "Estimated_Value": [500_000, 300_000, 1_200_000, 250_000],
            "Status": ["Qualified", "Contacted", "Qualified", "New"],
            "Created_Date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"]),
            "Sales_Rep": ["Aisha Khan", "Omar Ali", "Omar Ali", "Sara Noor"],
        }
    )


def _sample_deals() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Deal_ID": ["D001", "D002", "D003", "D004"],
            "Lead_ID": ["L001", "L001", "L002", "L002"],
            "Company_Name": ["Alpha Solar", "Alpha Solar", "Beta Solar", "Beta Solar"],
            "Deal_Stage": ["Closed Won", "Closed Lost", "Negotiation", "Proposal"],
            "Amount": [1_000_000, 500_000, 200_000, 700_000],
            "Closing_Date": pd.to_datetime(["2024-06-01", "2024-07-01", "2024-08-01", "2024-09-01"]),
            "Forecast_Category": ["Closed", "Omitted", "Pipeline", "At Risk"],
            "Product_Category": ["Panels", "Panels", "Batteries", "Batteries"],
        }
    )


def test_pipeline_value_question():
    result = answer_quantitative_question(
        "What is the total pipeline value?",
        _sample_leads(),
        _sample_deals(),
    )
    assert result["answer_type"] == "metric"
    assert result["title"] == "Total Pipeline Value"
    assert result["answer"] == "AED 200.00K"


def test_deals_by_stage_question():
    result = answer_quantitative_question(
        "Show me deals by stage",
        _sample_leads(),
        _sample_deals(),
    )
    assert result["answer_type"] == "table"
    assert result["title"] == "Deals by Stage"
    assert result["data"] is not None
    assert "Deal_Stage" in result["data"].columns


def test_leads_by_source_table_question():
    result = answer_quantitative_question(
        "Show me leads by source",
        _sample_leads(),
        _sample_deals(),
    )
    assert result["answer_type"] == "table"
    assert result["title"] == "Leads by Source"
    assert result["data"] is not None
    assert "Lead_Source" in result["data"].columns


def test_total_leads_metric_question():
    result = answer_quantitative_question(
        "How many leads do we have?",
        _sample_leads(),
        _sample_deals(),
    )
    assert result["answer_type"] == "metric"
    assert result["title"] == "Total Leads"
    assert result["answer"] == "4"


def test_unsupported_question():
    result = answer_quantitative_question(
        "Tell me about the weather today",
        _sample_leads(),
        _sample_deals(),
    )
    assert result["answer_type"] == "unsupported"
    assert UNSUPPORTED_ANSWER_MESSAGE in result["answer"]
    assert "Examples:" in result["answer"]
    assert "Available Leads fields:" in result["answer"]


def test_missing_optional_columns_for_product_and_region():
    leads_df = _sample_leads()
    deals_df = _sample_deals().drop(columns=["Product_Category"])

    product_result = answer_quantitative_question(
        "Which product category has the most revenue?",
        leads_df,
        deals_df,
    )
    assert product_result["answer"] == "N/A"
    assert "Product category data is not available" in product_result["recommended_action"]

    region_result = answer_quantitative_question(
        "Which region has the most opportunity value?",
        leads_df,
        deals_df,
    )
    assert region_result["answer"] == "N/A"
    assert "Region data is not available" in region_result["recommended_action"]


def test_forecast_quality_question_routes_before_top_product_revenue():
    result = answer_quantitative_question(
        "Which product category has the weakest forecast quality?",
        _sample_leads(),
        _sample_deals(),
    )
    assert result["answer_type"] == "metric"
    assert result["title"] == "Weakest Forecast Quality by Product Category"
    assert "Batteries" in result["answer"]
    assert "weakest forecast quality" in result["answer"].lower()
    assert "Top Product Category by Revenue" not in result["answer"]
    assert result["source"] == "forecast_quality_analysis"
    assert result["answer_source"] == "forecast quality analysis"


def test_highest_at_risk_forecast_routes_to_forecast_quality_analysis():
    result = answer_quantitative_question(
        "Which product category has the highest at-risk forecast?",
        _sample_leads(),
        _sample_deals(),
    )
    assert result["title"] == "Weakest Forecast Quality by Product Category"
    assert result["source"] == "forecast_quality_analysis"
    assert "Top Product Category by Revenue" not in result["answer"]


def test_most_weak_pipeline_routes_to_forecast_quality_analysis():
    result = answer_quantitative_question(
        "Which category has the most weak pipeline?",
        _sample_leads(),
        _sample_deals(),
    )
    assert result["title"] == "Weakest Forecast Quality by Product Category"
    assert result["source"] == "forecast_quality_analysis"
    assert "Top Product Category by Revenue" not in result["answer"]


def test_monthly_pipeline_trend_routes_before_total_pipeline_value():
    result = answer_quantitative_question(
        "Show monthly pipeline trend based on closing date.",
        _sample_leads(),
        _sample_deals(),
    )
    assert result["answer_type"] == "chart"
    assert result["title"] == "Monthly Pipeline Trend by Closing Date"
    assert result["chart_key"] == "monthly_pipeline_trend"
    assert "Closing_Month" in result["data"].columns
    assert result["answer_source"] == "time trend chart"


def test_monthly_pipeline_trend_includes_continuous_month_range():
    deals_df = pd.DataFrame(
        {
            "Deal_ID": ["D001", "D002"],
            "Lead_ID": ["L001", "L002"],
            "Company_Name": ["Alpha Solar", "Beta Solar"],
            "Deal_Stage": ["Negotiation", "Proposal"],
            "Amount": [250_000, 400_000],
            "Closing_Date": pd.to_datetime(["2024-06-01", "2024-08-01"]),
            "Forecast_Category": ["Pipeline", "Pipeline"],
            "Product_Category": ["Panels", "Batteries"],
        }
    )
    result = answer_quantitative_question(
        "Show monthly pipeline trend based on closing date.",
        _sample_leads(),
        deals_df,
    )
    assert result["title"] == "Monthly Pipeline Trend by Closing Date"
    assert result["answer"] != "AED 650.00K"
    assert result["data"]["Closing_Month"].tolist() == ["Jun 2024", "Jul 2024", "Aug 2024"]
    assert result["data"]["Total_Amount"].tolist() == [250_000, 0.0, 400_000]


def test_lead_source_high_value_low_conversion_routes_before_top_source():
    result = answer_quantitative_question(
        "Which lead source generates the highest estimated value but low conversion into deals?",
        _sample_leads(),
        _sample_deals(),
    )
    assert result["answer_type"] == "metric"
    assert result["title"] == "High Value but Low Conversion Lead Source"
    assert "conversion rate" in result["answer"].lower()
    assert result["title"] != "Top Lead Source by Estimated Value"
    assert result["answer_source"] == "lead conversion analysis"
    assert result["data"] is not None
    assert result["data"].columns.tolist() == [
        "Lead_Source",
        "Total_Leads",
        "Converted_Leads",
        "Estimated_Value",
        "Avg_Estimated_Value",
        "Conversion_Rate",
    ]


def test_chart_intent_detection():
    result = answer_quantitative_question(
        "Create a chart of deals by stage",
        _sample_leads(),
        _sample_deals(),
    )
    assert result["answer_type"] == "chart"
    assert result["chart_key"] == "deals_by_stage"
    assert result["data"] is not None


def test_chart_intent_detection_with_visualise_keyword():
    result = answer_quantitative_question(
        "Visualise forecast summary",
        _sample_leads(),
        _sample_deals(),
    )
    assert result["answer_type"] == "chart"
    assert result["chart_key"] == "forecast_summary"


def test_unsupported_chart_question_returns_helpful_message():
    result = answer_quantitative_question(
        "Create a chart of customer satisfaction",
        _sample_leads(),
        _sample_deals(),
    )
    assert result["answer_type"] == "unsupported"
    assert result["title"] == "Unsupported Chart Request"
    assert "I can currently create charts" in result["answer"]


def test_product_category_chart_requires_optional_column():
    result = answer_quantitative_question(
        "Create a chart of product category revenue",
        _sample_leads(),
        _sample_deals().drop(columns=["Product_Category"]),
    )
    assert result["answer_type"] == "unsupported"
    assert result["title"] == "Product Category Revenue Chart Unavailable"


def test_region_pipeline_chart_requires_optional_column():
    result = answer_quantitative_question(
        "Show region-wise pipeline chart",
        _sample_leads(),
        _sample_deals(),
    )
    assert result["answer_type"] == "unsupported"
    assert result["title"] == "Region Pipeline Chart Unavailable"


def test_missing_product_category_for_forecast_quality_returns_safe_response():
    result = answer_quantitative_question(
        "Which product category has the most at-risk forecast?",
        _sample_leads(),
        _sample_deals().drop(columns=["Product_Category"]),
    )
    assert result["answer_type"] == "unsupported"
    assert result["title"] == "Forecast Quality by Product Category Unavailable"


def test_missing_closing_date_for_monthly_trend_returns_safe_response():
    result = answer_quantitative_question(
        "Show monthly pipeline trend based on closing date",
        _sample_leads(),
        _sample_deals().drop(columns=["Closing_Date"]),
    )
    assert result["answer_type"] == "unsupported"
    assert result["title"] == "Monthly Trend Unavailable"


def test_invalid_closing_date_for_monthly_trend_returns_safe_response():
    deals_df = _sample_deals().copy()
    deals_df["Closing_Date"] = ["invalid", "bad", "still bad", "also bad"]
    result = answer_quantitative_question(
        "Show monthly pipeline trend based on closing date",
        _sample_leads(),
        deals_df,
    )
    assert result["answer_type"] == "unsupported"
    assert result["title"] == "Monthly Trend Unavailable"


def test_missing_lead_id_for_conversion_question_returns_safe_response():
    result = answer_quantitative_question(
        "Which lead source has high lead value but poor conversion?",
        _sample_leads().drop(columns=["Lead_ID"]),
        _sample_deals(),
    )
    assert result["answer_type"] == "unsupported"
    assert result["title"] == "Lead Source Conversion Analysis Unavailable"


def test_empty_dataframes_do_not_crash_for_new_intents():
    empty_leads = pd.DataFrame(columns=["Lead_ID", "Lead_Source", "Estimated_Value"])
    empty_deals = pd.DataFrame(
        columns=["Lead_ID", "Product_Category", "Forecast_Category", "Amount", "Closing_Date"]
    )

    forecast_result = answer_quantitative_question(
        "Which product category has the weakest forecast quality?",
        empty_leads,
        empty_deals,
    )
    trend_result = answer_quantitative_question(
        "Show monthly pipeline trend based on closing date",
        empty_leads,
        empty_deals,
    )
    conversion_result = answer_quantitative_question(
        "Which lead source generates the highest estimated value but low conversion into deals?",
        empty_leads,
        empty_deals,
    )

    assert forecast_result["answer_type"] == "unsupported"
    assert trend_result["answer_type"] == "unsupported"
    assert conversion_result["answer_type"] == "unsupported"
