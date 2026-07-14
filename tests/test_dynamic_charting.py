"""Unit tests for deterministic dynamic chart inference and generation."""

from __future__ import annotations

import pandas as pd

from src.dynamic_charting import (
    build_unsupported_chart_response,
    detect_chart_intent,
    generate_dynamic_chart,
    infer_chart_spec,
)


def _sample_leads() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Lead_ID": ["L001", "L002", "L003"],
            "Company_Name": ["Alpha Solar", "Beta Solar", "Gamma Energy"],
            "Lead_Source": ["Website Inquiry", "Partner Channel", "Website Inquiry"],
            "Industry": ["Solar EPC", "Commercial", "Utilities"],
            "Estimated_Value": [500_000, 300_000, 450_000],
            "Status": ["Qualified", "Contacted", "Qualified"],
            "Created_Date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
            "Sales_Rep": ["Aisha Khan", "Omar Ali", "Aisha Khan"],
            "Region": ["UAE", "Saudi Arabia", "UAE"],
            "Product_Interest": ["Panels", "Inverters", "Panels"],
        }
    )


def _sample_deals() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Deal_ID": ["D001", "D002", "D003"],
            "Lead_ID": ["L001", "L002", "L003"],
            "Company_Name": ["Alpha Solar", "Beta Solar", "Gamma Energy"],
            "Deal_Stage": ["Closed Won", "Negotiation", "Proposal"],
            "Amount": [1_000_000, 200_000, 600_000],
            "Closing_Date": pd.to_datetime(["2024-06-01", "2024-06-15", "2024-07-01"]),
            "Forecast_Category": ["Closed", "Pipeline", "Pipeline"],
            "Product_Category": ["Panels", "Inverters", "Panels"],
            "Region": ["UAE", "Saudi Arabia", "UAE"],
        }
    )


def test_detect_chart_intent_for_dynamic_request():
    assert detect_chart_intent("Show revenue by region") is True
    assert detect_chart_intent("Show me leads by source") is False


def test_infer_chart_spec_for_revenue_by_region():
    spec = infer_chart_spec("Show revenue by region", _sample_leads(), _sample_deals())
    assert spec is not None
    assert spec["dataset"] == "deals"
    assert spec["x_column"] == "Region"
    assert spec["y_column"] == "Amount"
    assert spec["aggregation"] == "sum"


def test_infer_chart_spec_for_average_deal_size_by_product_category():
    spec = infer_chart_spec(
        "Show average deal size by product category",
        _sample_leads(),
        _sample_deals(),
    )
    assert spec is not None
    assert spec["dataset"] == "deals"
    assert spec["x_column"] == "Product_Category"
    assert spec["y_column"] == "Amount"
    assert spec["aggregation"] == "mean"


def test_infer_chart_spec_for_leads_by_source():
    spec = infer_chart_spec("Plot leads by source", _sample_leads(), _sample_deals())
    assert spec is not None
    assert spec["dataset"] == "leads"
    assert spec["x_column"] == "Lead_Source"
    assert spec["aggregation"] == "count"


def test_generate_dynamic_chart_handles_missing_optional_columns_safely():
    deals_df = _sample_deals().drop(columns=["Product_Category"])
    spec = infer_chart_spec(
        "Show average deal size by product category",
        _sample_leads(),
        deals_df,
    )
    assert spec is not None
    result = generate_dynamic_chart(spec, _sample_leads(), deals_df)
    assert result["answer_type"] == "unsupported"
    assert "missing" in result["answer"].lower()


def test_unsupported_chart_response_is_useful():
    result = build_unsupported_chart_response(_sample_leads(), _sample_deals())
    assert result["answer_type"] == "unsupported"
    assert "safe dynamic charts" in result["answer"]
    assert "Available leads fields" in result["recommended_action"]


def test_generate_dynamic_chart_for_revenue_by_region():
    spec = infer_chart_spec("Show revenue by region", _sample_leads(), _sample_deals())
    result = generate_dynamic_chart(spec, _sample_leads(), _sample_deals())
    assert result["answer_type"] == "dynamic_chart"
    assert result["title"] == "Revenue by Region"
    assert not result["data"].empty
