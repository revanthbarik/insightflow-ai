"""Tests for the unified semantic BI query engine."""

from __future__ import annotations

import pandas as pd

import app
from src.followup import build_last_answer_context
from src.query_engine import answer_business_question
from src.query_parser import parse_business_question


def _sample_leads() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Lead_ID": ["L001", "L002", "L003", "L004", "L005"],
            "Lead_Source": ["Website Inquiry", "Partner Channel", "Partner Channel", "Referral", "Website Inquiry"],
            "Estimated_Value": [500_000, 300_000, 1_200_000, 250_000, 150_000],
            "Sales_Rep": ["Aisha Khan", "Omar Ali", "Omar Ali", "Sara Noor", "Aisha Khan"],
            "Region": ["UAE", "Saudi Arabia", "Saudi Arabia", "Oman", "Dubai"],
            "Customer_Type": ["EPC", "Distributor", "Installer", "Government Client", "Commercial Client"],
        }
    )


def _sample_deals() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Deal_ID": ["D001", "D002", "D003", "D004", "D005"],
            "Lead_ID": ["L001", "L001", "L002", "L003", "L005"],
            "Deal_Stage": ["Closed Won", "Closed Lost", "Negotiation", "Proposal", "Closed Won"],
            "Amount": [1_000_000, 500_000, 200_000, 700_000, 350_000],
            "Margin": [150_000, 60_000, 30_000, 90_000, 50_000],
            "Closing_Date": pd.to_datetime(["2024-06-01", "2024-07-01", "2024-08-01", "2024-09-01", "2024-11-01"]),
            "Forecast_Category": ["Closed", "Omitted", "Pipeline", "At Risk", "Pipeline"],
            "Product_Category": ["Panels", "Panels", "Batteries", "Batteries", "Panels"],
        }
    )


def test_show_revenue_by_region_works():
    result = answer_business_question("Show revenue by region", _sample_leads(), _sample_deals())
    assert result["answer_type"] == "chart"
    assert result["title"] == "Revenue by Region"
    assert result["x_axis"] == "Region"
    assert result["y_axis"] == "Total_Amount"
    assert result["data"] is not None
    assert not result["data"].empty


def test_percentage_distribution_question_returns_share_and_full_distribution():
    result = answer_business_question(
        "Which region contributes the highest percentage of our total revenue, and can you visualize the distribution across all regions?",
        _sample_leads(),
        _sample_deals(),
    )
    assert result["answer_type"] == "chart"
    assert result["metric_type"] == "revenue"
    assert result["x_axis"] == "Region"
    assert "representing" in result["answer"]
    assert "Share_Percentage" in result["data"].columns
    assert len(result["data"]) >= 3


def test_new_bi_style_questions_work():
    leads_df = _sample_leads()
    deals_df = _sample_deals()
    questions = [
        "Revenue by product category",
        "Pipeline by sales rep",
        "Average deal size by region",
        "Deal count by stage",
        "Margin by product category",
        "Win rate by lead source",
        "Top 5 regions by revenue",
        "Which region has the highest revenue",
        "Show revenue trend by month",
        "Show pipeline by forecast category",
    ]
    for question in questions:
        result = answer_business_question(question, leads_df, deals_df)
        assert result["answer_type"] != "unsupported", question


def test_compound_customer_type_performance_question_uses_joined_pandas_analytics():
    result = answer_business_question(
        "Compare the performance of each customer type based on total revenue and average deal size, then identify which customer segment deserves the highest management priority.",
        _sample_leads(),
        _sample_deals(),
    )
    assert result["answer_type"] != "unsupported"
    assert result["title"] == "Customer Type Performance"
    assert result["answer_source"] == "semantic BI calculation"
    assert result["query_debug"]["requires_join"] is True
    assert result["query_debug"]["join_key"] == "Lead_ID"
    assert result["query_debug"]["metrics"] == ["revenue", "average_deal_size"]
    assert {"Customer_Type", "Total_Revenue", "Average_Deal_Size", "Deal_Count", "Priority_Rank"}.issubset(result["data"].columns)
    assert "deserves the highest management priority" in result["answer"]


def test_synonym_parsing_for_revenue_and_sales_rep_terms():
    revenue_query = parse_business_question("show sales by area")
    rep_query = parse_business_question("who is the best salesman")
    owner_query = parse_business_question("pipeline by owner")
    assert revenue_query.metric_name == "revenue"
    assert revenue_query.dimension == "Region"
    assert rep_query.dimension == "Sales_Rep"
    assert owner_query.dimension == "Sales_Rep"


def test_ranking_synonym_parsing():
    worst_query = parse_business_question("rank the weakest products", active_context={"metric": "revenue", "dataset": "deals"})
    best_query = parse_business_question("show the strongest regions", active_context={"metric": "revenue", "dataset": "deals"})
    assert worst_query.sort_order == "asc"
    assert worst_query.dimension == "Product_Category"
    assert best_query.sort_order == "desc"
    assert best_query.dimension == "Region"


def test_missing_required_column_returns_helpful_error():
    deals_df = _sample_deals().drop(columns=["Margin"])
    result = answer_business_question("Margin by product category", _sample_leads(), deals_df)
    assert result["answer_type"] == "unsupported"
    assert "required column/table is missing" in result["answer"]
    assert "Margin" in result["answer"]


def test_unknown_metric_returns_helpful_unsupported_message():
    result = answer_business_question("Show employee happiness by office", _sample_leads(), _sample_deals())
    assert result["answer_type"] == "unsupported"
    assert "Examples:" in result["answer"]


def test_empty_dataframe_does_not_crash():
    result = answer_business_question(
        "Show revenue by region",
        pd.DataFrame(columns=["Lead_ID", "Region"]),
        pd.DataFrame(columns=["Deal_ID", "Lead_ID", "Amount"]),
    )
    assert result["answer_type"] == "unsupported"


def test_chart_context_created_for_grouped_question():
    result = answer_business_question("Show revenue by region", _sample_leads(), _sample_deals())
    assert result["chart_title"] == "Revenue by Region"
    assert result["table_preview"]
    assert result["currency"] == "AED"


def test_monthly_trend_fills_missing_months_with_zero():
    result = answer_business_question("Show revenue trend by month", _sample_leads(), _sample_deals())
    assert result["title"] == "Monthly Deal Amount Trend by Closing Date"
    assert result["data"]["Closing_Month"].tolist() == ["Jun 2024", "Jul 2024", "Aug 2024", "Sep 2024", "Oct 2024", "Nov 2024"]
    assert result["data"]["Total_Amount"].tolist()[4] == 0.0


def test_forecast_quality_uses_weak_forecast_ratio():
    result = answer_business_question("Which product category has weakest forecast quality", _sample_leads(), _sample_deals())
    assert result["title"] == "Weakest Forecast Quality by Product Category"
    assert "Batteries" in result["answer"]


def test_lead_source_conversion_joins_by_lead_id():
    result = answer_business_question(
        "Which lead source generates highest estimated value but low conversion",
        _sample_leads(),
        _sample_deals(),
    )
    assert result["answer_source"] == "lead conversion analysis"
    assert "Conversion_Rate" in result["data"].columns


def test_resolve_question_response_does_not_call_rag_for_analytics(monkeypatch):
    monkeypatch.setattr(app, "query_rag_store", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("RAG should not be called")))
    monkeypatch.setattr(app, "synthesize_rag_answer", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Ollama synthesis should not be called")))
    result = app.resolve_question_response(
        "Show revenue by region",
        _sample_leads(),
        _sample_deals(),
        auto_insights=[],
        rag_is_current=False,
    )
    assert result["answer_type"] == "chart"


def test_followup_last_answer_context_works_with_query_engine_result():
    result = answer_business_question("Show revenue by region", _sample_leads(), _sample_deals())
    context = build_last_answer_context(
        original_user_question="Show revenue by region",
        result=result,
        leads_df=_sample_leads(),
        deals_df=_sample_deals(),
        data_signature=("demo",),
    )
    assert context["answer_title"] == "Revenue by Region"
    assert context["chart_title"] == "Revenue by Region"


def test_contextual_followup_parser_inherits_metric_and_limit():
    parsed = parse_business_question(
        "give me 3 product categories that are doing the worst",
        active_context={"metric": "revenue", "dataset": "deals", "dimension": "Region", "aggregation": "sum", "filters": {}},
    )
    assert parsed.metric_name == "revenue"
    assert parsed.dimension == "Product_Category"
    assert parsed.sort_order == "asc"
    assert parsed.limit == 3


def test_contextual_followup_variants_parse_from_active_context():
    active_context = {"metric": "revenue", "dataset": "deals", "dimension": "Region", "aggregation": "sum", "filters": {}}
    assert parse_business_question("now show bottom 5 regions", active_context=active_context).limit == 5
    assert parse_business_question("compare this by sales rep", active_context=active_context).dimension == "Sales_Rep"
    assert parse_business_question("show it by product category", active_context=active_context).dimension == "Product_Category"
    filtered = parse_business_question("filter to Dubai", active_context=active_context)
    assert filtered.metric_name == "revenue"
    assert filtered.filters == {"Region": "Dubai"}


def test_compound_contextual_followup_inherits_metric_and_join_dimension():
    parsed = parse_business_question(
        "show it by customer type",
        active_context={"metrics": ["revenue", "average_deal_size"], "dataset": "deals", "dimension": "Region", "aggregation": "sum", "filters": {}},
    )
    assert parsed.metric_name == "revenue"
    assert parsed.dimensions[0] == "Customer_Type"
    assert parsed.requires_join is True
    assert parsed.join_key == "Lead_ID"
