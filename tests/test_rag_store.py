"""Unit tests for local ChromaDB RAG document building and safe querying."""

from __future__ import annotations

import pandas as pd

from src import rag_store


def _sample_leads() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Lead_ID": ["L001", "L002", "L003"],
            "Company_Name": ["Alpha Solar", "Beta Solar", "Gamma Energy"],
            "Lead_Source": ["Website Inquiry", "Partner Channel", "Referral"],
            "Industry": ["Solar EPC", "Commercial", "Utility"],
            "Estimated_Value": [500_000, 300_000, 450_000],
            "Status": ["Qualified", "Contacted", "Qualified"],
            "Created_Date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
            "Sales_Rep": ["Aisha Khan", "Omar Ali", "Sara Noor"],
            "Region": ["UAE", "Saudi Arabia", "Oman"],
        }
    )


def _sample_deals() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Deal_ID": ["D001", "D002", "D003", "D004"],
            "Lead_ID": ["L001", "L002", "L003", "L002"],
            "Company_Name": ["Alpha Solar", "Beta Solar", "Gamma Energy", "Beta Solar"],
            "Deal_Stage": ["Closed Won", "Negotiation", "Proposal", "PO Received"],
            "Amount": [1_000_000, 200_000, 600_000, 350_000],
            "Closing_Date": pd.to_datetime(["2024-06-01", "2024-07-01", "2024-08-01", "2024-08-15"]),
            "Forecast_Category": ["Closed", "Pipeline", "At Risk", "Commit"],
            "Product_Category": ["Panels", "Inverters", "Batteries", "Inverters"],
            "Region": ["UAE", "Saudi Arabia", "Oman", "Saudi Arabia"],
        }
    )


def _sample_insights() -> list[dict]:
    return [
        {
            "title": "At-Risk Pipeline Value",
            "metric": "AED 600.00K",
            "explanation": "At Risk deals total AED 600.00K.",
            "recommended_action": "Review at-risk deals with account owners.",
        }
    ]


def test_build_rag_documents_returns_non_empty_list():
    documents = rag_store.build_rag_documents(
        _sample_leads(),
        _sample_deals(),
        auto_insights=_sample_insights(),
    )
    assert isinstance(documents, list)
    assert len(documents) > 0


def test_build_rag_documents_include_schema_documents():
    documents = rag_store.build_rag_documents(_sample_leads(), _sample_deals())
    texts = [document["text"] for document in documents]
    assert any("Lead_ID" in text for text in texts)
    assert any("Deals represent active or closed commercial opportunities" in text for text in texts)


def test_build_rag_documents_include_forecast_category_definitions():
    documents = rag_store.build_rag_documents(_sample_leads(), _sample_deals())
    topics = [document["metadata"]["topic"] for document in documents]
    assert "Pipeline" in topics
    assert "At Risk" in topics


def test_build_rag_documents_include_deal_stage_definitions():
    documents = rag_store.build_rag_documents(_sample_leads(), _sample_deals())
    topics = [document["metadata"]["topic"] for document in documents]
    assert "PO Received" in topics
    assert "Closed Won" in topics


def test_build_rag_documents_include_metric_definitions():
    documents = rag_store.build_rag_documents(_sample_leads(), _sample_deals())
    texts = [document["text"] for document in documents]
    assert any("Monthly pipeline trend" in text for text in texts)
    assert any("Lead conversion rate" in text for text in texts)


def test_build_rag_documents_handle_missing_optional_columns_safely():
    deals_df = _sample_deals().drop(columns=["Region", "Product_Category"])
    documents = rag_store.build_rag_documents(_sample_leads(), deals_df)
    assert isinstance(documents, list)
    assert len(documents) > 0


def test_build_rag_documents_handle_empty_dataframes_safely():
    leads_df = pd.DataFrame(columns=["Lead_ID", "Lead_Source", "Estimated_Value"])
    deals_df = pd.DataFrame(columns=["Deal_ID", "Lead_ID", "Amount", "Closing_Date"])
    documents = rag_store.build_rag_documents(leads_df, deals_df)
    assert isinstance(documents, list)
    assert len(documents) > 0


def test_query_rag_store_handles_empty_collection_safely(monkeypatch):
    class FakeCollection:
        def count(self):
            return 0

    monkeypatch.setattr(rag_store, "_get_collection", lambda: FakeCollection())
    result = rag_store.query_rag_store("What does forecast quality mean?")
    assert result["ok"] is True
    assert result["results"] == []


def test_query_rag_store_handles_embedding_failure_safely(monkeypatch):
    class FakeCollection:
        def count(self):
            return 3

    monkeypatch.setattr(rag_store, "_get_collection", lambda: FakeCollection())
    monkeypatch.setattr(rag_store, "get_local_embedding", lambda text: None)
    result = rag_store.query_rag_store("What does forecast quality mean?")
    assert result["ok"] is False
    assert "embeddings" in result["error"].lower()
