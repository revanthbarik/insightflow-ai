"""Tests for in-memory InsightFlow export helpers."""

from __future__ import annotations

import pandas as pd

from src.export_utils import (
    ask_result_to_dataframe,
    dataframe_to_csv_bytes,
    insights_to_dataframe,
    make_safe_filename,
)


def test_safe_filename_normalizes_user_facing_labels():
    assert make_safe_filename("Revenue by Region!") == "revenue_by_region.csv"


def test_dataframe_csv_export_preserves_rows_and_columns():
    source = pd.DataFrame({"Region": ["Dubai"], "Amount": [125_000]})
    exported = dataframe_to_csv_bytes(source).decode("utf-8")

    assert "Region,Amount" in exported
    assert "Dubai,125000" in exported


def test_insights_are_converted_to_a_clean_export_table():
    insights = [
        {
            "title": "Pipeline risk",
            "metric": "AED 1.2M at risk",
            "explanation": "At-risk deals need attention.",
            "recommended_action": "Review forecast owners.",
        }
    ]
    exported = insights_to_dataframe(insights)

    assert exported.columns.tolist() == [
        "title",
        "metric",
        "explanation",
        "recommended_action",
    ]
    assert exported.iloc[0]["title"] == "Pipeline risk"


def test_ask_metric_export_has_a_safe_one_row_shape():
    exported = ask_result_to_dataframe(
        {
            "title": "Total pipeline value",
            "answer": "AED 2.4M",
            "answer_source": "semantic BI calculation",
        }
    )

    assert len(exported) == 1
    assert exported.iloc[0]["answer_title"] == "Total pipeline value"
