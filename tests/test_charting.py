"""Tests for the presentation-only Plotly chart utility layer."""

from __future__ import annotations

import pandas as pd

from src.charting import build_chart_figure, format_chart_value, select_chart_type


def _comparison_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Region": ["Saudi Arabia", "United Arab Emirates", "Oman"],
            "Total_Amount": [1_200_000, 850_000, 325_000],
        }
    )


def test_currency_labels_use_compact_aed_formatting():
    assert format_chart_value(1_250_000, is_currency=True) == "AED 1.2M"
    assert format_chart_value(42, is_currency=False) == "42"


def test_long_category_comparisons_use_horizontal_bars():
    data = _comparison_data()
    assert select_chart_type(data, "Region") == "horizontal_bar"

    figure = build_chart_figure(
        data,
        x_column="Region",
        y_column="Total_Amount",
        title="Revenue by Region",
    )
    assert figure.data[0].type == "bar"
    assert figure.data[0].orientation == "h"
    assert list(figure.data[0].x) == [1_200_000, 850_000, 325_000]


def test_time_based_data_uses_a_line_chart_in_chronological_order():
    data = pd.DataFrame(
        {
            "Closing_Month": ["2024-03", "2024-01", "2024-02"],
            "Total_Amount": [300_000, 100_000, 200_000],
        }
    )
    figure = build_chart_figure(
        data,
        x_column="Closing_Month",
        y_column="Total_Amount",
        title="Monthly Pipeline",
    )
    assert figure.data[0].type == "scatter"
    assert list(figure.data[0].x) == ["2024-01", "2024-02", "2024-03"]


def test_stage_composition_uses_a_donut_when_category_count_is_small():
    data = pd.DataFrame(
        {
            "Deal_Stage": ["Qualification", "Proposal", "Negotiation"],
            "Total_Amount": [300_000, 500_000, 200_000],
        }
    )
    figure = build_chart_figure(
        data,
        x_column="Deal_Stage",
        y_column="Total_Amount",
        title="Deals by Stage",
        chart_key="deals_by_stage",
    )
    assert figure.data[0].type == "pie"
    assert figure.data[0].hole == 0.62
    assert figure.layout.legend.orientation == "v"
    assert figure.layout.legend.x == 0.72


def test_chart_emphasis_is_a_presentation_only_height_setting():
    figure = build_chart_figure(
        _comparison_data(),
        x_column="Region",
        y_column="Total_Amount",
        title="Revenue by Region",
        emphasis="primary",
        highlight_top=True,
    )
    assert figure.layout.height == 520
    assert figure.data[0].marker.color[0] == "#7DD3FC"


def test_empty_data_returns_an_intentional_empty_state():
    figure = build_chart_figure(
        pd.DataFrame(),
        x_column="Region",
        y_column="Total_Amount",
        title="Revenue by Region",
        empty_message="This view requires Deals and Region fields.",
    )
    assert len(figure.data) == 0
    assert "No data available" in figure.layout.annotations[0].text
