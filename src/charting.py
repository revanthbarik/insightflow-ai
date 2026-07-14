"""Generate professional Matplotlib charts from Pandas analytics results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

from src.analytics import (
    format_aed_compact,
    get_deals_by_stage,
    get_forecast_summary,
    get_leads_by_source,
    get_product_category_revenue,
    get_region_pipeline,
    get_sales_rep_performance,
)
from src.config import CHARTS_DIR

DEFAULT_FIGURE_SIZE = (13, 7.5)
DEFAULT_DPI = 240
EMPTY_BAR_COLOR = "#9AA0A6"

CHART_STYLE = {
    "axes.facecolor": "#FFFFFF",
    "axes.edgecolor": "#DADCE0",
    "axes.labelcolor": "#1F2937",
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.color": "#374151",
    "ytick.color": "#374151",
    "grid.color": "#E5E7EB",
    "grid.linewidth": 0.8,
    "font.size": 10,
}

PROFESSIONAL_COLORS = {
    "leads_by_source": "#2F6B8A",
    "deals_by_stage": "#D97706",
    "forecast_summary": "#2E8B57",
    "sales_rep_performance": "#7C4D9E",
    "product_category_revenue": "#D9485F",
    "region_pipeline": "#2A9D8F",
    "dynamic": "#355C7D",
}


def _ensure_charts_dir() -> Path:
    """Create the charts output folder if it does not exist."""
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    return CHARTS_DIR


def _save_figure(fig: plt.Figure, chart_path: Path) -> Path:
    """Save a figure with presentation-ready defaults."""
    fig.tight_layout(pad=1.4)
    fig.savefig(chart_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)
    return chart_path


def _format_axis_label(column_name: str) -> str:
    """Convert a schema column name into a readable chart label."""
    return column_name.replace("_", " ")


def _is_currency_series(series: pd.Series) -> bool:
    """Return True when the values should be shown in AED."""
    lowered_name = series.name.lower()
    return any(token in lowered_name for token in ("amount", "value", "revenue", "pipeline"))


def _format_value_label(value: float, is_currency: bool) -> str:
    """Format labels for bars or points."""
    if is_currency:
        return format_aed_compact(value)
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:,.1f}"


def _get_value_formatter(is_currency: bool) -> FuncFormatter:
    """Return an axis formatter for numeric values."""
    if is_currency:
        return FuncFormatter(lambda value, _: format_aed_compact(value))
    return FuncFormatter(
        lambda value, _: f"{int(value):,}" if float(value).is_integer() else f"{value:,.1f}"
    )


def _save_empty_chart(
    chart_path: Path,
    title: str,
    empty_message: str,
    color: str = EMPTY_BAR_COLOR,
) -> Path:
    """Save a simple placeholder chart when no plottable data exists."""
    with plt.rc_context(CHART_STYLE):
        fig, ax = plt.subplots(figsize=DEFAULT_FIGURE_SIZE)
        ax.barh(["No Data"], [0], color=color, height=0.55)
        ax.set_title(title, pad=16)
        ax.set_xlabel("Value")
        ax.text(
            0.02,
            0.55,
            empty_message,
            transform=ax.transAxes,
            fontsize=11,
            color="#4B5563",
            va="center",
        )
        ax.grid(axis="x", linestyle="--", alpha=0.7)
        return _save_figure(fig, chart_path)


def render_categorical_chart(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str,
    chart_filename: str,
    color: str,
    chart_type: str = "bar",
    sort_desc: bool = True,
    empty_message: str = "No data is available for this chart.",
    x_label: str | None = None,
    y_label: str | None = None,
) -> Path:
    """Render a presentation-ready bar or line chart from aggregated data."""
    chart_path = _ensure_charts_dir() / chart_filename

    if data.empty or x_column not in data.columns or y_column not in data.columns:
        return _save_empty_chart(chart_path, title, empty_message, color=color)

    plot_df = data[[x_column, y_column]].dropna().copy()
    if plot_df.empty:
        return _save_empty_chart(chart_path, title, empty_message, color=color)

    if chart_type == "bar" and sort_desc:
        plot_df = plot_df.sort_values(y_column, ascending=False)

    is_currency = _is_currency_series(plot_df[y_column])
    use_horizontal = chart_type == "bar" and any(
        len(str(value)) > 12 for value in plot_df[x_column].astype(str)
    )

    with plt.rc_context(CHART_STYLE):
        fig, ax = plt.subplots(figsize=DEFAULT_FIGURE_SIZE)

        if chart_type == "line":
            if not sort_desc:
                plot_df = plot_df.reset_index(drop=True)
            ax.plot(
                plot_df[x_column],
                plot_df[y_column],
                color=color,
                linewidth=2.5,
                marker="o",
                markersize=7,
            )
            ax.grid(axis="y", linestyle="--", alpha=0.7)
            for x_value, y_value in zip(plot_df[x_column], plot_df[y_column]):
                ax.annotate(
                    _format_value_label(y_value, is_currency),
                    (x_value, y_value),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontsize=9,
                    color="#374151",
                )
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        elif use_horizontal:
            bars = ax.barh(
                plot_df[x_column].astype(str),
                plot_df[y_column],
                color=color,
                height=0.6,
            )
            ax.invert_yaxis()
            ax.grid(axis="x", linestyle="--", alpha=0.7)
            max_value = float(plot_df[y_column].max()) if not plot_df.empty else 0.0
            label_offset = max(max_value * 0.01, 1)
            for bar, value in zip(bars, plot_df[y_column]):
                ax.text(
                    bar.get_width() + label_offset,
                    bar.get_y() + (bar.get_height() / 2),
                    _format_value_label(value, is_currency),
                    va="center",
                    fontsize=9,
                    color="#374151",
                )
        else:
            bars = ax.bar(
                plot_df[x_column].astype(str),
                plot_df[y_column],
                color=color,
                width=0.65,
            )
            ax.grid(axis="y", linestyle="--", alpha=0.7)
            ax.bar_label(
                bars,
                labels=[_format_value_label(value, is_currency) for value in plot_df[y_column]],
                padding=4,
                fontsize=9,
                color="#374151",
            )
            plt.setp(ax.get_xticklabels(), rotation=25, ha="right")

        ax.set_title(title, loc="left", pad=16)
        ax.set_xlabel(x_label or _format_axis_label(x_column))
        ax.set_ylabel(y_label or _format_axis_label(y_column))

        if chart_type == "bar":
            value_axis = ax.xaxis if use_horizontal else ax.yaxis
        else:
            value_axis = ax.yaxis
        value_axis.set_major_formatter(_get_value_formatter(is_currency))

        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        return _save_figure(fig, chart_path)


def create_leads_by_source_chart(leads_df: pd.DataFrame) -> Path:
    """Create a chart of lead count by source."""
    return render_categorical_chart(
        data=get_leads_by_source(leads_df),
        x_column="Lead_Source",
        y_column="Lead_Count",
        title="Leads by Source",
        chart_filename="leads_by_source.png",
        color=PROFESSIONAL_COLORS["leads_by_source"],
        empty_message="No lead source data is available for this chart.",
        y_label="Lead Count",
    )


def create_deals_by_stage_chart(deals_df: pd.DataFrame) -> Path:
    """Create a chart of total deal amount by stage."""
    return render_categorical_chart(
        data=get_deals_by_stage(deals_df),
        x_column="Deal_Stage",
        y_column="Total_Amount",
        title="Deals by Stage",
        chart_filename="deals_by_stage.png",
        color=PROFESSIONAL_COLORS["deals_by_stage"],
        empty_message="No deal stage data is available for this chart.",
        y_label="Total Amount",
    )


def create_forecast_chart(deals_df: pd.DataFrame) -> Path:
    """Create a chart of total deal amount by forecast category."""
    return render_categorical_chart(
        data=get_forecast_summary(deals_df),
        x_column="Forecast_Category",
        y_column="Total_Amount",
        title="Forecast Summary",
        chart_filename="forecast_summary.png",
        color=PROFESSIONAL_COLORS["forecast_summary"],
        empty_message="No forecast data is available for this chart.",
        y_label="Total Amount",
    )


def create_sales_rep_chart(leads_df: pd.DataFrame, deals_df: pd.DataFrame) -> Path:
    """Create a chart of total deal amount by sales rep."""
    return render_categorical_chart(
        data=get_sales_rep_performance(leads_df, deals_df),
        x_column="Sales_Rep",
        y_column="Total_Deal_Amount",
        title="Sales Rep Performance",
        chart_filename="sales_rep_performance.png",
        color=PROFESSIONAL_COLORS["sales_rep_performance"],
        empty_message="No sales rep performance data is available for this chart.",
        y_label="Total Deal Amount",
    )


def create_product_category_revenue_chart(deals_df: pd.DataFrame) -> Path:
    """Create a chart of revenue by product category."""
    return render_categorical_chart(
        data=get_product_category_revenue(deals_df),
        x_column="Product_Category",
        y_column="Total_Amount",
        title="Product Category Revenue",
        chart_filename="product_category_revenue.png",
        color=PROFESSIONAL_COLORS["product_category_revenue"],
        empty_message="No product category revenue data is available for this chart.",
        y_label="Total Amount",
    )


def create_region_pipeline_chart(leads_df: pd.DataFrame, deals_df: pd.DataFrame) -> Path:
    """Create a chart of pipeline value by region."""
    return render_categorical_chart(
        data=get_region_pipeline(leads_df, deals_df),
        x_column="Region",
        y_column="Total_Amount",
        title="Region Pipeline",
        chart_filename="region_pipeline.png",
        color=PROFESSIONAL_COLORS["region_pipeline"],
        empty_message="No region pipeline data is available for this chart.",
        y_label="Pipeline Amount",
    )
