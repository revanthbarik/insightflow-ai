"""Reusable Plotly charts for InsightFlow's deterministic Pandas results."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

CHART_COLORS = ["#7DD3FC", "#60A5FA", "#A78BFA", "#34D399", "#FBBF24", "#FB7185", "#38BDF8"]
PRIMARY_COLOR = "#5B8DEF"
HIGHLIGHT_COLOR = "#7DD3FC"
SECONDARY_BAR_COLOR = "#3D638D"
CHART_HEIGHTS = {"primary": 520, "secondary": 430, "supporting": 360}
SURFACE_COLOR = "#111827"
GRID_COLOR = "#334155"
TEXT_COLOR = "#E2E8F0"
MUTED_TEXT_COLOR = "#94A3B8"


def format_chart_value(value: float, is_currency: bool = False) -> str:
    """Format chart labels consistently without changing the underlying values."""
    if pd.isna(value):
        return "—"
    if is_currency:
        magnitude = abs(float(value))
        if magnitude >= 1_000_000:
            return f"AED {value / 1_000_000:.1f}M"
        if magnitude >= 1_000:
            return f"AED {value / 1_000:.0f}K"
        return f"AED {value:,.0f}"
    return f"{int(value):,}" if float(value).is_integer() else f"{value:,.1f}"


def _is_currency_column(column_name: str) -> bool:
    return any(token in column_name.lower() for token in ("amount", "value", "revenue", "pipeline", "margin"))


def _label(column_name: str) -> str:
    return column_name.replace("_", " ")


def _base_figure(title: str, subtitle: str | None = None, height: int = 430) -> go.Figure:
    """Return a dark, presentation-ready figure shell."""
    title_text = f"<b>{title}</b>"
    if subtitle:
        title_text += (
            f"<br><span style='font-size:12px;color:{MUTED_TEXT_COLOR};"
            f"font-weight:400'>{subtitle}</span>"
        )
    figure = go.Figure()
    figure.update_layout(
        title={"text": title_text, "x": 0.045, "xanchor": "left", "y": 0.92, "yanchor": "top"},
        height=height,
        margin={"l": 56, "r": 52, "t": 88, "b": 42},
        paper_bgcolor=SURFACE_COLOR,
        plot_bgcolor=SURFACE_COLOR,
        font={"family": "Inter, ui-sans-serif, system-ui, sans-serif", "color": TEXT_COLOR},
        hoverlabel={"bgcolor": "#0F172A", "font_color": TEXT_COLOR, "bordercolor": GRID_COLOR},
        showlegend=False,
    )
    return figure


def render_empty_chart_state(title: str, message: str, height: int = 300) -> go.Figure:
    """Return an intentional, non-empty visual state for unavailable chart data."""
    figure = _base_figure(title, height=height)
    figure.add_annotation(
        text=f"<b>No data available</b><br><span style='color:{MUTED_TEXT_COLOR}'>{message}</span>",
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        align="center",
        font={"size": 15, "color": TEXT_COLOR},
    )
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    return figure


def _prepare_plot_data(data: pd.DataFrame, x_column: str, y_column: str, sort_desc: bool) -> pd.DataFrame:
    if data.empty or not {x_column, y_column}.issubset(data.columns):
        return pd.DataFrame()
    plot_df = data[[x_column, y_column]].dropna().copy()
    if plot_df.empty:
        return plot_df
    return plot_df.sort_values(y_column, ascending=not sort_desc)


def _value_axis(is_currency: bool) -> dict:
    return {
        "showgrid": True,
        "gridcolor": GRID_COLOR,
        "zeroline": False,
        "automargin": True,
        "tickprefix": "AED " if is_currency else "",
        "tickformat": "~s" if is_currency else ",",
        "title": None,
    }


def _bar_colors(item_count: int, highlight_top: bool) -> list[str] | str:
    """Use a calm default bar treatment with an optional single focal point."""
    if not highlight_top or item_count < 2:
        return PRIMARY_COLOR
    return [HIGHLIGHT_COLOR] + [SECONDARY_BAR_COLOR] * (item_count - 1)


def render_horizontal_bar_chart(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str,
    subtitle: str | None = None,
    highlight_top: bool = False,
    empty_message: str = "Upload data with the required fields to unlock this analysis.",
) -> go.Figure:
    """Render a descending horizontal comparison chart."""
    plot_df = _prepare_plot_data(data, x_column, y_column, sort_desc=True)
    if plot_df.empty:
        return render_empty_chart_state(title, empty_message)
    is_currency = _is_currency_column(y_column)
    labels = [format_chart_value(value, is_currency) for value in plot_df[y_column]]
    figure = _base_figure(title, subtitle, height=max(360, min(620, 110 + len(plot_df) * 48)))
    figure.add_bar(
        x=plot_df[y_column],
        y=plot_df[x_column].astype(str),
        orientation="h",
        marker={"color": _bar_colors(len(plot_df), highlight_top), "line": {"width": 0}},
        text=labels,
        textposition="outside",
        cliponaxis=False,
        customdata=[[label] for label in labels],
        hovertemplate=f"<b>%{{y}}</b><br>{_label(y_column)}: %{{customdata[0]}}<extra></extra>",
    )
    figure.update_yaxes(autorange="reversed", showgrid=False, title=None, automargin=True)
    figure.update_xaxes(**_value_axis(is_currency))
    return figure


def render_bar_chart(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str,
    subtitle: str | None = None,
    highlight_top: bool = False,
    empty_message: str = "Upload data with the required fields to unlock this analysis.",
) -> go.Figure:
    """Render a descending vertical comparison chart for short labels."""
    plot_df = _prepare_plot_data(data, x_column, y_column, sort_desc=True)
    if plot_df.empty:
        return render_empty_chart_state(title, empty_message)
    is_currency = _is_currency_column(y_column)
    labels = [format_chart_value(value, is_currency) for value in plot_df[y_column]]
    figure = _base_figure(title, subtitle)
    figure.add_bar(
        x=plot_df[x_column].astype(str),
        y=plot_df[y_column],
        marker={"color": _bar_colors(len(plot_df), highlight_top), "line": {"width": 0}},
        text=labels,
        textposition="outside",
        cliponaxis=False,
        customdata=[[label] for label in labels],
        hovertemplate=f"<b>%{{x}}</b><br>{_label(y_column)}: %{{customdata[0]}}<extra></extra>",
    )
    figure.update_xaxes(showgrid=False, tickangle=-20, title=None, automargin=True)
    figure.update_yaxes(**_value_axis(is_currency))
    return figure


def render_line_chart(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str,
    subtitle: str | None = None,
    empty_message: str = "No time-based data is available for this chart.",
) -> go.Figure:
    """Render a chronological deterministic trend chart."""
    plot_df = _prepare_plot_data(data, x_column, y_column, sort_desc=False)
    if plot_df.empty:
        return render_empty_chart_state(title, empty_message)
    plot_df = plot_df.sort_values(x_column)
    is_currency = _is_currency_column(y_column)
    labels = [format_chart_value(value, is_currency) for value in plot_df[y_column]]
    figure = _base_figure(title, subtitle)
    figure.add_scatter(
        x=plot_df[x_column],
        y=plot_df[y_column],
        mode="lines+markers",
        line={"color": PRIMARY_COLOR, "width": 3},
        marker={"color": "#E0F2FE", "size": 8, "line": {"color": PRIMARY_COLOR, "width": 2}},
        customdata=[[label] for label in labels],
        hovertemplate=f"<b>%{{x}}</b><br>{_label(y_column)}: %{{customdata[0]}}<extra></extra>",
    )
    figure.update_xaxes(showgrid=False, title=None, tickangle=-25, automargin=True)
    figure.update_yaxes(**_value_axis(is_currency))
    return figure


def render_donut_chart(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str,
    subtitle: str | None = None,
    empty_message: str = "No composition data is available for this chart.",
) -> go.Figure:
    """Render a readable part-to-whole chart for a small category set."""
    plot_df = _prepare_plot_data(data, x_column, y_column, sort_desc=True)
    if plot_df.empty:
        return render_empty_chart_state(title, empty_message)
    is_currency = _is_currency_column(y_column)
    figure = _base_figure(title, subtitle)
    figure.add_pie(
        labels=plot_df[x_column].astype(str),
        values=plot_df[y_column],
        hole=0.62,
        domain={"x": [0.0, 0.66], "y": [0.02, 0.98]},
        marker={"colors": CHART_COLORS, "line": {"color": SURFACE_COLOR, "width": 3}},
        textinfo="percent",
        textfont={"color": TEXT_COLOR},
        customdata=[[format_chart_value(value, is_currency)] for value in plot_df[y_column]],
        hovertemplate=f"<b>%{{label}}</b><br>{_label(y_column)}: %{{customdata[0]}}<br>Share: %{{percent}}<extra></extra>",
    )
    total = format_chart_value(plot_df[y_column].sum(), is_currency)
    figure.add_annotation(
        text=f"<span style='color:{MUTED_TEXT_COLOR};font-size:12px'>TOTAL</span><br><b>{total}</b>",
        x=0.33,
        y=0.5,
        showarrow=False,
        font={"size": 17, "color": TEXT_COLOR},
    )
    figure.update_layout(
        showlegend=True,
        margin={"l": 40, "r": 150, "t": 88, "b": 34},
        legend={
            "orientation": "v",
            "x": 0.72,
            "xanchor": "left",
            "y": 0.5,
            "yanchor": "middle",
            "font": {"size": 11, "color": TEXT_COLOR},
            "itemsizing": "constant",
        },
    )
    return figure


def render_stacked_bar_chart(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    color_column: str,
    title: str,
    subtitle: str | None = None,
) -> go.Figure:
    """Render a stacked breakdown only when a second categorical dimension exists."""
    required_columns = {x_column, y_column, color_column}
    if data.empty or not required_columns.issubset(data.columns):
        return render_empty_chart_state(title, "This view requires two category fields and a numeric value.")
    is_currency = _is_currency_column(y_column)
    plot_df = data[list(required_columns)].dropna().copy()
    figure = _base_figure(title, subtitle)
    for index, (category, group) in enumerate(plot_df.groupby(color_column, sort=False)):
        figure.add_bar(
            x=group[x_column].astype(str),
            y=group[y_column],
            name=str(category),
            marker={"color": CHART_COLORS[index % len(CHART_COLORS)]},
            hovertemplate=f"<b>%{{x}}</b><br>{_label(color_column)}: {category}<br>{_label(y_column)}: %{{y:,.0f}}<extra></extra>",
        )
    figure.update_layout(barmode="stack", showlegend=True, legend={"orientation": "h", "y": -0.2, "x": 0})
    figure.update_xaxes(showgrid=False, title=None, automargin=True)
    figure.update_yaxes(**_value_axis(is_currency))
    return figure


def select_chart_type(
    data: pd.DataFrame,
    x_column: str,
    requested_type: str | None = None,
    chart_key: str | None = None,
    secondary_column: str | None = None,
) -> str:
    """Choose a chart from data shape and known InsightFlow business contexts."""
    if requested_type == "line" or "month" in x_column.lower() or "date" in x_column.lower():
        return "line"
    if secondary_column and secondary_column in data.columns:
        return "stacked_bar"
    category_count = data[x_column].nunique(dropna=True) if x_column in data.columns else 0
    is_composition = chart_key in {"deals_by_stage", "forecast_summary"} or any(
        token in x_column.lower() for token in ("stage", "forecast", "status")
    )
    if is_composition and 2 <= category_count <= 6:
        return "donut"
    has_long_labels = (
        x_column in data.columns
        and not data.empty
        and data[x_column].astype(str).str.len().max() > 12
    )
    return "horizontal_bar" if has_long_labels or category_count > 6 else "bar"


def build_chart_figure(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str,
    subtitle: str | None = None,
    requested_type: str | None = None,
    chart_key: str | None = None,
    secondary_column: str | None = None,
    emphasis: str = "secondary",
    highlight_top: bool = False,
    empty_message: str = "Upload data with the required fields to unlock this analysis.",
) -> go.Figure:
    """Create an InsightFlow chart without changing data aggregation or meaning."""
    chart_type = select_chart_type(data, x_column, requested_type, chart_key, secondary_column)
    if chart_type == "line":
        figure = render_line_chart(data, x_column, y_column, title, subtitle, empty_message)
    elif chart_type == "donut":
        figure = render_donut_chart(data, x_column, y_column, title, subtitle, empty_message)
    elif chart_type == "stacked_bar" and secondary_column:
        figure = render_stacked_bar_chart(data, x_column, y_column, secondary_column, title, subtitle)
    elif chart_type == "horizontal_bar":
        figure = render_horizontal_bar_chart(
            data,
            x_column,
            y_column,
            title,
            subtitle,
            highlight_top,
            empty_message,
        )
    else:
        figure = render_bar_chart(
            data,
            x_column,
            y_column,
            title,
            subtitle,
            highlight_top,
            empty_message,
        )

    chart_height = CHART_HEIGHTS.get(emphasis, CHART_HEIGHTS["secondary"])
    title_size = 21 if emphasis == "primary" else 17 if emphasis == "secondary" else 15
    figure.update_layout(height=chart_height, title={"font": {"size": title_size}})
    return figure
