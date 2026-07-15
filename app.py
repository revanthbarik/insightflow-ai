"""Streamlit entry point for InsightFlow AI."""

from __future__ import annotations

import pandas as pd
import re
import streamlit as st
from pandas.util import hash_pandas_object

from src.analytics import (
    generate_auto_insights,
    get_empty_reason,
    get_deals_by_stage,
    get_forecast_summary,
    get_leads_by_source,
    get_product_category_revenue,
    get_region_pipeline,
    get_sales_rep_performance,
    get_total_deal_amount,
    get_total_deals,
    get_total_estimated_lead_value,
    get_total_leads,
)
from src.charting import build_chart_figure
from src.data_loader import (
    empty_deals_df,
    empty_leads_df,
    load_all_data,
    load_zoho_modules_from_files,
)
from src.data_health import DataHealthReport, build_data_health_report
from src.export_utils import (
    ask_result_to_dataframe,
    dataframe_to_csv_bytes,
    insights_to_dataframe,
    make_safe_filename,
)
from src.capability_detector import detect_available_analytics
from src.module_registry import SUPPORTED_MODULE_NAMES
from src.relationship_builder import build_relationships
from src.zoho_adapter import missing_supported_modules
from src.followup import (
    answer_followup_question,
    build_last_answer_context,
)
from src.llm import (
    check_ollama_available,
    check_ollama_model_available,
    synthesize_rag_answer,
)
from src.config import OLLAMA_EMBED_MODEL, OLLAMA_TEXT_MODEL
from src.query_engine import answer_business_question
from src.rag_store import (
    format_retrieved_context,
    get_rag_status,
    query_rag_store,
    refresh_rag_store,
    reset_rag_store,
)
from src.router import detect_qualitative_intent

CHART_DATA_BUILDERS = {
    "Leads by Source": lambda *chart_args: get_leads_by_source(chart_args[0]),
    "Deals by Stage": lambda *chart_args: get_deals_by_stage(chart_args[0]),
    "Forecast Summary": lambda *chart_args: get_forecast_summary(chart_args[0]),
    "Sales Rep Performance": lambda *chart_args: get_sales_rep_performance(chart_args[0], chart_args[1]),
    "Product Category Revenue": lambda *chart_args: get_product_category_revenue(chart_args[0]),
    "Region Pipeline": lambda *chart_args: get_region_pipeline(chart_args[0], chart_args[1]),
}

CHART_CONFIG = {
    "Leads by Source": ("Lead_Source", "Lead_Count", "leads_by_source", "Lead volume by acquisition channel"),
    "Deals by Stage": ("Deal_Stage", "Total_Amount", "deals_by_stage", "Pipeline value across deal stages"),
    "Forecast Summary": ("Forecast_Category", "Total_Amount", "forecast_summary", "Revenue mix by forecast category"),
    "Sales Rep Performance": ("Sales_Rep", "Total_Deal_Amount", "sales_rep_performance", "Deal value by sales owner"),
    "Product Category Revenue": ("Product_Category", "Total_Amount", "product_category_revenue", "Revenue by product category"),
    "Region Pipeline": ("Region", "Total_Amount", "region_pipeline", "Pipeline value by region"),
}

TABLE_CURRENCY_COLUMNS = {
    "Leads by Source": ["Total_Estimated_Value"],
    "Deals by Stage": ["Total_Amount"],
    "Forecast Summary": ["Total_Amount"],
    "Sales Rep Performance": ["Total_Estimated_Value", "Total_Deal_Amount"],
    "Product Category Revenue Chart": ["Total_Amount"],
    "Region-wise Pipeline Chart": ["Total_Amount"],
    "Monthly Pipeline Trend by Closing Date": ["Total_Amount"],
    "Monthly Deal Amount Trend by Closing Date": ["Total_Amount"],
    "High Value but Low Conversion Lead Source": ["Estimated_Value"],
}

CURRENCY_COLUMN_LABELS = {
    "Total_Estimated_Value": "Total Estimated Value",
    "Total_Amount": "Total Amount",
    "Total_Deal_Amount": "Total Deal Amount",
}

ASK_EXAMPLES = [
    "What is the total pipeline value?",
    "Which product category has the weakest forecast quality?",
    "Show monthly pipeline trend based on closing date",
    "Which lead source generates the highest estimated value but low conversion into deals?",
    "What does forecast quality mean?",
    "What should management focus on this week?",
    "Create a chart of deals by stage",
    "Show revenue by region",
    "Explain the forecast categories.",
]


def chart_widget_key(scope: str, title: str) -> str:
    """Create stable, distinct Streamlit keys for independently rendered charts."""
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    return f"{scope}_{slug}"


def display_csv_download(
    dataframe: pd.DataFrame,
    label: str,
    key: str,
    button_label: str = "Download CSV",
) -> None:
    """Show one compact in-memory CSV download when data is available."""
    if dataframe is None or dataframe.empty:
        return
    st.download_button(
        button_label,
        data=dataframe_to_csv_bytes(dataframe),
        file_name=make_safe_filename(label),
        mime="text/csv",
        key=key,
        type="tertiary",
        icon=":material/download:",
        on_click="ignore",
    )


def format_aed(value) -> str:
    """Format a numeric value as an AED currency string for display."""
    if value is None:
        return "N/A"

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "N/A"

    if numeric != numeric:
        return "N/A"

    return f"AED {numeric:,.0f}"


def format_table_for_display(df, currency_columns: list[str]):
    """Return a display copy with AED-formatted currency columns and readable labels."""
    if df is None:
        return pd.DataFrame()

    display_df = df.copy()

    for column in currency_columns:
        if column in display_df.columns:
            display_df[column] = display_df[column].apply(format_aed)

    rename_map = {
        column: CURRENCY_COLUMN_LABELS[column]
        for column in currency_columns
        if column in display_df.columns and column in CURRENCY_COLUMN_LABELS
    }
    return display_df.rename(columns=rename_map)


def display_aed_metric(label: str, formatted_value: str) -> None:
    """Show a visually consistent KPI card without changing metric values."""
    st.markdown(
        f"""
        <div class="kpi-card">
            <p class="kpi-label">
                {label}
            </p>
            <p class="kpi-value">
                {formatted_value}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_chart(
    title: str,
    *args,
    render_key: str,
    emphasis: str = "secondary",
    highlight_top: bool = False,
) -> None:
    """Render a chart-first Plotly view from existing deterministic aggregates."""
    data_builder = CHART_DATA_BUILDERS.get(title)
    chart_config = CHART_CONFIG.get(title)
    if data_builder is None or chart_config is None:
        st.warning("This chart configuration is unavailable.")
        return

    chart_data = data_builder(*args)
    x_column, y_column, chart_key, subtitle = chart_config
    empty_message = get_empty_reason(
        chart_data,
        f"No data is available for the {title.lower()} chart.",
    )
    # Plotly's dark surface provides the card contrast; avoiding a second
    # Streamlit border keeps the dashboard light and less boxed-in.
    with st.container():
        st.plotly_chart(
            build_chart_figure(
                chart_data,
                x_column=x_column,
                y_column=y_column,
                title=title,
                subtitle=subtitle,
                chart_key=chart_key,
                emphasis=emphasis,
                highlight_top=highlight_top,
                empty_message=empty_message,
            ),
            width="stretch",
            key=render_key,
            config={"displayModeBar": False, "responsive": True, "scrollZoom": False},
        )
        display_csv_download(
            chart_data,
            label=f"{title} chart data",
            key=f"export_chart_{render_key}",
            button_label="Export chart data",
        )
    if chart_data.empty:
        available_columns = chart_data.attrs.get("available_columns", [])
        if available_columns:
            st.caption(f"Available columns: {', '.join(available_columns)}")
        for filter_description in chart_data.attrs.get("applied_filters", []):
            st.caption(filter_description)


def display_insight_card(insight: dict) -> None:
    """Display one auto insight in a bordered card."""
    with st.container(border=True):
        st.markdown(f"#### {insight['title']}")
        st.markdown(f"**{insight['metric']}**")
        st.markdown(insight["explanation"])
        st.markdown(f"**Recommended action:** {insight['recommended_action']}")


def display_answer_source(answer_source: str) -> None:
    """Display where the Ask-tab answer came from."""
    st.caption(f"Answer source: {answer_source}")


def display_supporting_metric_table(result: dict) -> None:
    """Show a concise supporting table for metric answers when useful."""
    if not isinstance(result.get("data"), pd.DataFrame) or result["data"].empty:
        return

    if result.get("answer_source") == "lead conversion analysis":
        support_df = result["data"].copy().head(5)
        if "Conversion_Rate" in support_df.columns:
            support_df["Conversion_Rate"] = support_df["Conversion_Rate"].apply(
                lambda value: f"{value * 100:.1f}%"
            )
        with st.expander("View supporting data", expanded=False):
            st.dataframe(
                format_table_for_display(
                    support_df,
                    TABLE_CURRENCY_COLUMNS.get(result["title"], []),
                ),
                width="stretch",
                hide_index=True,
            )


def display_chart_data_table(
    data: pd.DataFrame,
    currency_columns: list[str],
    export_label: str,
    export_key: str,
) -> None:
    """Keep chart details available without letting the table dominate the view."""
    if data is None or data.empty:
        return
    with st.expander("View chart data", expanded=False):
        st.dataframe(
            format_table_for_display(data, currency_columns),
            width="stretch",
            hide_index=True,
        )
        display_csv_download(
            data,
            label=export_label,
            key=export_key,
            button_label="Export chart data",
        )


def display_question_answer(result: dict) -> None:
    """Display a rule-based or dynamic Ask-tab answer."""
    st.markdown(f"### {result['title']}")
    display_answer_source(result.get("answer_source", "supported response"))

    if result["answer_type"] == "unsupported":
        st.warning(result["answer"])
    else:
        st.markdown(result["answer"])

    if result["answer_type"] == "metric":
        display_supporting_metric_table(result)
    elif result["answer_type"] == "table" and result["data"] is not None:
        if result["data"].empty:
            st.info("There is no matching data to display for this question.")
        else:
            currency_columns = TABLE_CURRENCY_COLUMNS.get(result["title"], [])
            with st.expander("View result table", expanded=True):
                st.dataframe(
                    format_table_for_display(result["data"], currency_columns),
                    width="stretch",
                    hide_index=True,
                )
                display_csv_download(
                    result["data"],
                    label=f"ask result {result['title']}",
                    key=chart_widget_key("export_ask_table", result["title"]),
                    button_label="Export Ask result",
                )
    elif result["answer_type"] == "chart":
        chart_key = result.get("chart_key")
        chart_data = result.get("data") if isinstance(result.get("data"), pd.DataFrame) else pd.DataFrame()
        if result.get("x_axis") and result.get("y_axis"):
            x_column = result["x_axis"]
            y_column = result["y_axis"]
        elif chart_key == "monthly_pipeline_trend":
            x_column, y_column = "Closing_Month", "Total_Amount"
        else:
            chart_config = next(
                (config for _, config in CHART_CONFIG.items() if config[2] == chart_key),
                None,
            )
            if chart_config is None:
                st.warning("This chart type is not supported yet.")
                return
            x_column, y_column = chart_config[0], chart_config[1]

        st.plotly_chart(
            build_chart_figure(
                chart_data,
                x_column=x_column,
                y_column=y_column,
                title=result.get("chart_title", result["title"]),
                subtitle="Computed deterministically from the active dataset",
                requested_type=result.get("chart_type"),
                chart_key=chart_key,
                empty_message="No data is available for this chart.",
            ),
            width="stretch",
            key=chart_widget_key("ask", result.get("chart_title", result["title"])),
            config={"displayModeBar": False, "responsive": True},
        )
        currency_columns = [
            column
            for column in chart_data.columns
            if any(token in column.lower() for token in ("amount", "value", "margin", "revenue", "pipeline"))
        ]
        display_chart_data_table(
            chart_data,
            currency_columns,
            export_label=f"ask result {result['title']}",
            export_key=chart_widget_key("export_ask_chart", result["title"]),
        )
    elif result["answer_type"] == "dynamic_chart":
        chart_spec = result["chart_spec"]
        st.plotly_chart(
            build_chart_figure(
                result["data"],
                x_column=chart_spec["x_column"],
                y_column="Value",
                title=result["title"],
                subtitle="Computed deterministically from the active dataset",
                requested_type=chart_spec.get("chart_type"),
                empty_message="No matching data is available for this chart.",
            ),
            width="stretch",
            key=chart_widget_key("ask_dynamic", result["title"]),
            config={"displayModeBar": False, "responsive": True},
        )
        display_chart_data_table(
            result["data"],
            ["Value"],
            export_label=f"ask result {result['title']}",
            export_key=chart_widget_key("export_ask_dynamic_chart", result["title"]),
        )
        if result.get("explanation"):
            st.markdown(result["explanation"])
    elif result["answer_type"] == "rag":
        if result.get("retrieved_results"):
            with st.expander("Retrieved local context"):
                for index, retrieved in enumerate(result["retrieved_results"], start=1):
                    metadata = retrieved.get("metadata", {})
                    st.markdown(
                        f"**{index}. {metadata.get('topic', 'Context')}** "
                        f"({metadata.get('source_type', 'unknown')})"
                    )
                    st.markdown(retrieved.get("text", ""))

    if result.get("recommended_action"):
        st.markdown(f"**Recommended action:** {result['recommended_action']}")

    if result["answer_type"] == "metric":
        display_csv_download(
            ask_result_to_dataframe(result),
            label=f"ask result {result['title']}",
            key=chart_widget_key("export_ask_metric", result["title"]),
            button_label="Export Ask result",
        )


def display_table_section(title: str, df: pd.DataFrame, currency_columns: list[str]) -> None:
    """Display a secondary analytics table with a safe empty-data message."""
    if df.empty:
        st.info(get_empty_reason(df, f"No data is available for {title.lower()}."))
        return
    with st.expander(f"View {title} data", expanded=False):
        st.dataframe(
            format_table_for_display(df, currency_columns),
            width="stretch",
            hide_index=True,
        )
        display_csv_download(
            df,
            label=title,
            key=chart_widget_key("export_analytics", title),
        )


def display_data_readiness(report: DataHealthReport) -> None:
    """Render a concise, non-technical readiness view for active CRM data."""
    st.markdown(
        "<p class='section-kicker'>Data health</p>"
        "<p class='section-title'>Data readiness</p>"
        "<p class='section-copy'>Confirm the loaded CRM data is complete enough for "
        "the analysis, charts, and answers shown below.</p>",
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        module_col, rows_col, capability_col, warning_col = st.columns(4)
        module_col.metric("Modules loaded", report.module_count)
        rows_col.metric("Rows loaded", f"{report.total_rows:,}")
        capability_col.metric("Capabilities enabled", report.enabled_capability_count)
        warning_col.metric("Data warnings", len(report.warnings))

        st.markdown("**Module overview**")
        st.dataframe(
            report.module_overview_frame(),
            width="stretch",
            hide_index=True,
        )
        display_csv_download(
            report.module_overview_frame(),
            label="data readiness module overview",
            key="export_data_readiness_modules",
        )

        st.markdown("**Capability readiness**")
        st.dataframe(
            report.capability_frame(),
            width="stretch",
            hide_index=True,
        )
        display_csv_download(
            report.capability_frame(),
            label="data readiness capabilities",
            key="export_data_readiness_capabilities",
        )

        with st.expander("Join health details", expanded=False):
            st.dataframe(
                report.join_frame(),
                width="stretch",
                hide_index=True,
            )
            display_csv_download(
                report.join_frame(),
                label="data readiness joins",
                key="export_data_readiness_joins",
            )

        if report.resolved_mappings:
            with st.expander("Resolved field mappings", expanded=False):
                for module_name, mappings in report.resolved_mappings.items():
                    st.markdown(f"**{module_name}**")
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {"Source field": source, "InsightFlow field": target}
                                for source, target in mappings.items()
                            ]
                        ),
                        width="stretch",
                        hide_index=True,
                    )

        with st.expander("Data quality details", expanded=False):
            for module in report.module_health:
                required_columns = ", ".join(module.required_columns)
                st.markdown(f"**{module.module_name}** · required: `{required_columns}`")
                for warning in module.warnings:
                    st.caption(warning)
            if report.warnings:
                st.markdown("**Warnings to review**")
                for warning in report.warnings:
                    st.caption(f"• {warning}")
            else:
                st.caption("No important data quality issues were detected.")

        if report.warnings:
            st.warning(
                "Some data quality checks need attention. Analytics remain available "
                "where their required inputs are present.",
                icon=":material/warning:",
            )
        else:
            st.success(
                "Data is ready for the available InsightFlow analytics.",
                icon=":material/check_circle:",
            )


def load_active_data(uploaded_files) -> tuple:
    """Load uploaded Zoho CRM CSVs when provided, otherwise use default demo data."""
    if uploaded_files:
        try:
            loaded_modules, module_warnings, unknown_files, module_diagnostics = load_zoho_modules_from_files(uploaded_files)
            leads_df = loaded_modules.get("Leads", empty_leads_df())
            deals_df = loaded_modules.get("Deals", empty_deals_df())
            return (
                leads_df,
                deals_df,
                loaded_modules,
                module_warnings,
                unknown_files,
                module_diagnostics,
                "Uploaded Zoho CRM CSV data",
                "uploaded",
            )
        except (ValueError, pd.errors.ParserError) as error:
            st.error(f"Uploaded CSV validation failed: {error}")
            st.warning("Falling back to default synthetic data.")

    leads_df, deals_df = load_all_data()
    loaded_modules = {"Leads": leads_df, "Deals": deals_df}
    return leads_df, deals_df, loaded_modules, {}, [], {}, "Default synthetic demo data", "default"


def build_data_signature(
    leads_df: pd.DataFrame,
    deals_df: pd.DataFrame,
    loaded_modules: dict[str, pd.DataFrame] | None = None,
) -> tuple:
    """Build a stable signature so local state can detect data changes."""
    leads_hash = int(hash_pandas_object(leads_df.astype(str), index=True).sum()) if not leads_df.empty else 0
    deals_hash = int(hash_pandas_object(deals_df.astype(str), index=True).sum()) if not deals_df.empty else 0
    module_signature: tuple = tuple()
    if loaded_modules:
        module_signature = tuple(
            (
                module_name,
                tuple(df.columns.tolist()),
                len(df),
            )
            for module_name, df in sorted(loaded_modules.items())
        )
    return (
        tuple(leads_df.columns.tolist()),
        tuple(deals_df.columns.tolist()),
        len(leads_df),
        len(deals_df),
        leads_hash,
        deals_hash,
        module_signature,
    )


def build_rag_fallback_answer(
    question: str,
    error_message: str,
) -> dict:
    """Return a safe fallback when local RAG is unavailable."""
    return {
        "answer_type": "unsupported",
        "title": "Local Knowledge Base Unavailable",
        "answer": (
            "Local knowledge base is unavailable. The app will continue with Pandas-based metrics and charts. "
            f"Details: {error_message}"
        ),
        "recommended_action": (
            "Check that Ollama is running, `nomic-embed-text` is available, and the local knowledge base has been refreshed."
        ),
        "answer_source": "unsupported",
    }


def resolve_question_response(
    question: str,
    leads_df: pd.DataFrame,
    deals_df: pd.DataFrame,
    auto_insights: list[dict],
    rag_is_current: bool,
    extra_tables: dict[str, pd.DataFrame] | None = None,
) -> dict:
    """Route Ask-tab questions through deterministic metric/table/chart logic."""
    result = answer_business_question(
        question,
        leads_df=leads_df,
        deals_df=deals_df,
        extra_tables=extra_tables,
    )
    if "answer_source" not in result:
        if result["answer_type"] == "metric":
            result["answer_source"] = "metric calculation"
        elif result["answer_type"] == "chart":
            result["answer_source"] = (
                "time trend chart"
                if result.get("chart_key") == "monthly_pipeline_trend"
                else "semantic bi query engine"
            )
        elif result["answer_type"] == "table":
            result["answer_source"] = "table aggregation"
        else:
            result["answer_source"] = "unsupported"
    if result["answer_type"] == "unsupported" and detect_qualitative_intent(question):
        if not rag_is_current:
            return {
                "answer_type": "unsupported",
                "title": "Local Knowledge Base Needs Refresh",
                "answer": (
                    "Data changed. Refresh local knowledge base to update RAG context before asking qualitative questions."
                ),
                "recommended_action": "Use the Refresh Local Knowledge Base button in the sidebar.",
                "answer_source": "unsupported",
            }
        rag_query = query_rag_store(question, top_k=5)
        if not rag_query["ok"]:
            return build_rag_fallback_answer(question, rag_query["error"] or "Unknown RAG query error.")

        retrieved_results = rag_query["results"]
        if not retrieved_results:
            return {
                "answer_type": "unsupported",
                "title": "No Local Context Found",
                "answer": (
                    "The local knowledge base does not yet have enough context to answer this qualitative question. "
                    "Refresh the local knowledge base after loading data."
                ),
                "recommended_action": "Refresh Local Knowledge Base or ask a Pandas-based metric or chart question.",
                "answer_source": "local RAG context",
            }

        retrieved_context = format_retrieved_context(retrieved_results)
        if check_ollama_available() and check_ollama_model_available(OLLAMA_TEXT_MODEL):
            synthesis = synthesize_rag_answer(
                user_question=question,
                retrieved_context=retrieved_context,
                computed_context="\n".join(
                    [
                        f"{insight['title']}: {insight['metric']} — {insight['explanation']}"
                        for insight in auto_insights[:6]
                    ]
                ),
            )
            if synthesis["ok"] and synthesis["content"]:
                answer_text = synthesis["content"]
            else:
                answer_text = retrieved_context
        else:
            answer_text = (
                "Ollama is not available. Showing computed/RAG context without local AI synthesis.\n\n"
                + retrieved_context
            )

        return {
            "answer_type": "rag",
            "title": "Local RAG Context Answer",
            "answer": answer_text,
            "recommended_action": "Use the retrieved local context to guide management interpretation and next steps.",
            "answer_source": "local RAG context",
            "retrieved_results": retrieved_results,
        }
    return result


st.set_page_config(
    page_title="InsightFlow AI",
    page_icon=":material/analytics:",
    layout="wide",
)

st.markdown(
    """
    <style>
        .block-container {padding-top: 2.5rem; padding-bottom: 3.5rem;}
        .kpi-card {
            min-height: 102px; padding: 1.15rem 1.25rem; border: 1px solid rgba(148, 163, 184, 0.12);
            border-radius: 14px; background: linear-gradient(135deg, rgba(30, 41, 59, 0.72), rgba(15, 23, 42, 0.58));
        }
        .kpi-label {margin: 0 0 .45rem; font-size: .82rem; font-weight: 600; color: #94A3B8; text-transform: uppercase; letter-spacing: .04em;}
        .kpi-value {margin: 0; font-size: 1.65rem; font-weight: 700; color: #F8FAFC; white-space: nowrap;}
        .section-kicker {margin: 0 0 .3rem; font-size: .76rem; color: #7DD3FC; font-weight: 700; letter-spacing: .1em; text-transform: uppercase;}
        .section-title {margin: 0 0 .4rem; font-size: 1.45rem; font-weight: 700; letter-spacing: -.015em;}
        .section-copy {margin: 0 0 1.45rem; color: #94A3B8; font-size: .94rem; line-height: 1.5;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("InsightFlow AI")
st.caption("Private Revenue Operations Agent · Local data · Deterministic analytics")

with st.sidebar:
    st.header("Local Offline Prototype")
    st.markdown(
        """
        - **Runtime:** Local laptop
        - **Cloud APIs:** Disabled
        """
    )

    st.header("Data Source")
    uploaded_module_files = st.file_uploader(
        "Upload Zoho CRM CSV exports",
        type=["csv"],
        accept_multiple_files=True,
        help="Upload one or more Zoho CRM module exports such as Leads, Deals, Accounts, Products, or Contacts.",
    )

try:
    leads_df, deals_df, loaded_modules, module_warnings, unknown_module_files, module_diagnostics, active_data_source, data_source_key = load_active_data(
        uploaded_module_files,
    )
except (FileNotFoundError, ValueError) as error:
    st.error(f"Failed to load data: {error}")
    st.stop()

available_capabilities = detect_available_analytics(loaded_modules)
module_relationships = build_relationships(loaded_modules)

with st.sidebar:
    st.markdown(f"**Active source:** {active_data_source}")
    st.markdown("**Loaded Zoho Modules**")
    for module_name in SUPPORTED_MODULE_NAMES:
        if module_name in loaded_modules:
            st.markdown(f"- ✓ {module_name}")
    missing_modules = missing_supported_modules(loaded_modules)
    if missing_modules:
        st.markdown("**Missing**")
        for module_name in missing_modules:
            st.markdown(f"- • {module_name}")

    if unknown_module_files:
        st.markdown("**Unknown Uploads**")
        for file_name in unknown_module_files:
            st.markdown(f"- • {file_name}")

    st.markdown("**Available Analytics**")
    for capability in available_capabilities:
        if capability.available:
            st.markdown(f"- ✓ {capability.name}")

    st.markdown("**Unavailable Analytics**")
    for capability in available_capabilities:
        if not capability.available:
            st.markdown(f"- • {capability.name}")
            st.caption(f"Reason: {capability.reason}")

    if module_relationships:
        with st.expander("Module Relationships"):
            for relationship in module_relationships:
                indicator = "✓" if relationship.available else "•"
                st.markdown(
                    f"- {indicator} {relationship.left_module} ↔ {relationship.right_module} on `{relationship.join_key}`"
                )
                st.caption(relationship.reason)

    if module_warnings:
        with st.expander("Module Warnings"):
            for module_name, warnings in module_warnings.items():
                for warning in warnings:
                    st.markdown(f"- **{module_name}:** {warning}")

    if module_diagnostics:
        with st.expander("Data Diagnostics"):
            for module_name, diagnostic in module_diagnostics.items():
                st.markdown(f"**{module_name}**")
                st.caption(f"Rows: {diagnostic.row_count}")
                if diagnostic.mapped_columns:
                    mapped_pairs = ", ".join(
                        f"{original} -> {mapped}"
                        for original, mapped in diagnostic.mapped_columns.items()
                    )
                    st.caption(f"Mapped: {mapped_pairs}")
                st.caption(
                    "Available normalized columns: "
                    + ", ".join(diagnostic.normalized_columns)
                )
                if diagnostic.missing_required_columns:
                    st.caption(
                        "Missing required columns: "
                        + ", ".join(diagnostic.missing_required_columns)
                    )
                if diagnostic.original_columns:
                    st.caption("Original columns: " + ", ".join(diagnostic.original_columns))
                if diagnostic.sample_rows:
                    st.dataframe(pd.DataFrame(diagnostic.sample_rows), width="stretch")

if data_source_key == "uploaded":
    st.success("Using uploaded CSV data.")
else:
    st.info("Using default synthetic demo data.")

if leads_df.empty and deals_df.empty:
    st.warning(
        "Both uploaded datasets are empty. Metrics, charts, and answers will show limited results."
    )
elif leads_df.empty or deals_df.empty:
    st.warning(
        "One of the datasets is empty. Some analytics and Ask InsightFlow AI answers may be limited."
    )

data_health_report = build_data_health_report(
    loaded_modules=loaded_modules,
    capabilities=available_capabilities,
    relationships=module_relationships,
    module_diagnostics=module_diagnostics,
)

auto_insights = generate_auto_insights(leads_df, deals_df)

data_signature = build_data_signature(leads_df, deals_df, loaded_modules=loaded_modules)
if "rag_data_signature" not in st.session_state:
    st.session_state.rag_data_signature = None
if "rag_refresh_status" not in st.session_state:
    st.session_state.rag_refresh_status = None
if "rag_refresh_attempted_signature" not in st.session_state:
    st.session_state.rag_refresh_attempted_signature = None

if st.session_state.rag_refresh_attempted_signature != data_signature:
    st.session_state.rag_refresh_status = refresh_rag_store(
        leads_df,
        deals_df,
        auto_insights=auto_insights,
    )
    st.session_state.rag_refresh_attempted_signature = data_signature
    if st.session_state.rag_refresh_status.get("available"):
        st.session_state.rag_data_signature = data_signature

with st.sidebar:
    st.header("Local AI / RAG Status")
    ollama_available = check_ollama_available()
    text_model_available = check_ollama_model_available(OLLAMA_TEXT_MODEL)
    embed_model_available = check_ollama_model_available(OLLAMA_EMBED_MODEL)
    rag_status = get_rag_status()
    if st.session_state.rag_refresh_status is not None:
        rag_status.update(st.session_state.rag_refresh_status)

    st.markdown(f"- **Ollama available:** {'Yes' if ollama_available else 'No'}")
    st.markdown(f"- **{OLLAMA_TEXT_MODEL} available:** {'Yes' if text_model_available else 'No'}")
    st.markdown(f"- **{OLLAMA_EMBED_MODEL} available:** {'Yes' if embed_model_available else 'No'}")
    st.markdown(f"- **ChromaDB ready:** {'Yes' if rag_status.get('available') else 'No'}")
    st.markdown(f"- **Indexed documents:** {rag_status.get('document_count', 0)}")

    if st.button("Refresh Local Knowledge Base", width="stretch"):
        st.session_state.rag_refresh_status = refresh_rag_store(
            leads_df,
            deals_df,
            auto_insights=auto_insights,
        )
        st.session_state.rag_refresh_attempted_signature = data_signature
        if st.session_state.rag_refresh_status.get("available"):
            st.session_state.rag_data_signature = data_signature

    if st.button("Reset Local Knowledge Base", width="stretch"):
        st.session_state.rag_refresh_status = reset_rag_store()
        st.session_state.rag_data_signature = None
        st.session_state.rag_refresh_attempted_signature = None

    if rag_status.get("last_error"):
        st.warning(rag_status["last_error"])
    if st.session_state.rag_data_signature != data_signature:
        st.warning("Data changed. Refresh local knowledge base to update RAG context.")
    if not rag_status.get("available"):
        st.info(
            "Local knowledge base is unavailable. The app will continue with Pandas-based metrics and charts."
        )

num_leads = get_total_leads(leads_df)
num_deals = get_total_deals(deals_df)
total_estimated_value = get_total_estimated_lead_value(leads_df)
total_deal_amount = get_total_deal_amount(deals_df)

kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
with kpi_col1:
    display_aed_metric("Leads", f"{num_leads:,}")
with kpi_col2:
    display_aed_metric("Deals", f"{num_deals:,}")
with kpi_col3:
    display_aed_metric("Total Estimated Lead Value", format_aed(total_estimated_value))
with kpi_col4:
    display_aed_metric("Total Deal Amount", format_aed(total_deal_amount))

leads_tab, deals_tab, health_tab, analytics_tab, charts_tab, insights_tab, ask_tab = st.tabs(
    [
        "Leads",
        "Deals",
        "Data Readiness",
        "Analytics Preview",
        "Charts",
        "Auto Insights",
        "Ask InsightFlow AI",
    ]
)

with leads_tab:
    st.markdown("<p class='section-kicker'>Source records</p><p class='section-title'>Lead details</p><p class='section-copy'>Inspect the active lead data behind the dashboard metrics and charts.</p>", unsafe_allow_html=True)
    with st.expander("View leads data", expanded=True):
        st.dataframe(leads_df, width="stretch", hide_index=True)
        display_csv_download(leads_df, "leads", "export_leads")

with deals_tab:
    st.markdown("<p class='section-kicker'>Source records</p><p class='section-title'>Deal details</p><p class='section-copy'>Inspect the active deal data behind the pipeline and forecast views.</p>", unsafe_allow_html=True)
    with st.expander("View deals data", expanded=True):
        st.dataframe(deals_df, width="stretch", hide_index=True)
        display_csv_download(deals_df, "deals", "export_deals")

with health_tab:
    display_data_readiness(data_health_report)

with analytics_tab:
    st.markdown("<p class='section-kicker'>Analytics preview</p><p class='section-title'>Pipeline at a glance</p><p class='section-copy'>Primary visuals lead; detailed deterministic tables remain available on demand.</p>", unsafe_allow_html=True)
    preview_left, preview_right = st.columns(2)
    with preview_left:
        display_chart(
            "Deals by Stage",
            deals_df,
            render_key="analytics_deals_by_stage",
            emphasis="primary",
        )
    with preview_right:
        display_chart(
            "Forecast Summary",
            deals_df,
            render_key="analytics_forecast_summary",
            emphasis="secondary",
        )
    display_table_section(
        "Leads by Source",
        get_leads_by_source(leads_df),
        ["Total_Estimated_Value"],
    )
    display_table_section(
        "Deals by Stage",
        get_deals_by_stage(deals_df),
        ["Total_Amount"],
    )
    display_table_section(
        "Forecast Summary",
        get_forecast_summary(deals_df),
        ["Total_Amount"],
    )
    display_table_section(
        "Sales Rep Performance",
        get_sales_rep_performance(leads_df, deals_df),
        ["Total_Estimated_Value", "Total_Deal_Amount"],
    )

with charts_tab:
    st.markdown("<p class='section-kicker'>Visual analysis</p><p class='section-title'>Revenue and pipeline views</p><p class='section-copy'>Charts are generated from the same deterministic Pandas summaries used across InsightFlow.</p>", unsafe_allow_html=True)
    display_chart(
        "Sales Rep Performance",
        leads_df,
        deals_df,
        render_key="charts_sales_rep_performance",
        emphasis="primary",
        highlight_top=True,
    )

    comparison_col, composition_col = st.columns(2)
    with comparison_col:
        display_chart(
            "Leads by Source",
            leads_df,
            render_key="charts_leads_by_source",
            emphasis="supporting",
            highlight_top=True,
        )
    with composition_col:
        display_chart(
            "Deals by Stage",
            deals_df,
            render_key="charts_deals_by_stage",
            emphasis="secondary",
        )

    forecast_col, product_col = st.columns(2)
    with forecast_col:
        display_chart(
            "Forecast Summary",
            deals_df,
            render_key="charts_forecast_summary",
            emphasis="secondary",
        )

    if "Product_Category" in deals_df.columns:
        with product_col:
            display_chart(
                "Product Category Revenue",
                deals_df,
                render_key="charts_product_category_revenue",
                emphasis="secondary",
                highlight_top=True,
            )
    else:
        with product_col:
            st.info("Upload deals with `Product_Category` to unlock the product category revenue chart.")

    if "Region" in deals_df.columns or "Region" in leads_df.columns:
        display_chart(
            "Region Pipeline",
            leads_df,
            deals_df,
            render_key="charts_region_pipeline",
            emphasis="supporting",
            highlight_top=True,
        )
    else:
        st.info("Upload leads or deals with `Region` to unlock the region pipeline chart.")

with insights_tab:
    st.subheader("Auto Insights")
    st.caption("Pandas-generated insights from the active leads and deals dataset.")
    display_csv_download(
        insights_to_dataframe(auto_insights),
        label="auto insights summary",
        key="export_auto_insights",
        button_label="Export Auto Insights",
    )

    insight_columns = st.columns(2)

    for index, insight in enumerate(auto_insights):
        with insight_columns[index % 2]:
            display_insight_card(insight)

with ask_tab:
    if "ask_question" not in st.session_state:
        st.session_state.ask_question = ""
    if "last_answer_context" not in st.session_state:
        st.session_state.last_answer_context = None
    if "followup_chat_history" not in st.session_state:
        st.session_state.followup_chat_history = []
    if "last_main_question" not in st.session_state:
        st.session_state.last_main_question = None

    st.subheader("Ask InsightFlow AI")
    st.caption(
        "Metrics, tables, and charts are computed with Pandas. Plotly renders the presentation layer. Ollama, when enabled, only explains computed output."
    )

    with st.expander("Sample supported questions"):
        for example_index, example in enumerate(ASK_EXAMPLES):
            if st.button(example, key=f"ask_example_{example_index}", width="stretch"):
                st.session_state.ask_question = example

    user_question = st.text_area(
        "Your question",
        key="ask_question",
        placeholder="Example: Show revenue by region",
        height=100,
    )

    if st.button("Ask", type="primary"):
        if not user_question.strip():
            st.warning("Please enter a question first.")
        else:
            if st.session_state.last_main_question != user_question.strip():
                st.session_state.followup_chat_history = []
                st.session_state.last_main_question = user_question.strip()
            question_result = resolve_question_response(
                user_question,
                leads_df,
                deals_df,
                auto_insights,
                st.session_state.rag_data_signature == data_signature,
                extra_tables=loaded_modules,
            )
            if question_result["answer_type"] != "unsupported":
                st.session_state["active_business_context"] = question_result.get(
                    "active_business_context",
                    {
                        "metric": question_result.get("metric_type"),
                        "dataset": question_result.get("business_object"),
                        "dimension": question_result.get("parsed_query", {}).get("dimension"),
                        "aggregation": question_result.get("parsed_query", {}).get("aggregation"),
                        "filters": question_result.get("parsed_query", {}).get("filters", {}),
                        "chart_type": question_result.get("chart_type"),
                        "answer_title": question_result.get("title"),
                        "computed_answer": question_result.get("answer"),
                    },
                )
                st.session_state.last_answer_context = build_last_answer_context(
                    original_user_question=user_question.strip(),
                    result=question_result,
                    leads_df=leads_df,
                    deals_df=deals_df,
                    data_signature=data_signature,
                )
            with st.container(border=True):
                display_question_answer(question_result)

    if st.session_state.last_answer_context is not None:
        st.divider()
        followup_col1, followup_col2 = st.columns([3, 1])
        with followup_col1:
            st.markdown("### Ask a follow-up about this result")
        with followup_col2:
            if st.button("Clear follow-up chat", width="stretch"):
                st.session_state.followup_chat_history = []

        for message in st.session_state.followup_chat_history:
            with st.container(border=True):
                st.markdown(f"**{message.get('role', 'Follow-up')}**")
                st.markdown(message.get("content", ""))
                if message.get("answer_source"):
                    st.caption(f"Answer source: {message['answer_source']}")
                elif message.get("source"):
                    st.caption(f"Answer source: {message['source']}")

        with st.form("followup_form", clear_on_submit=True):
            followup_question = st.text_input(
                "Ask a follow-up",
                placeholder="Example: Why does this matter for management?",
                key="followup_question_input",
            )
            submitted_followup = st.form_submit_button("Ask follow-up")

        if submitted_followup:
            cleaned_followup_question = followup_question.strip()

            if cleaned_followup_question:
                followup_result = answer_followup_question(
                    followup_question=cleaned_followup_question,
                    last_answer_context=st.session_state.get("last_answer_context"),
                    leads_df=leads_df,
                    deals_df=deals_df,
                    active_business_context=st.session_state.get("active_business_context"),
                    extra_tables=loaded_modules,
                )

                if (
                    isinstance(followup_result, dict)
                    and followup_result.get("answer_type") != "unsupported"
                    and followup_result.get("followup_intent") == "analytics"
                ):
                    st.session_state["active_business_context"] = followup_result.get(
                        "active_business_context",
                        st.session_state.get("active_business_context"),
                    )
                    st.session_state["last_answer_context"] = build_last_answer_context(
                        original_user_question=cleaned_followup_question,
                        result=followup_result,
                        leads_df=leads_df,
                        deals_df=deals_df,
                        data_signature=data_signature,
                    )

                if "followup_chat_history" not in st.session_state:
                    st.session_state["followup_chat_history"] = []

                st.session_state["followup_chat_history"].append(
                    {
                        "role": "User",
                        "content": cleaned_followup_question,
                    }
                )
                st.session_state["followup_chat_history"].append(
                    {
                        "role": "InsightFlow AI",
                        "content": (
                            followup_result.get("answer", followup_result)
                            if isinstance(followup_result, dict)
                            else str(followup_result)
                        ),
                        "answer_source": (
                            followup_result.get("answer_source")
                            if isinstance(followup_result, dict)
                            else "follow-up"
                        ),
                    }
                )

                st.rerun()
            else:
                st.warning("Please enter a follow-up question first.")
