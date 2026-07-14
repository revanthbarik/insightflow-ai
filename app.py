"""Streamlit entry point for InsightFlow AI."""

from __future__ import annotations

import pandas as pd
import re
import streamlit as st
from pandas.util import hash_pandas_object

from src.analytics import (
    answer_quantitative_question,
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
from src.charting import (
    create_deals_by_stage_chart,
    create_forecast_chart,
    create_leads_by_source_chart,
    render_categorical_chart,
    create_product_category_revenue_chart,
    create_region_pipeline_chart,
    create_sales_rep_chart,
)
from src.data_loader import (
    empty_deals_df,
    empty_leads_df,
    load_all_data,
    load_zoho_modules_from_files,
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
    explain_with_ollama,
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

CHART_KEY_FUNCTIONS = {
    "leads_by_source": create_leads_by_source_chart,
    "deals_by_stage": create_deals_by_stage_chart,
    "forecast_summary": create_forecast_chart,
    "sales_rep_performance": create_sales_rep_chart,
    "product_category_revenue": create_product_category_revenue_chart,
    "region_pipeline": create_region_pipeline_chart,
}

CHART_KEY_ARGS = {
    "leads_by_source": lambda leads_df, deals_df: (leads_df,),
    "deals_by_stage": lambda leads_df, deals_df: (deals_df,),
    "forecast_summary": lambda leads_df, deals_df: (deals_df,),
    "sales_rep_performance": lambda leads_df, deals_df: (leads_df, deals_df),
    "product_category_revenue": lambda leads_df, deals_df: (deals_df,),
    "region_pipeline": lambda leads_df, deals_df: (leads_df, deals_df),
}

CHART_DATA_BUILDERS = {
    "Leads by Source": lambda *chart_args: get_leads_by_source(chart_args[0]),
    "Deals by Stage": lambda *chart_args: get_deals_by_stage(chart_args[0]),
    "Forecast Summary": lambda *chart_args: get_forecast_summary(chart_args[0]),
    "Sales Rep Performance": lambda *chart_args: get_sales_rep_performance(chart_args[0], chart_args[1]),
    "Product Category Revenue": lambda *chart_args: get_product_category_revenue(chart_args[0]),
    "Region Pipeline": lambda *chart_args: get_region_pipeline(chart_args[0], chart_args[1]),
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
    """Show a metric with a full-width AED value that will not be truncated."""
    st.markdown(
        f"""
        <div>
            <p style="margin: 0 0 0.25rem 0; font-size: 0.875rem; color: rgb(49, 51, 63);">
                {label}
            </p>
            <p style="margin: 0; font-size: 1.75rem; font-weight: 600; white-space: nowrap;">
                {formatted_value}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_chart(title: str, chart_function, *args) -> None:
    """Generate and display a saved chart, or show a clean error message."""
    st.subheader(title)
    data_builder = CHART_DATA_BUILDERS.get(title)
    if data_builder is not None:
        chart_data = data_builder(*args)
        if chart_data.empty:
            st.info(
                get_empty_reason(
                    chart_data,
                    f"No data is available for the {title.lower()} chart.",
                )
            )
            available_columns = chart_data.attrs.get("available_columns", [])
            if available_columns:
                st.caption(f"Available columns: {', '.join(available_columns)}")
            applied_filters = chart_data.attrs.get("applied_filters", [])
            for filter_description in applied_filters:
                st.caption(filter_description)
            return
    try:
        chart_path = chart_function(*args)
        st.image(str(chart_path), use_container_width=True)
    except Exception as error:
        st.error(
            f"Could not generate the {title.lower()} right now. "
            f"Please check the uploaded data and try again. Details: {error}"
        )


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
        st.dataframe(
            format_table_for_display(
                support_df,
                TABLE_CURRENCY_COLUMNS.get(result["title"], []),
            ),
            use_container_width=True,
        )


def display_question_answer(result: dict, leads_df, deals_df) -> None:
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
            st.dataframe(
                format_table_for_display(result["data"], currency_columns),
                use_container_width=True,
            )
    elif result["answer_type"] == "chart":
        chart_key = result.get("chart_key")
        if chart_key == "monthly_pipeline_trend":
            chart_path = render_categorical_chart(
                data=result["data"] if result.get("data") is not None else pd.DataFrame(),
                x_column="Closing_Month",
                y_column="Total_Amount",
                title=result["title"],
                chart_filename="monthly_pipeline_trend.png",
                color="#355C7D",
                chart_type=result.get("chart_type", "line"),
                sort_desc=False,
                empty_message="No monthly trend data is available for this chart.",
                x_label="Closing Month",
                y_label="Total Amount",
            )
            st.image(str(chart_path), use_container_width=True)
            if result.get("data") is not None and not result["data"].empty:
                st.dataframe(
                    format_table_for_display(result["data"], ["Total_Amount"]),
                    use_container_width=True,
                )
        elif result.get("data") is not None and not result["data"].empty and result.get("x_axis") and result.get("y_axis"):
            chart_filename = re.sub(r"[^a-z0-9_]+", "_", result.get("chart_title", result["title"]).lower()).strip("_") + ".png"
            chart_path = render_categorical_chart(
                data=result["data"],
                x_column=result["x_axis"],
                y_column=result["y_axis"],
                title=result.get("chart_title", result["title"]),
                chart_filename=chart_filename,
                color="#355C7D",
                chart_type=result.get("chart_type", "bar"),
                sort_desc=result.get("chart_type", "bar") != "line",
                empty_message="No data is available for this chart.",
                x_label=result.get("x_axis", "").replace("_", " "),
                y_label=result.get("y_axis", "").replace("_", " "),
            )
            st.image(str(chart_path), use_container_width=True)
            currency_columns = [
                column
                for column in result["data"].columns
                if any(token in column.lower() for token in ("amount", "value", "margin"))
            ]
            st.dataframe(
                format_table_for_display(result["data"], currency_columns),
                use_container_width=True,
            )
        else:
            chart_function = CHART_KEY_FUNCTIONS.get(chart_key)
            chart_args_builder = CHART_KEY_ARGS.get(chart_key)
            if chart_function is None or chart_args_builder is None:
                st.error("This chart type is not supported yet.")
            else:
                chart_path = chart_function(*chart_args_builder(leads_df, deals_df))
                st.image(str(chart_path), use_container_width=True)
    elif result["answer_type"] == "dynamic_chart":
        st.image(str(result["chart_path"]), use_container_width=True)
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


def display_table_section(title: str, df: pd.DataFrame, currency_columns: list[str]) -> None:
    """Display a table section with a safe empty-data message."""
    st.subheader(title)
    if df.empty:
        st.info(get_empty_reason(df, f"No data is available for {title.lower()}."))
        return
    st.dataframe(
        format_table_for_display(df, currency_columns),
        use_container_width=True,
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


def maybe_display_ollama_explanation(result: dict, enabled: bool) -> None:
    """Show optional Ollama explanation using computed context only."""
    if not enabled or result["answer_type"] == "unsupported":
        return

    context = {
        "title": result["title"],
        "answer_source": result.get("answer_source"),
        "answer": result["answer"],
        "recommended_action": result.get("recommended_action"),
        "currency": "AED",
        "summary_table_records": (
            result["data"].head(5).to_dict(orient="records")
            if isinstance(result.get("data"), pd.DataFrame)
            else None
        ),
    }

    if result.get("chart_key") == "monthly_pipeline_trend" and isinstance(
        result.get("data"), pd.DataFrame
    ) and not result["data"].empty:
        trend_df = result["data"].copy()
        top_row = trend_df.sort_values("Total_Amount", ascending=False).iloc[0]
        context.update(
            {
                "highest_month": top_row["Closing_Month"],
                "highest_amount": float(top_row["Total_Amount"]),
            }
        )

    explanation = explain_with_ollama(context)
    if explanation:
        st.caption("Answer source: Ollama explanation")
        st.markdown(explanation)
    else:
        st.info("Ollama explanation is unavailable right now. The computed InsightFlow answer is still shown above.")


st.set_page_config(page_title="InsightFlow AI", layout="wide")

st.title("InsightFlow AI — Private Revenue Operations Agent")

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
                    st.dataframe(pd.DataFrame(diagnostic.sample_rows), use_container_width=True)

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

    if st.button("Refresh Local Knowledge Base", use_container_width=True):
        st.session_state.rag_refresh_status = refresh_rag_store(
            leads_df,
            deals_df,
            auto_insights=auto_insights,
        )
        st.session_state.rag_refresh_attempted_signature = data_signature
        if st.session_state.rag_refresh_status.get("available"):
            st.session_state.rag_data_signature = data_signature

    if st.button("Reset Local Knowledge Base", use_container_width=True):
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

count_col1, count_col2 = st.columns(2)
count_col1.metric("Leads", f"{num_leads:,}")
count_col2.metric("Deals", f"{num_deals:,}")

value_col1, value_col2 = st.columns(2)
with value_col1:
    display_aed_metric(
        "Total Estimated Lead Value",
        format_aed(total_estimated_value),
    )
with value_col2:
    display_aed_metric("Total Deal Amount", format_aed(total_deal_amount))

leads_tab, deals_tab, analytics_tab, charts_tab, insights_tab, ask_tab = st.tabs(
    ["Leads", "Deals", "Analytics Preview", "Charts", "Auto Insights", "Ask InsightFlow AI"]
)

with leads_tab:
    st.dataframe(leads_df, use_container_width=True)

with deals_tab:
    st.dataframe(deals_df, use_container_width=True)

with analytics_tab:
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
    display_chart("Leads by Source", create_leads_by_source_chart, leads_df)
    display_chart("Deals by Stage", create_deals_by_stage_chart, deals_df)
    display_chart("Forecast Summary", create_forecast_chart, deals_df)
    display_chart(
        "Sales Rep Performance",
        create_sales_rep_chart,
        leads_df,
        deals_df,
    )

    if "Product_Category" in deals_df.columns:
        display_chart(
            "Product Category Revenue",
            create_product_category_revenue_chart,
            deals_df,
        )
    else:
        st.info("Upload deals with `Product_Category` to unlock the product category revenue chart.")

    if "Region" in deals_df.columns or "Region" in leads_df.columns:
        display_chart(
            "Region Pipeline",
            create_region_pipeline_chart,
            leads_df,
            deals_df,
        )
    else:
        st.info("Upload leads or deals with `Region` to unlock the region pipeline chart.")

with insights_tab:
    st.subheader("Auto Insights")
    st.caption("Pandas-generated insights from the active leads and deals dataset.")

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
        "Metrics, tables, and charts are computed with Pandas. Matplotlib renders charts. Ollama, when enabled, only explains computed output."
    )

    with st.expander("Sample supported questions"):
        for example_index, example in enumerate(ASK_EXAMPLES):
            if st.button(example, key=f"ask_example_{example_index}", use_container_width=True):
                st.session_state.ask_question = example

    ollama_explanations_enabled = st.checkbox(
        "Add Ollama explanation to the computed answer",
        value=False,
        help="This only explains already-computed results. It does not calculate metrics or generate Pandas code.",
    )

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
                display_question_answer(question_result, leads_df, deals_df)

    if st.session_state.last_answer_context is not None:
        st.divider()
        followup_col1, followup_col2 = st.columns([3, 1])
        with followup_col1:
            st.markdown("### Ask a follow-up about this result")
        with followup_col2:
            if st.button("Clear follow-up chat", use_container_width=True):
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
                        "question": cleaned_followup_question,
                        "answer": (
                            followup_result.get("answer", followup_result)
                            if isinstance(followup_result, dict)
                            else followup_result
                        ),
                        "source": (
                            followup_result.get("answer_source", "follow-up")
                            if isinstance(followup_result, dict)
                            else "follow-up"
                        ),
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
