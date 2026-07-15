"""Local ChromaDB vector store for grounded InsightFlow business context."""

from __future__ import annotations

import shutil
import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.analytics import (
    format_aed_compact,
    get_forecast_summary,
    get_lead_source_value_vs_conversion,
    get_leads_by_source,
    get_monthly_amount_trend,
    get_region_pipeline,
    get_sales_rep_performance,
    get_weakest_forecast_quality_by_product_category,
)
from src.config import (
    CHROMA_DB_DIR,
    RAG_COLLECTION_NAME,
    embedding_provider_identity,
)
from src.llm import get_local_embedding

try:
    import chromadb
except ImportError:  # pragma: no cover - safe runtime fallback
    chromadb = None


_LAST_RAG_STATUS: dict[str, Any] = {
    "available": chromadb is not None,
    "collection_name": "",
    "path": str(CHROMA_DB_DIR),
    "document_count": 0,
    "last_error": None,
}


def get_embedding_collection_name() -> str:
    """Isolate Chroma collections across embedding providers and models."""
    provider, model = embedding_provider_identity()
    suffix = re.sub(r"[^a-z0-9_-]+", "_", f"{provider}_{model}".lower()).strip("_")
    base = re.sub(r"[^a-z0-9_-]+", "_", RAG_COLLECTION_NAME.lower()).strip("_")
    collection_name = f"{base}__{suffix}"[:63].strip("_")
    return collection_name if len(collection_name) >= 3 else "insightflow_context"


_LAST_RAG_STATUS["collection_name"] = get_embedding_collection_name()

FORECAST_CATEGORY_DEFINITIONS = {
    "Pipeline": "Pipeline contains open opportunities that are active but still early or uncertain and should be reviewed for progression risk.",
    "Best Case": "Best Case contains opportunities that could close with strong execution but still need active follow-through from sales and management.",
    "Commit": "Commit contains deals that the team currently expects to close and should be checked for delivery readiness and execution confidence.",
    "Closed": "Closed represents finished won business that should be treated as recognized closed revenue in management reporting.",
    "At Risk": "At Risk flags opportunities that may slip, stall, or be lost and should be escalated for immediate review.",
    "Omitted": "Omitted captures deals that are not expected to close in the forecast window and should not be relied on for near-term planning.",
}

DEAL_STAGE_DEFINITIONS = {
    "New Inquiry": "New Inquiry is the earliest stage where an incoming opportunity has been captured but not yet deeply qualified.",
    "Technical Discussion": "Technical Discussion means solution fit, engineering, or product requirements are being validated with the customer.",
    "Quotation Sent": "Quotation Sent means pricing has been shared and management should track follow-up quality and next-step speed.",
    "Negotiation": "Negotiation means commercial terms are under active discussion and deal progression risk should be watched closely.",
    "PO Received": "PO Received means the customer has issued a purchase order and the business should prepare fulfillment, invoicing, and delivery execution.",
    "Closed Won": "Closed Won means the opportunity converted successfully and contributes to closed revenue performance.",
    "Closed Lost": "Closed Lost means the opportunity did not convert and should be reviewed for lessons, blockers, and pattern analysis.",
}

SCHEMA_DOCUMENTS = [
    ("schema_leads", "leads", "schema", "Leads represent early-stage opportunities before they become booked deals in the revenue pipeline."),
    ("schema_deals", "deals", "schema", "Deals represent active or closed commercial opportunities that carry forecast stage, amount, and closing expectations."),
    ("field_lead_id", "schema", "field", "Lead_ID is the shared identifier that links a lead record to downstream deal records for conversion analysis."),
    ("field_deal_id", "schema", "field", "Deal_ID uniquely identifies each commercial opportunity in the deals dataset."),
    ("field_estimated_value", "schema", "field", "Estimated_Value is the lead-side commercial potential used for lead quality and conversion analysis."),
    ("field_amount", "schema", "field", "Amount is the deal-side monetary value used for revenue, pipeline, forecast, and trend analysis."),
    ("field_forecast_category", "schema", "field", "Forecast_Category shows how management should interpret deal confidence and likely closing behavior."),
    ("field_deal_stage", "schema", "field", "Deal_Stage shows where the opportunity sits in the commercial process and how close it is to conversion."),
    ("field_product_category", "schema", "field", "Product_Category groups deals by solution line so revenue and forecast quality can be compared across offerings."),
    ("field_region", "schema", "field", "Region helps management compare pipeline concentration and commercial activity across geographies."),
    ("field_sales_rep", "schema", "field", "Sales_Rep identifies the owner or responsible salesperson for lead and deal performance tracking."),
    ("field_lead_source", "schema", "field", "Lead_Source identifies where leads originated so management can compare quality, conversion, and estimated value."),
]

METRIC_DEFINITIONS = {
    "Total pipeline value": "Total pipeline value is the sum of deal Amount for opportunities categorized as Pipeline. Pandas calculates the value before any explanation step.",
    "Committed forecast": "Committed forecast is the sum of deal Amount for opportunities tagged Commit and represents the team’s most reliable near-term closing expectation.",
    "Closed won revenue": "Closed won revenue is the sum of Amount for deals marked Closed Won and is used to evaluate realized commercial performance.",
    "At-risk pipeline": "At-risk pipeline is the sum of Amount for deals marked At Risk and highlights where management should review slippage or execution risk.",
    "Average deal size": "Average deal size is the mean deal Amount across the relevant deal set and is computed in Pandas for benchmarking commercial motion.",
    "Win rate": "Win rate compares Closed Won deals against the total of Closed Won plus Closed Lost deals and should be interpreted together with deal volume and stage mix.",
    "Forecast quality": "Forecast quality compares weak forecast categories against total opportunity value to show how much of a category or pipeline is still uncertain.",
    "Lead conversion rate": "Lead conversion rate compares unique leads that created deals against total leads from a source or cohort, using Lead_ID as the join key.",
    "Product category revenue": "Product category revenue groups deal Amount by Product_Category so management can compare commercial concentration across solution lines.",
    "Regional pipeline": "Regional pipeline groups open pipeline value by Region so management can assess geographic exposure and opportunity distribution.",
    "Sales rep performance": "Sales rep performance compares lead ownership and downstream deal amount by Sales_Rep to highlight commercial productivity and workload.",
    "Monthly pipeline trend": "Monthly pipeline trend groups deal Amount by closing month and fills missing months with zero to show how expected closing value changes over time.",
}


def _set_status(**updates: Any) -> dict[str, Any]:
    """Update and return the current RAG status snapshot."""
    _LAST_RAG_STATUS.update(updates)
    return dict(_LAST_RAG_STATUS)


def initialize_rag_store() -> dict[str, Any]:
    """Initialize the local persistent ChromaDB collection safely."""
    if chromadb is None:
        return _set_status(
            available=False,
            initialized=False,
            last_error="chromadb is not installed.",
        )

    try:
        Path(CHROMA_DB_DIR).mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        collection = client.get_or_create_collection(name=get_embedding_collection_name())
        return _set_status(
            available=True,
            initialized=True,
            last_error=None,
            document_count=collection.count(),
        )
    except Exception as error:  # pragma: no cover - runtime safety
        return _set_status(
            available=False,
            initialized=False,
            last_error=f"Failed to initialize ChromaDB: {error}",
        )


def _get_collection():
    """Return the initialized Chroma collection or None on failure."""
    status = initialize_rag_store()
    if not status.get("available"):
        return None
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        return client.get_or_create_collection(name=get_embedding_collection_name())
    except Exception as error:  # pragma: no cover - runtime safety
        _set_status(available=False, last_error=f"Failed to access collection: {error}")
        return None


def reset_rag_store() -> dict[str, Any]:
    """Delete and recreate the persistent local RAG store."""
    if chromadb is None:
        return _set_status(available=False, initialized=False, last_error="chromadb is not installed.")

    try:
        if Path(CHROMA_DB_DIR).exists():
            shutil.rmtree(CHROMA_DB_DIR)
    except Exception as error:  # pragma: no cover - runtime safety
        return _set_status(available=False, initialized=False, last_error=f"Failed to reset RAG store: {error}")

    return initialize_rag_store()


def _make_document(doc_id: str, text: str, source_type: str, module: str, topic: str) -> dict[str, Any]:
    """Create a consistent RAG document payload."""
    return {
        "id": doc_id,
        "text": text,
        "metadata": {
            "source_type": source_type,
            "module": module,
            "topic": topic,
            "created_by": "system",
        },
    }


def _build_schema_documents() -> list[dict[str, Any]]:
    """Build static schema and field-definition documents."""
    documents: list[dict[str, Any]] = []
    for doc_id, source_type, topic, text in SCHEMA_DOCUMENTS:
        documents.append(_make_document(doc_id, text, source_type, "schema", topic))
    return documents


def _build_forecast_documents() -> list[dict[str, Any]]:
    """Build forecast-category definition documents."""
    return [
        _make_document(
            f"forecast_{category.lower().replace(' ', '_')}",
            f"{category}: {definition}",
            "forecast_definition",
            "forecast",
            category,
        )
        for category, definition in FORECAST_CATEGORY_DEFINITIONS.items()
    ]


def _build_stage_documents() -> list[dict[str, Any]]:
    """Build deal-stage definition documents."""
    return [
        _make_document(
            f"deal_stage_{stage.lower().replace(' ', '_')}",
            f"{stage}: {definition}",
            "deal_stage_definition",
            "deal_stage",
            stage,
        )
        for stage, definition in DEAL_STAGE_DEFINITIONS.items()
    ]


def _build_metric_documents() -> list[dict[str, Any]]:
    """Build conceptual metric-definition documents."""
    return [
        _make_document(
            f"metric_{metric.lower().replace(' ', '_')}",
            f"{metric}: {definition}",
            "metric_definition",
            "metrics",
            metric,
        )
        for metric, definition in METRIC_DEFINITIONS.items()
    ]


def _top_row_text(df: pd.DataFrame, label_column: str, value_column: str) -> str | None:
    """Return a short top-row summary from a grouped DataFrame."""
    if df.empty or label_column not in df.columns or value_column not in df.columns:
        return None
    row = df.iloc[0]
    return f"{row[label_column]} at {format_aed_compact(row[value_column])}"


def _build_summary_documents(
    leads_df: pd.DataFrame,
    deals_df: pd.DataFrame,
    auto_insights: list[dict] | None,
) -> list[dict[str, Any]]:
    """Build compact business summary documents from Pandas-computed results."""
    documents: list[dict[str, Any]] = []

    lead_source_summary = get_leads_by_source(leads_df)
    top_lead_source = _top_row_text(
        lead_source_summary.sort_values("Total_Estimated_Value", ascending=False),
        "Lead_Source",
        "Total_Estimated_Value",
    )
    if top_lead_source:
        documents.append(
            _make_document(
                "summary_top_lead_source",
                (
                    "Current lead source summary: the top lead source by estimated value is "
                    f"{top_lead_source}. These values were computed by Pandas from the active leads data."
                ),
                "business_summary",
                "summaries",
                "lead_source",
            )
        )

    forecast_summary = get_forecast_summary(deals_df)
    if not forecast_summary.empty:
        forecast_bits = [
            f"{row['Forecast_Category']} is {format_aed_compact(row['Total_Amount'])}"
            for _, row in forecast_summary.head(6).iterrows()
        ]
        documents.append(
            _make_document(
                "summary_forecast",
                (
                    "Current forecast summary: " + "; ".join(forecast_bits)
                    + ". These values were computed by Pandas from the active deals data."
                ),
                "business_summary",
                "summaries",
                "forecast",
            )
        )

    rep_summary = get_sales_rep_performance(leads_df, deals_df)
    if not rep_summary.empty:
        top_rep = rep_summary.sort_values("Total_Deal_Amount", ascending=False).iloc[0]
        documents.append(
            _make_document(
                "summary_sales_rep",
                (
                    "Current sales rep performance summary: "
                    f"{top_rep['Sales_Rep']} leads deal amount at {format_aed_compact(top_rep['Total_Deal_Amount'])}. "
                    "This summary was computed by Pandas from the active leads and deals data."
                ),
                "business_summary",
                "summaries",
                "sales_rep",
            )
        )

    if "Product_Category" in deals_df.columns:
        product_summary = deals_df.groupby("Product_Category", as_index=False).agg(
            Total_Amount=("Amount", "sum")
        ).sort_values("Total_Amount", ascending=False)
        top_product = _top_row_text(product_summary, "Product_Category", "Total_Amount")
        if top_product:
            documents.append(
                _make_document(
                    "summary_top_product",
                    (
                        "Current top product category by revenue: "
                        f"{top_product}. This summary was computed by Pandas from the active deals data."
                    ),
                    "business_summary",
                    "summaries",
                    "product_category",
                )
            )

    region_summary = get_region_pipeline(leads_df, deals_df)
    if not region_summary.empty:
        top_region = _top_row_text(region_summary, "Region", "Total_Amount")
        if top_region:
            documents.append(
                _make_document(
                    "summary_region_pipeline",
                    (
                        "Current regional pipeline summary: the top region by pipeline value is "
                        f"{top_region}. This summary was computed by Pandas from the active data."
                    ),
                    "business_summary",
                    "summaries",
                    "region_pipeline",
                )
            )

    conversion_summary = get_lead_source_value_vs_conversion(leads_df, deals_df)
    if not conversion_summary.empty:
        row = conversion_summary.iloc[0]
        documents.append(
            _make_document(
                "summary_lead_conversion",
                (
                    "Current lead conversion summary: "
                    f"{row['Lead_Source']} has {format_aed_compact(row['Estimated_Value'])} estimated value "
                    f"with a conversion rate of {row['Conversion_Rate'] * 100:.1f}%. "
                    "These values were computed by Pandas using Lead_ID to connect leads and deals."
                ),
                "business_summary",
                "summaries",
                "lead_conversion",
            )
        )

    forecast_quality_summary = get_weakest_forecast_quality_by_product_category(deals_df)
    if not forecast_quality_summary.empty:
        row = forecast_quality_summary.iloc[0]
        documents.append(
            _make_document(
                "summary_forecast_quality",
                (
                    "Current forecast quality summary: "
                    f"{row['Product_Category']} has the weakest forecast quality with "
                    f"{format_aed_compact(row['Weak_Forecast_Amount'])} weak forecast amount "
                    f"representing {row['Weak_Forecast_Ratio'] * 100:.1f}% of its category total. "
                    "These values were computed by Pandas from the active deals data."
                ),
                "business_summary",
                "summaries",
                "forecast_quality",
            )
        )

    monthly_pipeline_summary = get_monthly_amount_trend(deals_df, forecast_filter="Pipeline")
    if not monthly_pipeline_summary.empty:
        peak_row = monthly_pipeline_summary.sort_values("Total_Amount", ascending=False).iloc[0]
        documents.append(
            _make_document(
                "summary_monthly_pipeline_trend",
                (
                    "Current monthly pipeline trend summary: "
                    f"the highest pipeline month is {peak_row['Closing_Month']} at {format_aed_compact(peak_row['Total_Amount'])}. "
                    "This trend was computed by Pandas from the active deals data and missing months were filled with zero."
                ),
                "business_summary",
                "summaries",
                "monthly_pipeline_trend",
            )
        )

    if auto_insights:
        for index, insight in enumerate(auto_insights):
            documents.append(
                _make_document(
                    f"auto_insight_{index}",
                    (
                        f"Auto Insight: {insight['title']}. Metric: {insight['metric']}. "
                        f"Explanation: {insight['explanation']} Recommended action: {insight['recommended_action']}"
                    ),
                    "auto_insight",
                    "auto_insights",
                    insight["title"],
                )
            )

    if {"Deal_ID", "Amount", "Forecast_Category"}.issubset(deals_df.columns):
        at_risk_deals = deals_df[deals_df["Forecast_Category"] == "At Risk"].sort_values(
            "Amount", ascending=False
        )
        if not at_risk_deals.empty:
            top_at_risk = [
                f"{row['Deal_ID']} at {format_aed_compact(row['Amount'])}"
                for _, row in at_risk_deals.head(10).iterrows()
            ]
            documents.append(
                _make_document(
                    "summary_high_risk_deals",
                    "Top high-value At Risk deals: " + "; ".join(top_at_risk),
                    "deal_summary",
                    "risk",
                    "at_risk_deals",
                )
            )

        commit_deals = deals_df[deals_df["Forecast_Category"] == "Commit"].sort_values(
            "Amount", ascending=False
        )
        if not commit_deals.empty:
            top_commit = [
                f"{row['Deal_ID']} at {format_aed_compact(row['Amount'])}"
                for _, row in commit_deals.head(10).iterrows()
            ]
            documents.append(
                _make_document(
                    "summary_commit_deals",
                    "Top Commit deals: " + "; ".join(top_commit),
                    "deal_summary",
                    "forecast",
                    "commit_deals",
                )
            )

    if {"Deal_ID", "Amount", "Closing_Date"}.issubset(deals_df.columns):
        upcoming_df = deals_df.copy()
        upcoming_df["Closing_Date"] = pd.to_datetime(
            upcoming_df["Closing_Date"],
            errors="coerce",
            format="mixed",
        )
        upcoming_df = upcoming_df.dropna(subset=["Closing_Date"]).sort_values(
            ["Closing_Date", "Amount"],
            ascending=[True, False],
        )
        if not upcoming_df.empty:
            top_upcoming = [
                f"{row['Deal_ID']} closing on {row['Closing_Date'].strftime('%d %b %Y')} at {format_aed_compact(row['Amount'])}"
                for _, row in upcoming_df.head(10).iterrows()
            ]
            documents.append(
                _make_document(
                    "summary_upcoming_closings",
                    "Top upcoming closing deals: " + "; ".join(top_upcoming),
                    "deal_summary",
                    "closing_date",
                    "upcoming_closings",
                )
            )

    return documents


def build_rag_documents(
    leads_df: pd.DataFrame,
    deals_df: pd.DataFrame,
    auto_insights: list[dict] | None = None,
) -> list[dict[str, Any]]:
    """Build compact, grounded documents for local qualitative retrieval."""
    documents: list[dict[str, Any]] = []
    documents.extend(_build_schema_documents())
    documents.extend(_build_forecast_documents())
    documents.extend(_build_stage_documents())
    documents.extend(_build_metric_documents())
    documents.extend(_build_summary_documents(leads_df, deals_df, auto_insights))
    return documents


def index_documents(documents: list[dict[str, Any]]) -> dict[str, Any]:
    """Embed and upsert documents into the local Chroma collection safely."""
    collection = _get_collection()
    if collection is None:
        return _set_status(available=False, last_error="RAG collection is unavailable.")

    if not documents:
        return _set_status(available=True, initialized=True, document_count=collection.count(), last_error=None)

    ids: list[str] = []
    texts: list[str] = []
    embeddings: list[list[float]] = []
    metadatas: list[dict[str, Any]] = []

    for document in documents:
        embedding = get_local_embedding(document["text"])
        if embedding is None:
            continue
        ids.append(document["id"])
        texts.append(document["text"])
        embeddings.append(embedding)
        metadatas.append(document["metadata"])

    if not ids:
        return _set_status(
            available=False,
            initialized=True,
            document_count=collection.count(),
            last_error="No documents were indexed because local embeddings are unavailable.",
        )

    try:
        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return _set_status(
            available=True,
            initialized=True,
            document_count=collection.count(),
            last_error=None,
        )
    except Exception as error:  # pragma: no cover - runtime safety
        return _set_status(
            available=False,
            initialized=True,
            last_error=f"Failed to index documents: {error}",
        )


def refresh_rag_store(
    leads_df: pd.DataFrame,
    deals_df: pd.DataFrame,
    auto_insights: list[dict] | None = None,
) -> dict[str, Any]:
    """Rebuild the local RAG store from the active dataset."""
    reset_status = reset_rag_store()
    if not reset_status.get("available"):
        return reset_status

    documents = build_rag_documents(leads_df, deals_df, auto_insights)
    status = index_documents(documents)
    status["document_blueprint_count"] = len(documents)
    return status


def query_rag_store(question: str, top_k: int = 5) -> dict[str, Any]:
    """Query the local RAG store safely."""
    collection = _get_collection()
    if collection is None:
        return {"ok": False, "results": [], "error": "RAG collection is unavailable."}

    if collection.count() == 0:
        return {"ok": True, "results": [], "error": None}

    embedding = get_local_embedding(question)
    if embedding is None:
        return {"ok": False, "results": [], "error": "Local embeddings are unavailable for this query."}

    try:
        response = collection.query(query_embeddings=[embedding], n_results=top_k)
    except Exception as error:  # pragma: no cover - runtime safety
        return {"ok": False, "results": [], "error": f"RAG query failed: {error}"}

    results: list[dict[str, Any]] = []
    documents = response.get("documents", [[]])[0]
    metadatas = response.get("metadatas", [[]])[0]
    distances = response.get("distances", [[]])[0]

    for index, text in enumerate(documents):
        metadata = metadatas[index] if index < len(metadatas) else {}
        distance = distances[index] if index < len(distances) else None
        results.append({"text": text, "metadata": metadata, "distance": distance})

    return {"ok": True, "results": results, "error": None}


def get_rag_status() -> dict[str, Any]:
    """Return the latest local RAG status snapshot."""
    status = initialize_rag_store()
    return dict(status)


def format_retrieved_context(results: list[dict[str, Any]]) -> str:
    """Format retrieved local context into a compact prompt-ready string."""
    if not results:
        return "No local context was retrieved."

    lines = []
    for index, result in enumerate(results, start=1):
        metadata = result.get("metadata", {})
        lines.append(
            f"[Context {index}] ({metadata.get('source_type', 'unknown')} | {metadata.get('topic', 'general')}) "
            f"{result.get('text', '')}"
        )
    return "\n".join(lines)
