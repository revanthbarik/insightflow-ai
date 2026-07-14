"""Semantic metric and dimension definitions for the BI query engine."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricDefinition:
    """Describe one supported business metric."""

    metric_name: str
    synonyms: tuple[str, ...]
    dataset_required: tuple[str, ...]
    required_columns: tuple[str, ...]
    default_aggregation: str
    valid_dimensions: tuple[str, ...]
    default_chart_type: str
    business_description: str


METRIC_DEFINITIONS: dict[str, MetricDefinition] = {
    "total_pipeline_value": MetricDefinition(
        metric_name="total_pipeline_value",
        synonyms=("total pipeline value", "open pipeline", "pipeline total"),
        dataset_required=("deals",),
        required_columns=("Amount", "Forecast_Category"),
        default_aggregation="sum",
        valid_dimensions=(),
        default_chart_type="bar",
        business_description="Total deal amount currently tagged as Pipeline.",
    ),
    "revenue": MetricDefinition(
        metric_name="revenue",
        synonyms=("revenue", "sales", "amount", "deal value", "value"),
        dataset_required=("deals",),
        required_columns=("Amount",),
        default_aggregation="sum",
        valid_dimensions=("Region", "Product_Category", "Product_Name", "Sales_Rep", "Deal_Stage", "Forecast_Category", "Lead_Source", "Account_Name", "Customer_Type"),
        default_chart_type="bar",
        business_description="Summed deal amount across the chosen grouping.",
    ),
    "pipeline": MetricDefinition(
        metric_name="pipeline",
        synonyms=("pipeline", "pipeline value", "opportunity value"),
        dataset_required=("deals",),
        required_columns=("Amount", "Forecast_Category"),
        default_aggregation="sum",
        valid_dimensions=("Sales_Rep", "Region", "Forecast_Category", "Product_Category", "Product_Name", "Account_Name"),
        default_chart_type="bar",
        business_description="Pipeline-tagged deal amount across the chosen grouping.",
    ),
    "deal_count": MetricDefinition(
        metric_name="deal_count",
        synonyms=("deal count", "deals", "number of deals"),
        dataset_required=("deals",),
        required_columns=("Deal_ID",),
        default_aggregation="count",
        valid_dimensions=("Deal_Stage", "Region", "Sales_Rep", "Forecast_Category", "Product_Category", "Product_Name", "Account_Name"),
        default_chart_type="bar",
        business_description="Count of deals across the chosen grouping.",
    ),
    "average_deal_size": MetricDefinition(
        metric_name="average_deal_size",
        synonyms=("average deal size", "avg deal size", "average amount"),
        dataset_required=("deals",),
        required_columns=("Amount", "Deal_ID"),
        default_aggregation="mean",
        valid_dimensions=("Region", "Product_Category", "Product_Name", "Sales_Rep", "Deal_Stage", "Account_Name", "Customer_Type"),
        default_chart_type="bar",
        business_description="Average deal amount within each selected group.",
    ),
    "sales_rep_performance": MetricDefinition(
        metric_name="sales_rep_performance",
        synonyms=("sales rep performance", "sales rep", "salesperson", "salesman", "owner"),
        dataset_required=("deals", "leads"),
        required_columns=("Amount",),
        default_aggregation="sum",
        valid_dimensions=("Sales_Rep",),
        default_chart_type="bar",
        business_description="Deal performance grouped by sales rep.",
    ),
    "forecast_quality": MetricDefinition(
        metric_name="forecast_quality",
        synonyms=("forecast quality", "weak forecast", "at-risk forecast", "weak pipeline"),
        dataset_required=("deals",),
        required_columns=("Product_Category", "Forecast_Category", "Amount"),
        default_aggregation="ratio",
        valid_dimensions=("Product_Category",),
        default_chart_type="bar",
        business_description="Weak forecast ratio by product category.",
    ),
    "lead_source_conversion": MetricDefinition(
        metric_name="lead_source_conversion",
        synonyms=("lead conversion", "source conversion", "low conversion"),
        dataset_required=("leads", "deals"),
        required_columns=("Lead_ID", "Lead_Source", "Estimated_Value"),
        default_aggregation="comparison",
        valid_dimensions=("Lead_Source",),
        default_chart_type="bar",
        business_description="Lead source value compared against conversion into deals.",
    ),
    "monthly_pipeline_trend": MetricDefinition(
        metric_name="monthly_pipeline_trend",
        synonyms=("monthly pipeline trend", "pipeline by month", "revenue trend by month", "revenue trend by closing date"),
        dataset_required=("deals",),
        required_columns=("Closing_Date", "Amount"),
        default_aggregation="sum",
        valid_dimensions=("Closing_Date",),
        default_chart_type="line",
        business_description="Monthly amount trend grouped by closing month.",
    ),
    "win_rate": MetricDefinition(
        metric_name="win_rate",
        synonyms=("win rate", "conversion to closed won"),
        dataset_required=("deals",),
        required_columns=("Deal_Stage",),
        default_aggregation="ratio",
        valid_dimensions=("Lead_Source", "Sales_Rep", "Region"),
        default_chart_type="bar",
        business_description="Closed Won share out of Closed Won plus Closed Lost.",
    ),
    "margin": MetricDefinition(
        metric_name="margin",
        synonyms=("margin", "profit margin"),
        dataset_required=("deals",),
        required_columns=("Margin",),
        default_aggregation="sum",
        valid_dimensions=("Product_Category", "Region", "Sales_Rep"),
        default_chart_type="bar",
        business_description="Margin summed across the chosen grouping.",
    ),
}


DIMENSION_SYNONYMS: dict[str, tuple[str, ...]] = {
    "Region": ("region", "regional", "area", "market", "location"),
    "Product_Category": ("product category", "product categories", "product", "item", "category"),
    "Product_Name": ("product name", "item name", "sku"),
    "Sales_Rep": ("sales rep", "salesperson", "salesman", "owner", "rep"),
    "Deal_Stage": ("deal stage", "stage", "status"),
    "Lead_Source": ("lead source", "source"),
    "Forecast_Category": ("forecast category", "forecast", "forecast bucket"),
    "Closing_Date": ("closing date", "close date", "month"),
    "Account_Name": ("account name", "customer name", "account", "customer"),
    "Customer_Type": ("customer type", "customer segment", "segment"),
}


METRIC_SYNONYM_TO_NAME: dict[str, str] = {}
for metric_name, definition in METRIC_DEFINITIONS.items():
    for synonym in definition.synonyms:
        METRIC_SYNONYM_TO_NAME[synonym] = metric_name


DIMENSION_SYNONYM_TO_NAME: dict[str, str] = {}
for dimension_name, synonyms in DIMENSION_SYNONYMS.items():
    for synonym in synonyms:
        DIMENSION_SYNONYM_TO_NAME[synonym] = dimension_name
