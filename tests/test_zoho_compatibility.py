"""Tests for the Zoho CRM compatibility layer."""

from __future__ import annotations

from io import StringIO

import pandas as pd

from src.capability_detector import detect_available_analytics
from src.analytics import get_region_pipeline, get_sales_rep_performance
from src.query_engine import answer_business_question
from src.relationship_builder import build_relationships, validate_join_key
from src.zoho_adapter import (
    detect_module_name,
    load_uploaded_zoho_modules,
    load_uploaded_zoho_modules_with_diagnostics,
    map_zoho_columns,
)


class NamedStringIO(StringIO):
    """StringIO with a .name attribute for upload-style tests."""

    def __init__(self, value: str, name: str) -> None:
        super().__init__(value)
        self.name = name


def _deals_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Deal_ID": ["D001", "D002", "D003"],
            "Lead_ID": ["L001", "L002", "L003"],
            "Account_ID": ["A001", "A002", "A001"],
            "Product_ID": ["P001", "P002", "P001"],
            "Amount": [100000.0, 200000.0, 50000.0],
            "Forecast_Category": ["Pipeline", "Commit", "Closed"],
            "Region": ["UAE", "Saudi Arabia", "UAE"],
        }
    )


def _leads_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Lead_ID": ["L001", "L002", "L003"],
            "Lead_Source": ["Website", "Partner", "Referral"],
            "Estimated_Value": [150000.0, 220000.0, 80000.0],
            "Customer_Type": ["Installer", "Distributor", "Installer"],
        }
    )


def _accounts_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Account_ID": ["A001", "A002"],
            "Account_Name": ["Alpha Energy", "Beta Grid"],
            "Customer_Type": ["Installer", "Distributor"],
            "Region": ["UAE", "Saudi Arabia"],
        }
    )


def test_column_mapping_aliases_map_common_zoho_fields():
    raw_df = pd.DataFrame(
        {
            "Deal Owner": ["Aisha"],
            "Deal Amount": ["100000"],
            "Close Date": ["2025-01-01"],
            "Product": ["Batteries"],
        }
    )
    mapped_df, mapping, unknown_columns = map_zoho_columns(raw_df)

    assert mapping["Deal Owner"] == "Sales_Rep"
    assert mapping["Deal Amount"] == "Amount"
    assert mapping["Close Date"] == "Closing_Date"
    assert mapping["Product"] == "Product_Category"
    assert unknown_columns == []
    assert {"Sales_Rep", "Amount", "Closing_Date", "Product_Category"} == set(mapped_df.columns)


def test_uploaded_leads_with_lead_owner_map_to_sales_rep():
    raw_df = pd.DataFrame(
        {
            "Lead Owner": ["Sara"],
            "Lead ID": ["L001"],
            "Estimated Value": ["1000"],
        }
    )
    mapped_df, mapping, _ = map_zoho_columns(raw_df)
    assert mapping["Lead Owner"] == "Sales_Rep"
    assert "Sales_Rep" in mapped_df.columns


def test_module_detection_prefers_filename_and_columns():
    deals_df = pd.DataFrame({"Deal_ID": ["D001"], "Lead_ID": ["L001"], "Amount": [1000]})
    accounts_df = pd.DataFrame({"Account_ID": ["A001"], "Account_Name": ["Alpha Energy"]})

    assert detect_module_name(deals_df, file_name="Zoho_Deals_Export.csv") == "Deals"
    assert detect_module_name(accounts_df, file_name="customer_accounts.csv") == "Accounts"


def test_load_uploaded_zoho_modules_handles_unknown_csv_safely():
    unknown_upload = NamedStringIO("Foo,Bar\n1,2\n", "mystery.csv")
    modules, warnings, unknown_files = load_uploaded_zoho_modules([unknown_upload])

    assert modules == {}
    assert "mystery.csv" in unknown_files
    assert "mystery.csv" in warnings


def test_relationship_detection_and_join_validation_work():
    loaded_modules = {
        "Deals": _deals_df(),
        "Leads": _leads_df(),
        "Accounts": _accounts_df(),
    }
    relationships = build_relationships(loaded_modules)
    relationship_map = {
        (relationship.left_module, relationship.right_module): relationship
        for relationship in relationships
    }

    assert relationship_map[("Deals", "Leads")].available is True
    assert relationship_map[("Deals", "Accounts")].available is True

    invalid, reason = validate_join_key(
        _deals_df(),
        pd.DataFrame({"Lead_ID": ["L001", "L001"]}),
        "Lead_ID",
    )
    assert invalid is False
    assert "duplicates" in reason


def test_available_analytics_detector_handles_partial_uploads():
    capabilities = detect_available_analytics({"Deals": _deals_df()})
    capability_map = {capability.name: capability for capability in capabilities}

    assert capability_map["Revenue"].available is True
    assert capability_map["Pipeline"].available is True
    assert capability_map["Lead Conversion"].available is False
    assert capability_map["Lead Conversion"].reason == "Leads module missing."


def test_existing_leads_and_deals_workflow_still_loads():
    deals_upload = NamedStringIO(
        "Deal ID,Lead ID,Amount,Forecast Category\nD001,L001,100000,Pipeline\n",
        "Deals.csv",
    )
    leads_upload = NamedStringIO(
        "Lead ID,Lead Source,Estimated Value\nL001,Website,200000\n",
        "Leads.csv",
    )

    modules, warnings, unknown_files = load_uploaded_zoho_modules([deals_upload, leads_upload])

    assert set(modules) == {"Deals", "Leads"}
    assert unknown_files == []
    assert "Amount" in modules["Deals"].columns
    assert "Estimated_Value" in modules["Leads"].columns
    assert isinstance(warnings["Deals"], list)


def test_sales_rep_performance_works_with_deal_owner_column():
    deals_upload = NamedStringIO(
        "Deal ID,Lead ID,Deal Owner,Deal Amount,Forecast Category\nD001,L001,Aisha,100000,Pipeline\nD002,L002,Omar,250000,Commit\n",
        "Deals.csv",
    )
    modules, _, _, diagnostics = load_uploaded_zoho_modules_with_diagnostics([deals_upload])
    performance = get_sales_rep_performance(pd.DataFrame(), modules["Deals"])

    assert not performance.empty
    assert performance["Sales_Rep"].tolist() == ["Omar", "Aisha"]
    assert performance["Total_Deal_Amount"].tolist() == [250000.0, 100000.0]
    assert diagnostics["Deals"].row_count == 2
    assert diagnostics["Deals"].mapped_columns["Deal Owner"] == "Sales_Rep"


def test_sales_rep_performance_empty_reason_is_clear_when_rep_missing():
    performance = get_sales_rep_performance(
        pd.DataFrame(),
        pd.DataFrame({"Deal_ID": ["D001"], "Amount": [1000.0]}),
    )
    assert performance.empty
    assert "Sales_Rep is missing or empty after column mapping" in performance.attrs["empty_reason"]


def test_pipeline_by_region_includes_all_open_pipeline_regions_and_excludes_closed():
    deals_df = pd.DataFrame(
        {
            "Deal_ID": ["D001", "D002", "D003", "D004", "D005"],
            "Lead_ID": ["L001", "L002", "L003", "L004", "L005"],
            "Amount": [100000.0, 200000.0, 300000.0, 400000.0, 500000.0],
            "Forecast_Category": ["Pipeline", "Commit", "Closed", "At Risk", "Best Case"],
            "Deal_Stage": ["Negotiation", "PO Received", "Closed Won", "Technical Discussion", "Closed Lost"],
            "Region": ["Dubai", "Abu Dhabi", "Dubai", "Saudi Arabia", "Oman"],
        }
    )
    summary = get_region_pipeline(pd.DataFrame(), deals_df)

    assert summary["Region"].tolist() == ["Saudi Arabia", "Abu Dhabi", "Dubai"]
    assert summary["Total_Amount"].tolist() == [400000.0, 200000.0, 100000.0]
    assert "Oman" not in summary["Region"].tolist()


def test_chart_empty_data_returns_clear_reason():
    summary = get_region_pipeline(
        pd.DataFrame(),
        pd.DataFrame({"Deal_ID": ["D001"], "Amount": [100000.0], "Deal_Stage": ["Closed Won"]}),
    )
    assert summary.empty
    assert "No open pipeline deals were found" in summary.attrs["empty_reason"]


def test_diagnostics_include_row_count_and_mapped_columns():
    deals_upload = NamedStringIO(
        "Deal ID,Lead ID,Deal Owner,Revenue,Close Date\nD001,L001,Aisha,100000,2025-01-01\n",
        "Zoho Deals.csv",
    )
    modules, warnings, unknown_files, diagnostics = load_uploaded_zoho_modules_with_diagnostics([deals_upload])

    assert set(modules) == {"Deals"}
    assert unknown_files == []
    assert isinstance(warnings["Deals"], list)
    assert diagnostics["Deals"].row_count == 1
    assert diagnostics["Deals"].mapped_columns["Deal Owner"] == "Sales_Rep"
    assert diagnostics["Deals"].mapped_columns["Revenue"] == "Amount"
    assert "Sales_Rep" in diagnostics["Deals"].normalized_columns


def test_semantic_bi_engine_can_use_accounts_module_for_customer_revenue():
    result = answer_business_question(
        "Show revenue by customer",
        leads_df=_leads_df(),
        deals_df=_deals_df(),
        extra_tables={"Accounts": _accounts_df()},
    )

    assert result["answer_type"] == "chart"
    assert result["x_axis"] == "Account_Name"
    assert "Alpha Energy" in result["data"]["Account_Name"].tolist()
    assert "Total_Amount" in result["data"].columns


def test_no_regression_for_existing_region_revenue_query_with_extra_modules():
    result = answer_business_question(
        "Show revenue by region",
        leads_df=_leads_df(),
        deals_df=_deals_df(),
        extra_tables={"Accounts": _accounts_df()},
    )

    assert result["answer_type"] == "chart"
    assert result["title"] == "Revenue by Region"
    assert not result["data"].empty
