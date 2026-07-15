"""Tests for the data readiness report built from existing module metadata."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from src.data_health import build_data_health_report
from src.relationship_builder import build_relationships
from src.zoho_adapter import ModuleDiagnostics


def _healthy_modules() -> dict[str, pd.DataFrame]:
    return {
        "Leads": pd.DataFrame(
            {
                "Lead_ID": ["L001", "L002"],
                "Created_Date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
                "Region": ["Dubai", "Abu Dhabi"],
            }
        ),
        "Deals": pd.DataFrame(
            {
                "Deal_ID": ["D001", "D002"],
                "Lead_ID": ["L001", "L002"],
                "Amount": [100_000, 200_000],
                "Closing_Date": pd.to_datetime(["2025-02-01", "2025-02-02"]),
                "Forecast_Category": ["Pipeline", "Closed"],
            }
        ),
    }


def _capabilities() -> list[SimpleNamespace]:
    return [
        SimpleNamespace(name="Revenue", available=True, reason="Available from Deals."),
        SimpleNamespace(name="Lead Conversion", available=True, reason="Valid Lead_ID relationship."),
    ]


def test_readiness_report_uses_registry_requirements_and_join_coverage():
    modules = _healthy_modules()
    report = build_data_health_report(
        modules,
        capabilities=_capabilities(),
        relationships=build_relationships(modules),
    )

    deals_health = next(module for module in report.module_health if module.module_name == "Deals")
    lead_join = next(
        join
        for join in report.join_health
        if (join.left_module, join.right_module) == ("Deals", "Leads")
    )

    assert deals_health.status == "Healthy"
    assert deals_health.missing_required_columns == ()
    assert lead_join.status == "Healthy"
    assert lead_join.rows_checked == 2
    assert lead_join.matched_rows == 2
    assert report.enabled_capability_count == 2
    join_frame = report.join_frame()
    assert str(join_frame["Rows checked"].dtype) == "Int64"
    assert str(join_frame["Matched"].dtype) == "Int64"


def test_readiness_report_surfaces_missing_columns_and_unmatched_joins():
    modules = _healthy_modules()
    modules["Deals"] = modules["Deals"].drop(columns=["Amount"])
    modules["Deals"].loc[1, "Lead_ID"] = "L999"

    report = build_data_health_report(
        modules,
        capabilities=_capabilities(),
        relationships=build_relationships(modules),
    )

    deals_health = next(module for module in report.module_health if module.module_name == "Deals")
    lead_join = next(
        join
        for join in report.join_health
        if (join.left_module, join.right_module) == ("Deals", "Leads")
    )

    assert deals_health.status == "Warning"
    assert deals_health.missing_required_columns == ("Amount",)
    assert lead_join.status == "Partial"
    assert lead_join.unmatched_rows == 1


def test_readiness_report_exposes_existing_alias_mappings_only_when_supplied():
    modules = _healthy_modules()
    diagnostics = {
        "Deals": ModuleDiagnostics(
            module_name="Deals",
            file_name="deals.csv",
            row_count=2,
            original_columns=["Deal ID", "Lead ID", "Deal Amount"],
            mapped_columns={"Deal ID": "Deal_ID", "Deal Amount": "Amount"},
            normalized_columns=modules["Deals"].columns.tolist(),
            missing_required_columns=[],
            sample_rows=[],
        )
    }
    report = build_data_health_report(
        modules,
        capabilities=_capabilities(),
        relationships=build_relationships(modules),
        module_diagnostics=diagnostics,
    )

    assert report.resolved_mappings["Deals"]["Deal Amount"] == "Amount"
