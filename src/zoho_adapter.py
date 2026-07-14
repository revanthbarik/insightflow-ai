"""Zoho CRM CSV detection and column canonicalization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import BinaryIO

import pandas as pd

from src.module_registry import MODULE_REGISTRY, ModuleDefinition, SUPPORTED_MODULE_NAMES

CANONICAL_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "Lead_ID": ("lead id", "lead_id", "leadid"),
    "Deal_ID": ("deal id", "deal_id", "potential id", "opportunity id"),
    "Account_ID": ("account id", "account_id"),
    "Contact_ID": ("contact id", "contact_id"),
    "Product_ID": ("product id", "product_id", "item id"),
    "Quote_ID": ("quote id", "quote_id"),
    "Invoice_ID": ("invoice id", "invoice_id"),
    "Sales_Order_ID": ("sales order id", "sales_order_id"),
    "Purchase_Order_ID": ("purchase order id", "purchase_order_id"),
    "Vendor_ID": ("vendor id", "vendor_id"),
    "Campaign_ID": ("campaign id", "campaign_id"),
    "Activity_ID": ("activity id", "activity_id"),
    "Task_ID": ("task id", "task_id"),
    "Call_ID": ("call id", "call_id"),
    "Meeting_ID": ("meeting id", "meeting_id", "event id"),
    "Sales_Rep": ("deal owner", "lead owner", "account owner", "owner", "assigned to", "sales rep", "salesperson", "salesman"),
    "Amount": ("amount", "deal amount", "revenue", "value", "total revenue"),
    "Closing_Date": ("closing date", "close date"),
    "Product_Category": ("product", "product category", "item category"),
    "Forecast_Category": ("forecast category", "forecast"),
    "Deal_Stage": ("stage", "deal stage", "status"),
    "Lead_Source": ("lead source", "source"),
    "Estimated_Value": ("estimated value", "estimated_value", "lead value", "estimated revenue"),
    "Created_Date": ("created date", "creation date", "lead created time"),
    "Region": ("region", "market", "geography", "area", "country", "emirate"),
    "Customer_Type": ("customer type", "customer segment", "segment"),
    "Expected_Margin": ("expected margin",),
    "Margin": ("margin",),
    "Company_Name": ("company name", "company", "account name"),
    "Product_Name": ("product name", "item name"),
    "Account_Name": ("account name", "customer name"),
    "Contact_Name": ("contact name", "full name"),
    "Vendor_Name": ("vendor name", "supplier name"),
    "Campaign_Name": ("campaign name",),
    "Status": ("status",),
}


@dataclass
class ModuleLoadResult:
    """Normalized result for one uploaded module."""

    module_name: str
    dataframe: pd.DataFrame
    unknown_columns: list[str]
    warnings: list[str]
    original_columns: list[str]
    mapped_columns: dict[str, str]
    missing_required_columns: list[str]


@dataclass
class ModuleDiagnostics:
    """User-facing diagnostics for one normalized uploaded module."""

    module_name: str
    file_name: str
    row_count: int
    original_columns: list[str]
    mapped_columns: dict[str, str]
    normalized_columns: list[str]
    missing_required_columns: list[str]
    sample_rows: list[dict[str, object]]


def _normalize_label(label: str) -> str:
    return " ".join(
        str(label)
        .replace("_", " ")
        .replace("-", " ")
        .strip()
        .lower()
        .split()
    )


def build_alias_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    for canonical, aliases in CANONICAL_COLUMN_ALIASES.items():
        lookup[_normalize_label(canonical)] = canonical
        for alias in aliases:
            lookup[_normalize_label(alias)] = canonical
    return lookup


ALIAS_LOOKUP = build_alias_lookup()

NUMERIC_COLUMNS = ("Amount", "Estimated_Value", "Expected_Margin", "Margin", "Budget")
DATE_COLUMNS = ("Closing_Date", "Created_Date", "Due_Date", "Start_Time")


def map_zoho_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str], list[str]]:
    """Map common Zoho column names to canonical InsightFlow names."""
    mapped_names: dict[str, str] = {}
    unknown_columns: list[str] = []
    for column in df.columns:
        normalized = _normalize_label(column)
        mapped_names[column] = ALIAS_LOOKUP.get(normalized, column)
        if mapped_names[column] == column and normalized not in ALIAS_LOOKUP:
            unknown_columns.append(column)
    mapped_df = df.rename(columns=mapped_names)
    return mapped_df, mapped_names, unknown_columns


def coerce_canonical_types(df: pd.DataFrame) -> pd.DataFrame:
    """Apply lightweight numeric/date coercion for canonical InsightFlow fields."""
    coerced_df = df.copy()
    for column in NUMERIC_COLUMNS:
        if column in coerced_df.columns:
            coerced_df[column] = pd.to_numeric(coerced_df[column], errors="coerce")
    for column in DATE_COLUMNS:
        if column in coerced_df.columns:
            coerced_df[column] = pd.to_datetime(coerced_df[column], errors="coerce")
    return coerced_df


def detect_module_name(df: pd.DataFrame, file_name: str | None = None) -> str | None:
    """Detect the most likely Zoho module from filename and canonicalized columns."""
    normalized_name = _normalize_label(file_name or "")
    for module_name, definition in MODULE_REGISTRY.items():
        if any(alias in normalized_name for alias in definition.aliases):
            return module_name

    best_match: tuple[str, int] | None = None
    columns = set(df.columns)
    for module_name, definition in MODULE_REGISTRY.items():
        required_matches = sum(1 for column in definition.required_columns if column in columns)
        optional_matches = sum(1 for column in definition.optional_columns if column in columns)
        score = (required_matches * 10) + optional_matches
        if required_matches > 0 and (best_match is None or score > best_match[1]):
            best_match = (module_name, score)
    return best_match[0] if best_match else None


def validate_module_dataframe(df: pd.DataFrame, definition: ModuleDefinition) -> list[str]:
    """Return validation warnings instead of throwing for partial/empty uploads."""
    warnings: list[str] = []
    if df.empty:
        warnings.append(f"{definition.module_name} is empty.")
    missing_required = sorted(set(definition.required_columns).difference(df.columns))
    if missing_required:
        warnings.append(
            f"{definition.module_name} is missing required columns: {', '.join(missing_required)}."
        )
    if definition.id_column in df.columns and df[definition.id_column].duplicated().any():
        warnings.append(f"{definition.module_name} contains duplicate IDs in {definition.id_column}.")
    return warnings


def get_missing_required_columns(df: pd.DataFrame, definition: ModuleDefinition) -> list[str]:
    """Return missing required canonical columns for a detected module."""
    return sorted(set(definition.required_columns).difference(df.columns))


def build_module_diagnostics(result: ModuleLoadResult, file_name: str) -> ModuleDiagnostics:
    """Build sidebar-friendly diagnostics for one normalized module."""
    return ModuleDiagnostics(
        module_name=result.module_name,
        file_name=file_name,
        row_count=len(result.dataframe),
        original_columns=result.original_columns,
        mapped_columns={
            original: mapped
            for original, mapped in result.mapped_columns.items()
            if original != mapped
        },
        normalized_columns=result.dataframe.columns.tolist(),
        missing_required_columns=result.missing_required_columns,
        sample_rows=result.dataframe.head(3).fillna("").to_dict(orient="records"),
    )


def load_zoho_module_from_file(uploaded_file: BinaryIO) -> ModuleLoadResult:
    """Read one CSV export, canonicalize columns, and detect its Zoho module."""
    file_name = getattr(uploaded_file, "name", "uploaded_module.csv")
    df = pd.read_csv(uploaded_file)
    original_columns = df.columns.tolist()
    mapped_df, mapped_columns, unknown_columns = map_zoho_columns(df)
    mapped_df = coerce_canonical_types(mapped_df)
    module_name = detect_module_name(mapped_df, file_name=file_name) or "Unknown"
    warnings = []
    missing_required_columns: list[str] = []
    if module_name != "Unknown":
        definition = MODULE_REGISTRY[module_name]
        missing_required_columns = get_missing_required_columns(mapped_df, definition)
        warnings.extend(validate_module_dataframe(mapped_df, definition))
    else:
        warnings.append(
            f"Could not confidently detect a supported Zoho CRM module for {file_name}."
        )
    return ModuleLoadResult(
        module_name=module_name,
        dataframe=mapped_df,
        unknown_columns=unknown_columns,
        warnings=warnings,
        original_columns=original_columns,
        mapped_columns=mapped_columns,
        missing_required_columns=missing_required_columns,
    )


def load_uploaded_zoho_modules(uploaded_files: list[BinaryIO] | None) -> tuple[dict[str, pd.DataFrame], dict[str, list[str]], list[str]]:
    """Load multiple Zoho CSV exports into a module-name to DataFrame mapping."""
    modules: dict[str, pd.DataFrame] = {}
    warnings: dict[str, list[str]] = {}
    unknown_files: list[str] = []
    for uploaded_file in uploaded_files or []:
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
        result = load_zoho_module_from_file(uploaded_file)
        file_name = getattr(uploaded_file, "name", "uploaded_module.csv")
        if result.module_name == "Unknown":
            unknown_files.append(file_name)
            warnings[file_name] = result.warnings
            continue
        existing_warnings = warnings.setdefault(result.module_name, [])
        if result.module_name in modules:
            existing_warnings.append(
                f"Multiple uploads matched {result.module_name}. Using the latest file: {file_name}."
            )
        modules[result.module_name] = result.dataframe
        module_warnings = list(result.warnings)
        if result.unknown_columns:
            module_warnings.append(
                f"Unmapped columns kept as-is: {', '.join(result.unknown_columns[:8])}."
            )
        warnings[result.module_name] = existing_warnings + module_warnings
    return modules, warnings, unknown_files


def load_uploaded_zoho_modules_with_diagnostics(
    uploaded_files: list[BinaryIO] | None,
) -> tuple[dict[str, pd.DataFrame], dict[str, list[str]], list[str], dict[str, ModuleDiagnostics]]:
    """Load multiple Zoho exports and include per-module diagnostics."""
    modules: dict[str, pd.DataFrame] = {}
    warnings: dict[str, list[str]] = {}
    unknown_files: list[str] = []
    diagnostics: dict[str, ModuleDiagnostics] = {}
    for uploaded_file in uploaded_files or []:
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
        result = load_zoho_module_from_file(uploaded_file)
        file_name = getattr(uploaded_file, "name", "uploaded_module.csv")
        if result.module_name == "Unknown":
            unknown_files.append(file_name)
            warnings[file_name] = result.warnings
            continue
        existing_warnings = warnings.setdefault(result.module_name, [])
        if result.module_name in modules:
            existing_warnings.append(
                f"Multiple uploads matched {result.module_name}. Using the latest file: {file_name}."
            )
        modules[result.module_name] = result.dataframe
        diagnostics[result.module_name] = build_module_diagnostics(result, file_name)
        module_warnings = list(result.warnings)
        if result.unknown_columns:
            module_warnings.append(
                f"Unmapped columns kept as-is: {', '.join(result.unknown_columns[:8])}."
            )
        warnings[result.module_name] = existing_warnings + module_warnings
    return modules, warnings, unknown_files, diagnostics


def missing_supported_modules(loaded_modules: dict[str, pd.DataFrame]) -> list[str]:
    """Return supported modules not currently loaded."""
    return [module_name for module_name in SUPPORTED_MODULE_NAMES if module_name not in loaded_modules]
