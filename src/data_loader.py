"""Load and validate synthetic and Zoho CRM CSV data with Pandas."""

from pathlib import Path
from typing import BinaryIO

import pandas as pd

from src.config import DEALS_CSV, LEADS_CSV
from src.zoho_adapter import load_uploaded_zoho_modules_with_diagnostics

REQUIRED_LEADS_COLUMNS = [
    "Lead_ID",
    "Company_Name",
    "Lead_Source",
    "Industry",
    "Estimated_Value",
    "Status",
    "Created_Date",
    "Sales_Rep",
]

REQUIRED_DEALS_COLUMNS = [
    "Deal_ID",
    "Lead_ID",
    "Company_Name",
    "Deal_Stage",
    "Amount",
    "Closing_Date",
    "Forecast_Category",
]


def validate_columns(df: pd.DataFrame, required_columns: list[str], file_name: str) -> None:
    """Raise ValueError if any required column is missing."""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"{file_name} is missing required columns: {', '.join(missing)}"
        )


def _clean_leads_df(df: pd.DataFrame, file_name: str) -> pd.DataFrame:
    """Validate and clean a leads dataframe."""
    validate_columns(df, REQUIRED_LEADS_COLUMNS, file_name)

    cleaned_df = df.copy()
    cleaned_df["Created_Date"] = pd.to_datetime(cleaned_df["Created_Date"])
    cleaned_df["Estimated_Value"] = pd.to_numeric(cleaned_df["Estimated_Value"])

    return cleaned_df


def _clean_deals_df(df: pd.DataFrame, file_name: str) -> pd.DataFrame:
    """Validate and clean a deals dataframe."""
    validate_columns(df, REQUIRED_DEALS_COLUMNS, file_name)

    cleaned_df = df.copy()
    cleaned_df["Closing_Date"] = pd.to_datetime(cleaned_df["Closing_Date"])
    cleaned_df["Amount"] = pd.to_numeric(cleaned_df["Amount"])

    return cleaned_df


def load_leads(path: Path = LEADS_CSV) -> pd.DataFrame:
    """Load leads CSV, validate columns, and apply type conversions."""
    if not path.exists():
        raise FileNotFoundError(f"Leads file not found: {path}")

    df = pd.read_csv(path)
    return _clean_leads_df(df, path.name)


def load_deals(path: Path = DEALS_CSV) -> pd.DataFrame:
    """Load deals CSV, validate columns, and apply type conversions."""
    if not path.exists():
        raise FileNotFoundError(f"Deals file not found: {path}")

    df = pd.read_csv(path)
    return _clean_deals_df(df, path.name)


def load_leads_from_file(uploaded_file: BinaryIO) -> pd.DataFrame:
    """Load and clean leads from an uploaded CSV file."""
    file_name = getattr(uploaded_file, "name", "uploaded_leads.csv")
    df = pd.read_csv(uploaded_file)
    return _clean_leads_df(df, file_name)


def load_deals_from_file(uploaded_file: BinaryIO) -> pd.DataFrame:
    """Load and clean deals from an uploaded CSV file."""
    file_name = getattr(uploaded_file, "name", "uploaded_deals.csv")
    df = pd.read_csv(uploaded_file)
    return _clean_deals_df(df, file_name)


def load_all_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and return leads and deals DataFrames."""
    leads_df = load_leads()
    deals_df = load_deals()
    return leads_df, deals_df


def empty_leads_df() -> pd.DataFrame:
    """Return an empty canonical leads dataframe."""
    return pd.DataFrame(columns=REQUIRED_LEADS_COLUMNS)


def empty_deals_df() -> pd.DataFrame:
    """Return an empty canonical deals dataframe."""
    return pd.DataFrame(columns=REQUIRED_DEALS_COLUMNS)


def load_zoho_modules_from_files(
    uploaded_files: list[BinaryIO] | None,
) -> tuple[dict[str, pd.DataFrame], dict[str, list[str]], list[str], dict[str, object]]:
    """Load arbitrary Zoho CRM module exports from uploaded CSV files."""
    return load_uploaded_zoho_modules_with_diagnostics(uploaded_files)
