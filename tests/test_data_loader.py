"""Unit tests for data_loader."""

import pandas as pd
import pytest

from src.data_loader import REQUIRED_LEADS_COLUMNS, validate_columns


def test_validate_columns_passes_with_all_required_columns():
    df = pd.DataFrame(columns=REQUIRED_LEADS_COLUMNS)
    validate_columns(df, REQUIRED_LEADS_COLUMNS, "leads.csv")


def test_validate_columns_raises_when_columns_missing():
    df = pd.DataFrame(columns=["Lead_ID", "Company_Name"])
    with pytest.raises(ValueError, match="missing required columns"):
        validate_columns(df, REQUIRED_LEADS_COLUMNS, "leads.csv")
