"""Relationship inference for loaded Zoho CRM modules."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.module_registry import MODULE_REGISTRY


@dataclass
class RelationshipStatus:
    """Describe one module-to-module relationship candidate."""

    left_module: str
    right_module: str
    join_key: str
    available: bool
    reason: str


PREFERRED_RELATIONSHIPS = (
    ("Deals", "Leads", "Lead_ID"),
    ("Deals", "Accounts", "Account_ID"),
    ("Deals", "Products", "Product_ID"),
    ("Deals", "Contacts", "Contact_ID"),
    ("Quotes", "Deals", "Deal_ID"),
    ("Invoices", "Deals", "Deal_ID"),
    ("Sales Orders", "Deals", "Deal_ID"),
    ("Purchase Orders", "Vendors", "Vendor_ID"),
)


def validate_join_key(left_df: pd.DataFrame, right_df: pd.DataFrame, join_key: str) -> tuple[bool, str]:
    """Validate that a join key exists and can be used safely."""
    if join_key not in left_df.columns or join_key not in right_df.columns:
        return False, f"Missing join key `{join_key}` in one or both modules."
    if left_df.empty or right_df.empty:
        return False, f"Join key `{join_key}` exists but one module is empty."
    if right_df[join_key].dropna().duplicated().any():
        return False, f"Join key `{join_key}` has duplicates on the lookup side."
    return True, f"Join key `{join_key}` is available."


def build_relationships(loaded_modules: dict[str, pd.DataFrame]) -> list[RelationshipStatus]:
    """Infer known relationships from the currently loaded Zoho modules."""
    relationships: list[RelationshipStatus] = []
    for left_module, right_module, join_key in PREFERRED_RELATIONSHIPS:
        if left_module not in loaded_modules or right_module not in loaded_modules:
            reason = "One or both modules are not loaded."
            relationships.append(
                RelationshipStatus(left_module, right_module, join_key, False, reason)
            )
            continue
        available, reason = validate_join_key(
            loaded_modules[left_module],
            loaded_modules[right_module],
            join_key,
        )
        relationships.append(
            RelationshipStatus(left_module, right_module, join_key, available, reason)
        )
    return relationships


def relationship_lookup(loaded_modules: dict[str, pd.DataFrame]) -> dict[tuple[str, str], RelationshipStatus]:
    """Build a quick lookup for inferred module relationships."""
    return {
        (relationship.left_module, relationship.right_module): relationship
        for relationship in build_relationships(loaded_modules)
    }
