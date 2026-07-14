"""Detect which analytics are available from the loaded Zoho CRM modules."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.relationship_builder import relationship_lookup


@dataclass
class CapabilityStatus:
    """One user-facing analytic capability."""

    name: str
    available: bool
    reason: str


def detect_available_analytics(loaded_modules: dict[str, pd.DataFrame]) -> list[CapabilityStatus]:
    """Return supported and unavailable analytics for the current module set."""
    relationships = relationship_lookup(loaded_modules)
    deals_df = loaded_modules.get("Deals", pd.DataFrame())
    leads_df = loaded_modules.get("Leads", pd.DataFrame())
    accounts_df = loaded_modules.get("Accounts", pd.DataFrame())
    products_df = loaded_modules.get("Products", pd.DataFrame())

    capabilities: list[CapabilityStatus] = []

    capabilities.append(
        CapabilityStatus(
            "Revenue",
            "Deals" in loaded_modules and "Amount" in deals_df.columns,
            "Deals module with Amount is required." if "Deals" not in loaded_modules or "Amount" not in deals_df.columns else "Available from Deals.",
        )
    )
    capabilities.append(
        CapabilityStatus(
            "Pipeline",
            "Deals" in loaded_modules and {"Amount", "Forecast_Category"}.issubset(deals_df.columns),
            "Forecast_Category is required in Deals." if "Deals" in loaded_modules else "Deals module missing.",
        )
    )
    capabilities.append(
        CapabilityStatus(
            "Forecast",
            "Deals" in loaded_modules and "Forecast_Category" in deals_df.columns,
            "Forecast_Category missing from Deals." if "Deals" in loaded_modules else "Deals module missing.",
        )
    )
    capabilities.append(
        CapabilityStatus(
            "Customer Analysis",
            (
                ("Customer_Type" in deals_df.columns)
                or (
                    "Deals" in loaded_modules
                    and "Leads" in loaded_modules
                    and relationships.get(("Deals", "Leads"))
                    and relationships[("Deals", "Leads")].available
                    and "Customer_Type" in leads_df.columns
                )
                or (
                    "Deals" in loaded_modules
                    and "Accounts" in loaded_modules
                    and relationships.get(("Deals", "Accounts"))
                    and relationships[("Deals", "Accounts")].available
                    and "Account_Name" in accounts_df.columns
                )
            ),
            "Needs Customer_Type in Leads or an Accounts join." if "Deals" in loaded_modules else "Deals module missing.",
        )
    )
    capabilities.append(
        CapabilityStatus(
            "Product Analysis",
            (
                "Product_Category" in deals_df.columns
                or (
                    "Deals" in loaded_modules
                    and "Products" in loaded_modules
                    and relationships.get(("Deals", "Products"))
                    and relationships[("Deals", "Products")].available
                )
            ),
            "Needs Product_Category in Deals or a valid Products join." if "Deals" in loaded_modules else "Deals module missing.",
        )
    )
    capabilities.append(
        CapabilityStatus(
            "Regional Analysis",
            (
                "Region" in deals_df.columns
                or (
                    "Leads" in loaded_modules
                    and "Region" in leads_df.columns
                    and relationships.get(("Deals", "Leads"))
                    and relationships[("Deals", "Leads")].available
                )
                or ("Accounts" in loaded_modules and "Region" in accounts_df.columns)
            ),
            "Needs Region in Deals, Leads, or Accounts." if "Deals" in loaded_modules or "Leads" in loaded_modules or "Accounts" in loaded_modules else "No regional modules loaded.",
        )
    )
    capabilities.append(
        CapabilityStatus(
            "Lead Conversion",
            (
                "Leads" in loaded_modules
                and "Deals" in loaded_modules
                and relationships.get(("Deals", "Leads"))
                and relationships[("Deals", "Leads")].available
            ),
            "Leads module missing." if "Leads" not in loaded_modules else "Lead_ID relationship between Deals and Leads is required.",
        )
    )

    return capabilities
