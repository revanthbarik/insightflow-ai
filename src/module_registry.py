"""Module registry for supported Zoho CRM CSV exports."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModuleDefinition:
    """Describe one supported Zoho CRM module."""

    module_name: str
    aliases: tuple[str, ...]
    required_columns: tuple[str, ...]
    optional_columns: tuple[str, ...]
    id_column: str
    relationship_keys: tuple[str, ...]
    supported_metrics: tuple[str, ...]
    supported_dimensions: tuple[str, ...]


MODULE_REGISTRY: dict[str, ModuleDefinition] = {
    "Leads": ModuleDefinition(
        module_name="Leads",
        aliases=("lead", "leads", "lead export"),
        required_columns=("Lead_ID",),
        optional_columns=(
            "Company_Name",
            "Lead_Source",
            "Industry",
            "Estimated_Value",
            "Status",
            "Created_Date",
            "Sales_Rep",
            "Region",
            "Customer_Type",
            "Account_ID",
            "Contact_ID",
            "Product_ID",
        ),
        id_column="Lead_ID",
        relationship_keys=("Lead_ID", "Account_ID", "Contact_ID", "Product_ID"),
        supported_metrics=("lead_count", "estimated_lead_value", "lead_conversion"),
        supported_dimensions=("Lead_Source", "Sales_Rep", "Region", "Industry", "Customer_Type"),
    ),
    "Deals": ModuleDefinition(
        module_name="Deals",
        aliases=("deal", "deals", "potentials", "opportunities"),
        required_columns=("Deal_ID", "Lead_ID", "Amount"),
        optional_columns=(
            "Company_Name",
            "Deal_Stage",
            "Closing_Date",
            "Forecast_Category",
            "Product_Category",
            "Region",
            "Sales_Rep",
            "Expected_Margin",
            "Margin",
            "Account_ID",
            "Product_ID",
            "Contact_ID",
        ),
        id_column="Deal_ID",
        relationship_keys=("Lead_ID", "Account_ID", "Product_ID", "Contact_ID"),
        supported_metrics=("revenue", "pipeline", "forecast_quality", "average_deal_size", "deal_count", "win_rate", "margin"),
        supported_dimensions=("Region", "Sales_Rep", "Deal_Stage", "Forecast_Category", "Product_Category", "Account_ID", "Product_ID", "Contact_ID"),
    ),
    "Accounts": ModuleDefinition(
        module_name="Accounts",
        aliases=("account", "accounts", "customers"),
        required_columns=("Account_ID",),
        optional_columns=("Account_Name", "Region", "Industry", "Customer_Type", "Account_Owner"),
        id_column="Account_ID",
        relationship_keys=("Account_ID",),
        supported_metrics=("revenue", "account_count"),
        supported_dimensions=("Account_Name", "Region", "Industry", "Customer_Type", "Account_Owner"),
    ),
    "Contacts": ModuleDefinition(
        module_name="Contacts",
        aliases=("contact", "contacts"),
        required_columns=("Contact_ID",),
        optional_columns=("Contact_Name", "Account_ID", "Region", "Title"),
        id_column="Contact_ID",
        relationship_keys=("Contact_ID", "Account_ID"),
        supported_metrics=("contact_count",),
        supported_dimensions=("Contact_Name", "Region", "Title"),
    ),
    "Products": ModuleDefinition(
        module_name="Products",
        aliases=("product", "products", "items"),
        required_columns=("Product_ID",),
        optional_columns=("Product_Name", "Product_Category", "Vendor_ID"),
        id_column="Product_ID",
        relationship_keys=("Product_ID", "Vendor_ID"),
        supported_metrics=("revenue", "margin", "product_count"),
        supported_dimensions=("Product_Name", "Product_Category", "Vendor_ID"),
    ),
    "Activities": ModuleDefinition(
        module_name="Activities",
        aliases=("activity", "activities"),
        required_columns=("Activity_ID",),
        optional_columns=("Owner", "Status", "Due_Date", "Related_To_ID"),
        id_column="Activity_ID",
        relationship_keys=("Related_To_ID", "Lead_ID", "Deal_ID", "Account_ID"),
        supported_metrics=("activity_count",),
        supported_dimensions=("Owner", "Status"),
    ),
    "Tasks": ModuleDefinition(
        module_name="Tasks",
        aliases=("task", "tasks"),
        required_columns=("Task_ID",),
        optional_columns=("Owner", "Status", "Due_Date", "Related_To_ID"),
        id_column="Task_ID",
        relationship_keys=("Related_To_ID", "Lead_ID", "Deal_ID", "Account_ID"),
        supported_metrics=("task_count",),
        supported_dimensions=("Owner", "Status"),
    ),
    "Calls": ModuleDefinition(
        module_name="Calls",
        aliases=("call", "calls"),
        required_columns=("Call_ID",),
        optional_columns=("Owner", "Call_Result", "Related_To_ID"),
        id_column="Call_ID",
        relationship_keys=("Related_To_ID", "Lead_ID", "Deal_ID", "Account_ID"),
        supported_metrics=("call_count",),
        supported_dimensions=("Owner", "Call_Result"),
    ),
    "Meetings": ModuleDefinition(
        module_name="Meetings",
        aliases=("meeting", "meetings", "events"),
        required_columns=("Meeting_ID",),
        optional_columns=("Owner", "Status", "Start_Time", "Related_To_ID"),
        id_column="Meeting_ID",
        relationship_keys=("Related_To_ID", "Lead_ID", "Deal_ID", "Account_ID"),
        supported_metrics=("meeting_count",),
        supported_dimensions=("Owner", "Status"),
    ),
    "Quotes": ModuleDefinition(
        module_name="Quotes",
        aliases=("quote", "quotes"),
        required_columns=("Quote_ID",),
        optional_columns=("Deal_ID", "Account_ID", "Amount", "Stage"),
        id_column="Quote_ID",
        relationship_keys=("Quote_ID", "Deal_ID", "Account_ID"),
        supported_metrics=("quote_value", "quote_count"),
        supported_dimensions=("Stage", "Account_ID"),
    ),
    "Sales Orders": ModuleDefinition(
        module_name="Sales Orders",
        aliases=("sales orders", "sales order", "so"),
        required_columns=("Sales_Order_ID",),
        optional_columns=("Deal_ID", "Account_ID", "Amount", "Status"),
        id_column="Sales_Order_ID",
        relationship_keys=("Sales_Order_ID", "Deal_ID", "Account_ID"),
        supported_metrics=("sales_order_value", "sales_order_count"),
        supported_dimensions=("Status", "Account_ID"),
    ),
    "Purchase Orders": ModuleDefinition(
        module_name="Purchase Orders",
        aliases=("purchase orders", "purchase order", "po"),
        required_columns=("Purchase_Order_ID",),
        optional_columns=("Vendor_ID", "Amount", "Status"),
        id_column="Purchase_Order_ID",
        relationship_keys=("Purchase_Order_ID", "Vendor_ID"),
        supported_metrics=("purchase_order_value", "purchase_order_count"),
        supported_dimensions=("Status", "Vendor_ID"),
    ),
    "Invoices": ModuleDefinition(
        module_name="Invoices",
        aliases=("invoice", "invoices"),
        required_columns=("Invoice_ID",),
        optional_columns=("Deal_ID", "Account_ID", "Amount", "Status"),
        id_column="Invoice_ID",
        relationship_keys=("Invoice_ID", "Deal_ID", "Account_ID"),
        supported_metrics=("invoice_value", "invoice_count"),
        supported_dimensions=("Status", "Account_ID"),
    ),
    "Vendors": ModuleDefinition(
        module_name="Vendors",
        aliases=("vendor", "vendors", "suppliers"),
        required_columns=("Vendor_ID",),
        optional_columns=("Vendor_Name", "Region"),
        id_column="Vendor_ID",
        relationship_keys=("Vendor_ID",),
        supported_metrics=("vendor_count",),
        supported_dimensions=("Vendor_Name", "Region"),
    ),
    "Campaigns": ModuleDefinition(
        module_name="Campaigns",
        aliases=("campaign", "campaigns"),
        required_columns=("Campaign_ID",),
        optional_columns=("Campaign_Name", "Status", "Budget"),
        id_column="Campaign_ID",
        relationship_keys=("Campaign_ID",),
        supported_metrics=("campaign_count", "campaign_budget"),
        supported_dimensions=("Campaign_Name", "Status"),
    ),
}


SUPPORTED_MODULE_NAMES = tuple(MODULE_REGISTRY.keys())
