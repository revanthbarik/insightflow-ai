#!/usr/bin/env python3
"""Generate synthetic leads and deals CSV data for InsightFlow AI demo."""

import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

LEAD_SOURCES = [
    "Website Inquiry",
    "WhatsApp Inquiry",
    "Sales Referral",
    "Trade Exhibition",
    "Partner Channel",
    "Cold Outreach",
    "Repeat Customer",
    "Zoho Campaign",
]

INDUSTRIES = [
    "Solar EPC Contractor",
    "Commercial & Industrial",
    "Residential Installer",
    "Trading Partner",
    "Utility Project",
    "Facility Management",
    "Real Estate Developer",
    "Government / Semi-Government",
]

LEAD_STATUSES = [
    "New Inquiry",
    "Contacted",
    "Qualified",
    "Quotation Sent",
    "Converted",
    "Lost",
]

REGIONS = [
    "Dubai",
    "Abu Dhabi",
    "Sharjah",
    "Ajman",
    "Ras Al Khaimah",
    "Fujairah",
    "Oman",
    "Saudi Arabia",
    "Africa Export",
]

CUSTOMER_TYPES = [
    "EPC",
    "Distributor",
    "Installer",
    "Commercial Client",
    "Government Client",
    "Export Customer",
]

PRODUCT_INTERESTS = [
    "Solar Panels",
    "Inverters",
    "Batteries",
    "Mounting Structures",
    "Cables & Accessories",
    "Complete Solar Kit",
]

DEAL_STAGES = [
    "New Inquiry",
    "Technical Discussion",
    "Quotation Sent",
    "Negotiation",
    "PO Received",
    "Closed Won",
    "Closed Lost",
]

FORECAST_CATEGORIES = [
    "Pipeline",
    "Best Case",
    "Commit",
    "Closed",
    "At Risk",
    "Omitted",
]

SALES_REPS = [
    "Aisha Khan",
    "Omar Ali",
    "Revanth Barik",
    "Sarah Mathew",
    "Zayed Hussain",
]

SOLAR_PREFIXES = [
    "SunPeak",
    "Desert",
    "Gulf",
    "Bright",
    "Green",
    "Solar",
    "Horizon",
    "Nova",
    "Apex",
    "Prime",
    "Golden",
    "Clear",
    "Skyline",
    "Volt",
    "Ray",
]

SOLAR_CORES = [
    "Ray",
    "Volt",
    "Grid",
    "Beam",
    "Power",
    "Energy",
    "Sun",
    "Solar",
    "Watt",
    "Photon",
    "Lumen",
    "Helios",
    "Radiance",
    "Current",
    "Charge",
]

SOLAR_SUFFIXES = [
    "EPC",
    "Solar",
    "Renewables",
    "Power Solutions",
    "Energy Trading",
    "Installations",
    "Projects",
    "Distribution",
    "Systems",
    "Contracting",
]

STAGE_FORECAST = {
    "New Inquiry": ["Pipeline"],
    "Technical Discussion": ["Pipeline", "Best Case"],
    "Quotation Sent": ["Pipeline", "Best Case", "At Risk"],
    "Negotiation": ["Best Case", "Commit", "At Risk"],
    "PO Received": ["Commit"],
    "Closed Won": ["Closed"],
    "Closed Lost": ["Omitted"],
}

STAGE_PROBABILITY = {
    "New Inquiry": (10, 20),
    "Technical Discussion": (20, 40),
    "Quotation Sent": (40, 60),
    "Negotiation": (60, 75),
    "PO Received": (80, 95),
    "Closed Won": (100, 100),
    "Closed Lost": (0, 0),
}

INDUSTRY_VALUE_RANGES = {
    "Solar EPC Contractor": (120_000, 1_200_000),
    "Commercial & Industrial": (150_000, 1_500_000),
    "Residential Installer": (35_000, 450_000),
    "Trading Partner": (80_000, 900_000),
    "Utility Project": (500_000, 3_000_000),
    "Facility Management": (60_000, 650_000),
    "Real Estate Developer": (200_000, 2_000_000),
    "Government / Semi-Government": (300_000, 2_500_000),
}

INDUSTRY_CUSTOMER_TYPE = {
    "Solar EPC Contractor": "EPC",
    "Commercial & Industrial": "Commercial Client",
    "Residential Installer": "Installer",
    "Trading Partner": "Distributor",
    "Utility Project": "Government Client",
    "Facility Management": "Commercial Client",
    "Real Estate Developer": "Commercial Client",
    "Government / Semi-Government": "Government Client",
}

DEAL_READY_STATUSES = ["Qualified", "Quotation Sent", "Converted", "Contacted"]

NUM_LEADS = 150
NUM_DEALS = 90
RANDOM_SEED = 42


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _generate_company_names(count: int, rng: random.Random) -> list[str]:
    names: set[str] = set()
    while len(names) < count:
        name = (
            f"{rng.choice(SOLAR_PREFIXES)} {rng.choice(SOLAR_CORES)} "
            f"{rng.choice(SOLAR_SUFFIXES)}"
        )
        names.add(name)
    return sorted(names)


def _random_created_date(rng: random.Random) -> datetime:
    start = datetime(2024, 1, 1)
    end = datetime(2025, 12, 31)
    return start + timedelta(days=rng.randint(0, (end - start).days))


def _estimated_value(industry: str, rng: random.Random) -> int:
    low, high = INDUSTRY_VALUE_RANGES[industry]
    raw = rng.randint(low, high)
    return int(round(raw / 5_000) * 5_000)


def _deal_amount(estimated_value: int, stage: str, rng: random.Random) -> int:
    if stage == "Closed Lost":
        factor = rng.uniform(0.75, 0.95)
    elif stage in ("Closed Won", "PO Received", "Negotiation"):
        factor = rng.uniform(0.92, 1.08)
    else:
        factor = rng.uniform(0.85, 1.05)
    return int(round(estimated_value * factor / 1_000) * 1_000)


def _closing_date(created_date: datetime, stage: str, rng: random.Random) -> datetime:
    if stage in ("Closed Won", "Closed Lost"):
        min_days, max_days = 30, 180
    elif stage in ("PO Received", "Negotiation"):
        min_days, max_days = 45, 240
    else:
        min_days, max_days = 60, 365
    return created_date + timedelta(days=rng.randint(min_days, max_days))


def _probability_for_stage(stage: str, rng: random.Random) -> int:
    low, high = STAGE_PROBABILITY[stage]
    if low == high:
        return low
    return rng.randint(low, high)


def _expected_margin(amount: int, stage: str, rng: random.Random) -> int:
    if stage == "Closed Lost":
        return 0
    margin_rate = rng.uniform(0.08, 0.22)
    return int(round(amount * margin_rate / 100) * 100)


def _customer_type(industry: str, region: str, rng: random.Random) -> str:
    if region == "Africa Export":
        return "Export Customer"
    if rng.random() < 0.75:
        return INDUSTRY_CUSTOMER_TYPE[industry]
    return rng.choice(CUSTOMER_TYPES)


def generate_leads(rng: random.Random) -> pd.DataFrame:
    company_names = _generate_company_names(NUM_LEADS, rng)
    rows = []

    for index in range(1, NUM_LEADS + 1):
        industry = rng.choice(INDUSTRIES)
        region = rng.choice(REGIONS)

        rows.append(
            {
                "Lead_ID": f"L{index:03d}",
                "Company_Name": company_names[index - 1],
                "Lead_Source": rng.choice(LEAD_SOURCES),
                "Industry": industry,
                "Estimated_Value": _estimated_value(industry, rng),
                "Status": rng.choice(LEAD_STATUSES),
                "Created_Date": _random_created_date(rng).strftime("%Y-%m-%d"),
                "Sales_Rep": rng.choice(SALES_REPS),
                "Region": region,
                "Customer_Type": _customer_type(industry, region, rng),
                "Product_Interest": rng.choice(PRODUCT_INTERESTS),
            }
        )

    return pd.DataFrame(rows)


def generate_deals(leads_df: pd.DataFrame, rng: random.Random) -> pd.DataFrame:
    deal_candidates = leads_df[leads_df["Status"].isin(DEAL_READY_STATUSES)]
    if len(deal_candidates) < NUM_DEALS:
        deal_candidates = leads_df

    selected_leads = deal_candidates.sample(n=NUM_DEALS, random_state=RANDOM_SEED)
    rows = []

    for deal_index, (_, lead) in enumerate(selected_leads.iterrows(), start=1):
        created_date = datetime.strptime(lead["Created_Date"], "%Y-%m-%d")
        stage = rng.choice(DEAL_STAGES)
        estimated_value = int(lead["Estimated_Value"])
        amount = _deal_amount(estimated_value, stage, rng)

        if rng.random() < 0.85:
            product_category = lead["Product_Interest"]
        else:
            product_category = rng.choice(PRODUCT_INTERESTS)

        rows.append(
            {
                "Deal_ID": f"D{deal_index:03d}",
                "Lead_ID": lead["Lead_ID"],
                "Company_Name": lead["Company_Name"],
                "Deal_Stage": stage,
                "Amount": amount,
                "Closing_Date": _closing_date(created_date, stage, rng).strftime(
                    "%Y-%m-%d"
                ),
                "Forecast_Category": rng.choice(STAGE_FORECAST[stage]),
                "Product_Category": product_category,
                "Region": lead["Region"],
                "Probability": _probability_for_stage(stage, rng),
                "Expected_Margin": _expected_margin(amount, stage, rng),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    rng = random.Random(RANDOM_SEED)
    data_dir = _project_root() / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    leads_df = generate_leads(rng)
    deals_df = generate_deals(leads_df, rng)

    leads_path = data_dir / "leads.csv"
    deals_path = data_dir / "deals.csv"

    leads_df.to_csv(leads_path, index=False)
    deals_df.to_csv(deals_path, index=False)

    print(f"Success: wrote {len(leads_df)} leads to {leads_path}")
    print(f"Success: wrote {len(deals_df)} deals to {deals_path}")


if __name__ == "__main__":
    main()
