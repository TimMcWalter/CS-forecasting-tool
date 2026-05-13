import json
from urllib.error import URLError, HTTPError
from urllib.request import urlopen

import numpy as np
import pandas as pd
''' 

IMPORTANT: Use Date as Primary Key, and expect %Y-%m-%d
'''
# Public holiday API endpoint (Nager.Date) used for CH/DE/AT holiday signals.
HOLIDAY_API_URL = "https://date.nager.at/api/v3/PublicHolidays/{year}/{country}"

# Hard-coded demand uplift on public holidays.
DEMAND_HOLIDAY_FACTOR = 1.25

# Hard-coded day-of-week demand multipliers.
DEMAND_DOW_FACTORS = {
    0: 1.08,  # Monday
    1: 1.03,  # Tuesday
    2: 1.00,  # Wednesday
    3: 1.02,  # Thursday
    4: 1.12,  # Friday
    5: 0.92,  # Saturday
    6: 0.88,  # Sunday
}


def _fetch_holidays_for_country(year: int, country: str) -> pd.DataFrame:
    # Build country/year endpoint, e.g. .../PublicHolidays/2026/CH
    url = HOLIDAY_API_URL.format(year=year, country=country)
    try:
        # Read JSON response from API.
        with urlopen(url, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (URLError, HTTPError, TimeoutError):
        # Keep generation robust if API is temporarily unavailable.
        payload = []

    if not payload:
        return pd.DataFrame(columns=["date", "country", "holiday_name"])

    # Keep only the fields needed by the generator.
    rows = [
        {
            "date": pd.to_datetime(item["date"]),
            "country": country,
            "holiday_name": item.get("localName") or item.get("name", "Holiday"),
        }
        for item in payload
        if "date" in item
    ]
    return pd.DataFrame(rows)


def fetch_holiday_calendar(startdate, enddate, countries=("CH", "DE", "AT")) -> pd.DataFrame:
    # Normalize date boundaries once to support filtering and year iteration.
    start = pd.to_datetime(startdate)
    end = pd.to_datetime(enddate)
    years = range(start.year, end.year + 1)

    # Pull holidays per (country, year) and combine into one table.
    holiday_parts = []
    for country in countries:
        for year in years:
            holiday_parts.append(_fetch_holidays_for_country(year, country))

    holidays = pd.concat(holiday_parts, ignore_index=True) if holiday_parts else pd.DataFrame()
    if holidays.empty:
        return pd.DataFrame(columns=["date", "country", "holiday_name"])

    # Keep only requested date range and remove duplicate country/date entries.
    holidays = holidays[(holidays["date"] >= start) & (holidays["date"] <= end)].copy()
    holidays.drop_duplicates(subset=["date", "country"], inplace=True)
    return holidays


def gen_company(startdate, enddate, countries=("CH", "DE", "AT"), seed=42):
    """
    Generate daily company-level data for CH/DE/AT.
    Output columns include:
    - website_traffic
    - add_to_carts
    - conversion_rate
    """
    rng = np.random.default_rng(seed)
    # Daily calendar index for the requested window.
    dates = pd.date_range(start=pd.to_datetime(startdate), end=pd.to_datetime(enddate), freq="D")
    holidays = fetch_holiday_calendar(startdate, enddate, countries=countries)
    # Fast lookup: (normalized_date, country) -> holiday flag.
    holiday_lookup = set(zip(holidays["date"].dt.normalize(), holidays["country"]))

    # Base traffic by market size (simple hard-coded assumptions).
    base_traffic = {"CH": 9500, "DE": 17500, "AT": 7800}
    rows = []

    for country in countries:
        for dt in dates:
            # Calendar flags used as shared drivers across all SKUs.
            is_weekend = dt.weekday() >= 5
            is_holiday = (dt.normalize(), country) in holiday_lookup

            # Traffic model: base * weekend effect * holiday effect * random noise.
            traffic = base_traffic.get(country, 9000)
            traffic *= 0.87 if is_weekend else 1.0
            traffic *= 1.10 if is_holiday else 1.0
            traffic *= 1.0 + rng.normal(0, 0.09)
            website_traffic = max(100, int(traffic))

            # Conversion model with small holiday uplift and bounded range.
            conversion_rate = 0.021 + rng.normal(0, 0.0035)
            conversion_rate += 0.002 if is_holiday else 0.0
            conversion_rate = float(np.clip(conversion_rate, 0.005, 0.08))

            # Add-to-cart is derived from conversion via a simple multiplicative ratio.
            add_to_cart_rate = conversion_rate * (2.7 + rng.normal(0, 0.2))
            add_to_cart_rate = float(np.clip(add_to_cart_rate, 0.02, 0.50))
            add_to_carts = int(website_traffic * add_to_cart_rate)

            rows.append(
                {
                    "date": dt,
                    "country": country,
                    "is_holiday": is_holiday,
                    "website_traffic": website_traffic,
                    "add_to_carts": add_to_carts,
                    "conversion_rate": round(conversion_rate, 4),
                }
            )

    return pd.DataFrame(rows)


def gen_demand(numberOfSKU, startdate, enddate, countries=("CH", "DE", "AT"), seed=42):
    """
    Generate daily SKU demand by distributing expected company orders across SKUs.
    Includes hard-coded demand multipliers for holidays and day-of-week.
    """
    rng = np.random.default_rng(seed)
    # Demand is conditioned on company-level signals generated above.
    company_df = gen_company(startdate, enddate, countries=countries, seed=seed)

    # Stable SKU ID list and fixed SKU mix weights (sum to 1).
    sku_ids = [f"SKU_{i+1:03d}" for i in range(numberOfSKU)]
    weights = rng.dirichlet(np.ones(numberOfSKU))

    demand_rows = []
    for _, row in company_df.iterrows():
        # Hard-coded calendar multipliers applied directly in demand generation.
        weekday = int(row["date"].weekday())
        holiday_factor = DEMAND_HOLIDAY_FACTOR if bool(row["is_holiday"]) else 1.0
        weekday_factor = DEMAND_DOW_FACTORS.get(weekday, 1.0)

        # Expected daily orders at country/date level.
        expected_orders = row["website_traffic"] * row["conversion_rate"] * holiday_factor * weekday_factor
        for sku_id, weight in zip(sku_ids, weights):
            # Per-SKU demand sampled from a Poisson process around its expected share.
            lam = max(0.0, expected_orders * weight)
            demand = int(rng.poisson(lam))
            demand_rows.append(
                {
                    "date": row["date"],
                    "country": row["country"],
                    "sku": sku_id,
                    "demand": demand,
                    "holiday_factor": holiday_factor,
                    "weekday_factor": weekday_factor,
                }
            )

    demand_df = pd.DataFrame(demand_rows)
    return demand_df, company_df


def datagen(numberOfSKU, startdate, enddate, countries=("CH", "DE", "AT"), seed=42):
    # Generate both tables from the same configuration/seed.
    demand_df, company_df = gen_demand(
        numberOfSKU=numberOfSKU,
        startdate=startdate,
        enddate=enddate,
        countries=countries,
        seed=seed,
    )

    # Canonical sort order for deterministic exports and easier merges/debugging.
    company_df = company_df.copy().sort_values(["date", "country"]).reset_index(drop=True)
    demand_df = demand_df.copy().sort_values(["date", "country", "sku"]).reset_index(drop=True)

    # Validate primary keys required by downstream model training.
    company_pk_ok = not company_df.duplicated(subset=["date", "country"]).any()
    demand_pk_ok = not demand_df.duplicated(subset=["date", "country", "sku"]).any()
    if not company_pk_ok:
        raise ValueError("Company data has duplicate primary keys: (date, country).")
    if not demand_pk_ok:
        raise ValueError("Demand data has duplicate primary keys: (date, country, sku).")

    return {"company": company_df, "demand": demand_df}


