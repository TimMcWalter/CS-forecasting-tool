from pathlib import Path

import pandas as pd

from datagen import datagen
'''
Script to pilot data gen, 

'''
def main(
    sku_count: int,
    startdate: str,
    enddate: str,
    countries=("CH", "DE", "AT"),
    seed: int = 42,
    outdir: str = "generated_data",
):
    result = datagen(
        numberOfSKU=sku_count,
        startdate=startdate,
        enddate=enddate,
        countries=countries,
        seed=seed,
    )

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    #
    '''
    Copy results of the datagen into dedicated export df
    espacially to ensure dataformating
    '''
    company_export = result["company"].copy()
    company_export["date"] = pd.to_datetime(company_export["date"]).dt.normalize()
    company_export = company_export[
        ["date", "country", "is_holiday", "website_traffic", "add_to_carts", "conversion_rate"]
    ].sort_values(["date", "country"])
    company_export["is_holiday"] = company_export["is_holiday"].astype(int)
    company_export["conversion_rate"] = company_export["conversion_rate"].round(4)
    company_export["date"] = company_export["date"].dt.strftime("%Y-%m-%d")

    demand_export = result["demand"].copy()
    demand_export["date"] = pd.to_datetime(demand_export["date"]).dt.normalize()
    demand_export = demand_export[["date", "country", "sku", "demand"]].sort_values(
        ["date", "country", "sku"]
    )
    demand_export["date"] = demand_export["date"].dt.strftime("%Y-%m-%d")

    company_path = outdir / "company.csv"
    demand_path = outdir / "demand.csv"

    company_export.to_csv(company_path, index=False)
    demand_export.to_csv(demand_path, index=False)

    print(f"Generated company rows: {len(company_export)}")
    print(f"Generated demand rows: {len(demand_export)}")
    print(f"Saved: {company_path}")
    print(f"Saved: {demand_path}")
    return {"company": company_export, "demand": demand_export}


if __name__ == "__main__":
    main(
        sku_count=50,
        startdate="2024-01-01",
        enddate="2026-03-31",
        countries=("CH", "DE", "AT"),
        seed=42,
        outdir="generated_data",
    )
