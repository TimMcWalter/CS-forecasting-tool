# CS Forecasting Tool (DACH MVP)

Simple  project pipeline for demand forecasting of small ecommerce companies in `CH`, `DE`, and `AT`.

## Data Generation

Run:

```bash
uv run python main.py
```

This writes:

- `generated_data/company.csv`
- `generated_data/demand.csv`

## Dataset Shapes

### `company.csv`

Grain / primary key:

- `date + country`

Columns:

- `date` (`YYYY-MM-DD`)
- `country` (`CH`, `DE`, `AT`)
- `is_holiday` (0/1)
- `website_traffic`
- `add_to_carts`
- `conversion_rate`

Use:

- Shared drivers that affect all SKUs in a country on the same date.

### `demand.csv`

Grain / primary key:

- `date + country + sku`

Columns:

- `date` (`YYYY-MM-DD`)
- `country`
- `sku` (e.g. `SKU_001`)
- `demand` (target)

Use:

- SKU-level demand target per country and date.

### Merge Logic for ML

Train at SKU-country level by merging:

- `demand.csv` + `company.csv` on `date + country`

This creates one training row per `date + country + sku`.

## Training

Run:

```bash
uv run python train.py
```

Feature engineering and split settings come from `config.yaml`:

- `feature_config` for lag/moving/exponential-moving features
- `training_config` for train/val/test split and forecast horizon

## Training Exports

`train.py` writes to `generated_data/model_outputs/`:

### `predictions_by_sku_country.csv`

- Grain: `date + country + sku`
- Columns:
  - `date`
  - `country`
  - `sku`
  - `actual_demand` (empty for future forecast rows)
  - `predicted_demand`
  - `split` (`train`, `validation`, `test`, `forecast`)

### `feature_importance_by_sku_country.csv`

- Grain: `country + sku + feature`
- Columns:
  - `country`
  - `sku`
  - `feature`
  - `importance`
  - `coefficient`

### `global_predictions_by_country.csv`

- Aggregated over all SKUs
- Grain: `date + country + split`
- Columns:
  - `date`
  - `country`
  - `split`
  - `actual_demand_sum`
  - `predicted_demand_sum`
