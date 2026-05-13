from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yaml


def _require(d: Dict, path: str):
    # parser for YAML file
    node = d
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            raise ValueError(f"Missing required config key: {path}")
        node = node[key]
    return node


def _normalize_dataset_sizes(dataset_sizes: Dict) -> Dict:
    #make sure the splitting is full and even
    training = float(dataset_sizes.get("training"))
    validation = float(dataset_sizes.get("validation"))
    testing = float(dataset_sizes.get("testing"))
    total = training + validation + testing
    if total <= 0:
        raise ValueError("dataset_sizes must sum to a positive value.")
    return {
        "training": training / total,
        "validation": validation / total,
        "testing": testing / total,
    }


def load_config(config_path: str) -> Dict:
    """
    Load and validate YAML config.
    """
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("Config file must be a YAML object.")

    _require(cfg, "data_config.company_path")
    _require(cfg, "data_config.demand_path")
    _require(cfg, "data_config.output_dir")
    _require(cfg, "data_config.export_filenames.predictions_by_sku_country")
    _require(cfg, "data_config.export_filenames.feature_importance_by_sku_country")
    _require(cfg, "data_config.export_filenames.global_predictions_by_sku")
    _require(cfg, "data_config.export_filenames.global_predictions_all")
    _require(cfg, "data_config.export_filenames.training_metrics")
    _require(cfg, "data_config.keys.date")
    _require(cfg, "data_config.keys.country")
    _require(cfg, "data_config.keys.sku")
    _require(cfg, "data_config.keys.target")
    _require(cfg, "data_config.company_feature_columns")
    _require(cfg, "feature_config.groups")
    _require(cfg, "feature_config.calendar_features")
    _require(cfg, "training_config.dataset_sizes.training")
    _require(cfg, "training_config.dataset_sizes.validation")
    _require(cfg, "training_config.dataset_sizes.testing")
    _require(cfg, "training_config.forecast_horizon")
    _require(cfg, "training_config.min_history_length")
    _require(cfg, "training_config.ridge_alpha")
    _require(cfg, "training_config.non_negative_predictions")
    _require(cfg, "training_config.future_exogenous_strategy")

    cfg["training_config"]["dataset_sizes"] = _normalize_dataset_sizes(
        cfg["training_config"]["dataset_sizes"]
    )
    return cfg


def load_merged_training_table(config: Dict) -> pd.DataFrame:
    #checks for naming
    data_cfg = config["data_config"]
    keys = data_cfg["keys"]
    date_col = keys["date"]
    country_col = keys["country"]

    company_df = pd.read_csv(data_cfg["company_path"])
    demand_df = pd.read_csv(data_cfg["demand_path"])
    company_df[date_col] = pd.to_datetime(company_df[date_col])
    demand_df[date_col] = pd.to_datetime(demand_df[date_col])
    #merge on date col
    merge_keys = [date_col, country_col]
    df = demand_df.merge(company_df, on=merge_keys, how="left")

    required_company_features = list(data_cfg["company_feature_columns"])
    if df[required_company_features].isna().any().any():
        raise ValueError(
            "Merged table has missing company features. Check keys/date alignment in generated CSVs."
        )
    return df


def build_features(df: pd.DataFrame, config: Dict) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build lag/moving/exponential-moving features per (country, sku) series.
    """
    keys = config["data_config"]["keys"]
    feature_cfg = config["feature_config"]

    date_col = keys["date"]
    country_col = keys["country"]
    sku_col = keys["sku"]
    target_col = keys["target"]

    out = df.copy().sort_values([country_col, sku_col, date_col]).reset_index(drop=True)
    groups_cfg = feature_cfg["groups"]
    # go through yaml feats. and calc 
    for group_name, group_spec in groups_cfg.items():
        includes = list(group_spec.get("includes", []))
        apply_ops = dict(group_spec.get("apply", {}))

        for source_col in includes:
            if source_col not in out.columns:
                continue
            grouped = out.groupby([country_col, sku_col], sort=False)[source_col]

            for lag_k in apply_ops.get("lag", []):
                out[f"{group_name}_{source_col}_lag_{lag_k}"] = grouped.shift(int(lag_k))

            for win in apply_ops.get("mov", []):
                out[f"{group_name}_{source_col}_mov_{win}"] = grouped.transform(
                    lambda s, w=int(win): s.shift(1).rolling(w).mean()
                )

            for win in apply_ops.get("emov", []):
                out[f"{group_name}_{source_col}_emov_{win}"] = grouped.transform(
                    lambda s, w=int(win): s.shift(1).ewm(span=w, adjust=False).mean()
                )

    # Optional calendar features controlled by YAML.
    calendar_features = list(feature_cfg.get("calendar_features", []))
    if "day_of_week" in calendar_features:
        out["day_of_week"] = out[date_col].dt.weekday
    if "month" in calendar_features:
        out["month"] = out[date_col].dt.month
    if "day_of_month" in calendar_features:
        out["day_of_month"] = out[date_col].dt.day
    if "is_weekend" in calendar_features:
        out["is_weekend"] = (out[date_col].dt.weekday >= 5).astype(int)

    feature_cols = [
        c
        for c in out.columns
        if c not in {date_col, country_col, sku_col, target_col}
    ]
    return out, feature_cols


def prepare_model_data(config: Dict) -> Tuple[pd.DataFrame, List[str]]:
    """
    Full preprocessing pipeline for training:
    - load CSVs
    - merge on date+country
    - engineer features
    - drop rows with missing feature values
    """
    df = load_merged_training_table(config)
    fe_df, feature_cols = build_features(df, config)
    model_df = fe_df.dropna(subset=feature_cols).copy()
    if model_df.empty:
        raise ValueError("No training rows left after feature engineering.")
    return model_df, feature_cols
