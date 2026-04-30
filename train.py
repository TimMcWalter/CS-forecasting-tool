from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _parse_list_value(raw: str) -> List[int]:
    raw = raw.strip()
    if not raw.startswith("[") or not raw.endswith("]"):
        return []
    inner = raw[1:-1].strip()
    if not inner:
        return []
    values = []
    for part in inner.split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    return values


def load_config(config_path: str = "config.yaml") -> Dict:
    """
    Load a minimal config schema from config.yaml without external YAML dependencies.
    Expected schema:
      feature_config:
        group_x:
          includes: [...]
          apply:
            lag: [...]
            mov: [...]
            emov: [...]
      training_config:
        dataset_sizes:
          training: 0.6
          validation: 0.2
          testing: 0.2
        forecast_horizon: 30
    """
    text = Path(config_path).read_text(encoding="utf-8")
    lines = text.splitlines()

    config = {
        "feature_config": {},
        "training_config": {
            "dataset_sizes": {"training": 0.6, "validation": 0.2, "testing": 0.2},
            "forecast_horizon": 30,
        },
    }

    top_section = None
    current_group = None
    in_includes = False
    in_apply = False
    in_dataset_sizes = False

    for line in lines:
        # Ignore comments/blank lines.
        if not line.strip() or line.strip().startswith("#"):
            continue

        indent = len(line) - len(line.lstrip(" "))
        value = line.strip()

        if indent == 0 and value.endswith(":"):
            top_section = value[:-1]
            current_group = None
            in_includes = False
            in_apply = False
            in_dataset_sizes = False
            continue

        if top_section == "feature_config":
            if indent == 2 and value.endswith(":"):
                current_group = value[:-1]
                config["feature_config"][current_group] = {"includes": [], "apply": {}}
                in_includes = False
                in_apply = False
                continue

            if current_group is None:
                continue

            if indent == 4 and value.startswith("includes:"):
                after = value.split(":", 1)[1].strip()
                if after.startswith("["):
                    config["feature_config"][current_group]["includes"] = [
                        x.strip() for x in after[1:-1].split(",") if x.strip()
                    ]
                    in_includes = False
                else:
                    in_includes = True
                in_apply = False
                continue

            if indent == 4 and value == "apply:":
                in_apply = True
                in_includes = False
                continue

            if in_includes and indent >= 6 and value.startswith("- "):
                feat = value[2:].strip()
                if feat:
                    config["feature_config"][current_group]["includes"].append(feat)
                continue

            if in_apply and indent >= 6 and ":" in value:
                op, raw = value.split(":", 1)
                op = op.strip()
                config["feature_config"][current_group]["apply"][op] = _parse_list_value(raw)
                continue

        if top_section == "training_config":
            if indent == 2 and value == "dataset_sizes:":
                in_dataset_sizes = True
                continue

            if indent == 2 and value.startswith("forecast_horizon:"):
                raw = value.split(":", 1)[1].strip()
                if raw:
                    config["training_config"]["forecast_horizon"] = int(raw)
                continue

            if in_dataset_sizes and indent >= 4 and ":" in value:
                key, raw = value.split(":", 1)
                key = key.strip()
                raw = raw.strip()
                if key in {"training", "validation", "testing"} and raw:
                    config["training_config"]["dataset_sizes"][key] = float(raw)
                continue

    # Safety normalization of splits.
    splits = config["training_config"]["dataset_sizes"]
    total = splits["training"] + splits["validation"] + splits["testing"]
    if total <= 0:
        splits.update({"training": 0.6, "validation": 0.2, "testing": 0.2})
    elif abs(total - 1.0) > 1e-9:
        splits["training"] /= total
        splits["validation"] /= total
        splits["testing"] /= total

    return config


def build_features(df: pd.DataFrame, feature_config: Dict) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build engineered features per (country, sku) time series according to config.
    """
    out = df.copy().sort_values(["country", "sku", "date"]).reset_index(drop=True)

    for group_name, group_spec in feature_config.items():
        includes = group_spec.get("includes", [])
        apply_ops = group_spec.get("apply", {})

        for source_col in includes:
            if source_col not in out.columns:
                continue
            grouped = out.groupby(["country", "sku"], sort=False)[source_col]

            for lag_k in apply_ops.get("lag", []):
                out[f"{group_name}_{source_col}_lag_{lag_k}"] = grouped.shift(lag_k)

            for win in apply_ops.get("mov", []):
                out[f"{group_name}_{source_col}_mov_{win}"] = grouped.transform(
                    lambda s: s.shift(1).rolling(win).mean()
                )

            for win in apply_ops.get("emov", []):
                out[f"{group_name}_{source_col}_emov_{win}"] = grouped.transform(
                    lambda s: s.shift(1).ewm(span=win, adjust=False).mean()
                )

    # Simple calendar features.
    out["day_of_week"] = out["date"].dt.weekday
    out["month"] = out["date"].dt.month
    out["day_of_month"] = out["date"].dt.day
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)

    feature_cols = [
        c
        for c in out.columns
        if c
        not in {
            "date",
            "country",
            "sku",
            "demand",
        }
    ]
    return out, feature_cols


def _fit_linear_regression(X: np.ndarray, y: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """
    Fit a simple ridge-regularized linear model with intercept.
    Returns coefficients including intercept as coef[0].
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    X_aug = np.column_stack([np.ones(len(X)), X])
    eye = np.eye(X_aug.shape[1])
    eye[0, 0] = 0.0  # do not regularize intercept
    beta = np.linalg.pinv(X_aug.T @ X_aug + ridge * eye) @ X_aug.T @ y
    return beta


def _predict_linear(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    X_aug = np.column_stack([np.ones(len(X)), X])
    return X_aug @ beta


def _split_boundaries(n: int, train_ratio: float, val_ratio: float) -> Tuple[int, int]:
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    train_end = max(1, min(train_end, n))
    val_end = max(train_end, min(val_end, n))
    return train_end, val_end


def train_and_predict(
    company_path: str = "generated_data/company.csv",
    demand_path: str = "generated_data/demand.csv",
    config_path: str = "config.yaml",
    outdir: str = "generated_data/model_outputs",
):
    config = load_config(config_path)
    feature_config = config["feature_config"]
    train_cfg = config["training_config"]

    train_ratio = float(train_cfg["dataset_sizes"]["training"])
    val_ratio = float(train_cfg["dataset_sizes"]["validation"])
    horizon = int(train_cfg.get("forecast_horizon", 30))

    company_df = pd.read_csv(company_path)
    demand_df = pd.read_csv(demand_path)

    company_df["date"] = pd.to_datetime(company_df["date"])
    demand_df["date"] = pd.to_datetime(demand_df["date"])

    # Merge country-level shared features into each SKU row.
    df = demand_df.merge(company_df, on=["date", "country"], how="left")
    if df[["website_traffic", "add_to_carts", "conversion_rate"]].isna().any().any():
        raise ValueError("Merged training table has missing company features. Check date/country keys.")

    fe_df, feature_cols = build_features(df, feature_config)
    # Drop rows where lag/moving-window features are not available yet.
    model_df = fe_df.dropna(subset=feature_cols).copy()

    if model_df.empty:
        raise ValueError("No rows left after feature engineering. Reduce lag/window sizes in config.yaml.")

    predictions_rows = []
    importance_rows = []

    for (country, sku), grp in model_df.groupby(["country", "sku"], sort=False):
        grp = grp.sort_values("date").reset_index(drop=True)
        n = len(grp)
        if n < 10:
            continue

        train_end, val_end = _split_boundaries(n, train_ratio, val_ratio)

        X = grp[feature_cols].to_numpy(dtype=float)
        y = grp["demand"].to_numpy(dtype=float)

        X_train = X[:train_end]
        y_train = y[:train_end]
        beta = _fit_linear_regression(X_train, y_train)

        y_pred = np.maximum(0.0, _predict_linear(beta, X))

        for i, row in grp.iterrows():
            split = "train" if i < train_end else ("validation" if i < val_end else "test")
            predictions_rows.append(
                {
                    "date": row["date"],
                    "country": country,
                    "sku": sku,
                    "actual_demand": float(row["demand"]),
                    "predicted_demand": float(y_pred[i]),
                    "split": split,
                }
            )

        # Coefficient-based feature importance (scaled by train std for comparability).
        train_std = np.std(X_train, axis=0)
        scaled = np.abs(beta[1:] * train_std)
        denom = float(np.sum(scaled))
        importances = scaled / denom if denom > 0 else np.zeros_like(scaled)
        for feat, imp, coef in zip(feature_cols, importances, beta[1:]):
            importance_rows.append(
                {
                    "country": country,
                    "sku": sku,
                    "feature": feat,
                    "importance": float(imp),
                    "coefficient": float(coef),
                }
            )

        # Simple recursive future forecast for requested horizon.
        hist = grp[["date", "demand", "website_traffic", "add_to_carts", "conversion_rate", "is_holiday"]].copy()
        for _ in range(horizon):
            next_date = hist["date"].iloc[-1] + pd.Timedelta(days=1)
            next_row = {
                "date": next_date,
                "demand": np.nan,
                # Keep exogenous company metrics simple: carry last known level.
                "website_traffic": float(hist["website_traffic"].iloc[-1]),
                "add_to_carts": float(hist["add_to_carts"].iloc[-1]),
                "conversion_rate": float(hist["conversion_rate"].iloc[-1]),
                "is_holiday": int(hist["is_holiday"].iloc[-1]),
            }
            hist = pd.concat([hist, pd.DataFrame([next_row])], ignore_index=True)

            # Build a one-row frame to reuse feature logic consistently.
            tmp = hist.copy()
            tmp["country"] = country
            tmp["sku"] = sku
            tmp_fe, _ = build_features(tmp, feature_config)
            x_next = tmp_fe.iloc[-1][feature_cols].to_numpy(dtype=float)

            # If future feature row still has NaN (rare for large histories), use 0 fallback.
            x_next = np.nan_to_num(x_next, nan=0.0)
            y_next = max(0.0, float(_predict_linear(beta, x_next.reshape(1, -1))[0]))
            hist.loc[hist.index[-1], "demand"] = y_next

            predictions_rows.append(
                {
                    "date": next_date,
                    "country": country,
                    "sku": sku,
                    "actual_demand": np.nan,
                    "predicted_demand": y_next,
                    "split": "forecast",
                }
            )

    predictions_df = pd.DataFrame(predictions_rows).sort_values(["date", "country", "sku"]).reset_index(
        drop=True
    )
    feature_importance_df = pd.DataFrame(importance_rows).sort_values(
        ["country", "sku", "importance"], ascending=[True, True, False]
    )

    # Country-agnostic SKU table: aggregate over countries => date x sku x split.
    global_by_sku_df = (
        predictions_df.groupby(["date", "sku", "split"], as_index=False)[
            ["actual_demand", "predicted_demand"]
        ]
        .sum(min_count=1)
        .sort_values(["date", "sku", "split"])
        .reset_index(drop=True)
    )

    # Fully global table: aggregate over all countries and all SKUs => date x split.
    global_all_df = (
        predictions_df.groupby(["date", "split"], as_index=False)[["actual_demand", "predicted_demand"]]
        .sum(min_count=1)
        .rename(columns={"actual_demand": "actual_demand_sum", "predicted_demand": "predicted_demand_sum"})
        .sort_values(["date", "split"])
        .reset_index(drop=True)
    )

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    pred_path = out_path / "predictions_by_sku_country.csv"
    imp_path = out_path / "feature_importance_by_sku_country.csv"
    global_by_sku_path = out_path / "global_predictions_by_sku.csv"
    global_all_path = out_path / "global_predictions_all.csv"

    predictions_df.to_csv(pred_path, index=False)
    feature_importance_df.to_csv(imp_path, index=False)
    global_by_sku_df.to_csv(global_by_sku_path, index=False)
    global_all_df.to_csv(global_all_path, index=False)

    print(f"Saved: {pred_path}")
    print(f"Saved: {imp_path}")
    print(f"Saved: {global_by_sku_path}")
    print(f"Saved: {global_all_path}")

    return {
        "predictions": predictions_df,
        "feature_importance": feature_importance_df,
        "global_by_sku": global_by_sku_df,
        "global_all": global_all_df,
    }


def main():
    return train_and_predict(
        company_path="generated_data/company.csv",
        demand_path="generated_data/demand.csv",
        config_path="config.yaml",
        outdir="generated_data/model_outputs",
    )


if __name__ == "__main__":
    main()
