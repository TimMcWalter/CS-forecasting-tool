from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from preprocessing import build_features, load_config, prepare_model_data


def _fit_linear_regression(X: np.ndarray, y: np.ndarray, ridge_alpha: float) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    X_aug = np.column_stack([np.ones(len(X)), X])
    eye = np.eye(X_aug.shape[1], dtype=float)
    eye[0, 0] = 0.0
    beta = np.linalg.pinv(X_aug.T @ X_aug + float(ridge_alpha) * eye) @ X_aug.T @ y
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


def _metrics_from_frame(df: pd.DataFrame, actual_col: str, pred_col: str) -> Dict:
    if df.empty:
        return {"n_obs": 0, "mae": None, "rmse": None, "mape_pct": None}

    actual = df[actual_col].to_numpy(dtype=float)
    pred = df[pred_col].to_numpy(dtype=float)
    err = pred - actual

    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))

    non_zero_mask = np.abs(actual) > 1e-12
    if np.any(non_zero_mask):
        mape = float(np.mean(np.abs(err[non_zero_mask] / actual[non_zero_mask])) * 100.0)
    else:
        mape = None

    return {"n_obs": int(len(df)), "mae": mae, "rmse": rmse, "mape_pct": mape}


def train_and_predict(config_path: str):
    config = load_config(config_path)

    data_cfg = config["data_config"]
    keys = data_cfg["keys"]
    train_cfg = config["training_config"]

    date_col = keys["date"]
    country_col = keys["country"]
    sku_col = keys["sku"]
    target_col = keys["target"]

    train_ratio = float(train_cfg["dataset_sizes"]["training"])
    val_ratio = float(train_cfg["dataset_sizes"]["validation"])
    horizon = int(train_cfg["forecast_horizon"])
    min_history_length = int(train_cfg["min_history_length"])
    ridge_alpha = float(train_cfg["ridge_alpha"])
    non_negative = bool(train_cfg["non_negative_predictions"])
    future_exog_strategy = str(train_cfg["future_exogenous_strategy"]).strip().lower()

    if future_exog_strategy != "carry_last":
        raise ValueError("Only future_exogenous_strategy='carry_last' is supported in this MVP.")

    model_df, feature_cols = prepare_model_data(config)
    model_df = model_df.sort_values([date_col, country_col, sku_col]).reset_index(drop=True)

    predictions_rows = []
    importance_rows = []

    company_feature_columns = list(data_cfg["company_feature_columns"])

    for (country, sku), grp in model_df.groupby([country_col, sku_col], sort=False):
        grp = grp.sort_values(date_col).reset_index(drop=True)
        n = len(grp)
        if n < min_history_length:
            continue

        train_end, val_end = _split_boundaries(n, train_ratio, val_ratio)

        X = grp[feature_cols].to_numpy(dtype=float)
        y = grp[target_col].to_numpy(dtype=float)

        X_train = X[:train_end]
        y_train = y[:train_end]
        beta = _fit_linear_regression(X_train, y_train, ridge_alpha=ridge_alpha)

        y_pred_hist = _predict_linear(beta, X)
        if non_negative:
            y_pred_hist = np.maximum(0.0, y_pred_hist)

        for i, row in grp.iterrows():
            split = "train" if i < train_end else ("validation" if i < val_end else "test")
            predictions_rows.append(
                {
                    date_col: row[date_col],
                    country_col: country,
                    sku_col: sku,
                    "actual_demand": float(row[target_col]),
                    "predicted_demand": float(y_pred_hist[i]),
                    "split": split,
                }
            )

        train_std = np.std(X_train, axis=0)
        scaled = np.abs(beta[1:] * train_std)
        denom = float(np.sum(scaled))
        importances = scaled / denom if denom > 0 else np.zeros_like(scaled)
        for feat, imp, coef in zip(feature_cols, importances, beta[1:]):
            importance_rows.append(
                {
                    country_col: country,
                    sku_col: sku,
                    "feature": feat,
                    "importance": float(imp),
                    "coefficient": float(coef),
                }
            )

        # Recursive future prediction.
        hist_cols = [date_col, target_col, *company_feature_columns]
        hist = grp[hist_cols].copy()
        for _ in range(horizon):
            next_date = hist[date_col].iloc[-1] + pd.Timedelta(days=1)
            next_row = {date_col: next_date, target_col: np.nan}
            for col in company_feature_columns:
                next_row[col] = hist[col].iloc[-1]

            hist = pd.concat([hist, pd.DataFrame([next_row])], ignore_index=True)
            tmp = hist.copy()
            tmp[country_col] = country
            tmp[sku_col] = sku
            tmp_fe, _ = build_features(tmp, config)
            x_next = tmp_fe.iloc[-1][feature_cols].to_numpy(dtype=float)
            x_next = np.nan_to_num(x_next, nan=0.0)

            y_next = float(_predict_linear(beta, x_next.reshape(1, -1))[0])
            if non_negative:
                y_next = max(0.0, y_next)

            hist.loc[hist.index[-1], target_col] = y_next
            predictions_rows.append(
                {
                    date_col: next_date,
                    country_col: country,
                    sku_col: sku,
                    "actual_demand": np.nan,
                    "predicted_demand": y_next,
                    "split": "forecast",
                }
            )

    predictions_df = pd.DataFrame(predictions_rows).sort_values(
        [date_col, country_col, sku_col]
    ).reset_index(drop=True)
    feature_importance_df = pd.DataFrame(importance_rows).sort_values(
        [country_col, sku_col, "importance"], ascending=[True, True, False]
    )

    global_by_sku_df = (
        predictions_df.groupby([date_col, sku_col, "split"], as_index=False)[
            ["actual_demand", "predicted_demand"]
        ]
        .sum(min_count=1)
        .sort_values([date_col, sku_col, "split"])
        .reset_index(drop=True)
    )

    global_all_df = (
        predictions_df.groupby([date_col, "split"], as_index=False)[["actual_demand", "predicted_demand"]]
        .sum(min_count=1)
        .rename(columns={"actual_demand": "actual_demand_sum", "predicted_demand": "predicted_demand_sum"})
        .sort_values([date_col, "split"])
        .reset_index(drop=True)
    )

    eval_df = predictions_df[predictions_df["split"].isin(["train", "validation", "test"])].copy()
    eval_df = eval_df.dropna(subset=["actual_demand", "predicted_demand"])

    per_model_metrics = []
    for (country, sku, split), grp in eval_df.groupby([country_col, sku_col, "split"], sort=False):
        metric = _metrics_from_frame(grp, "actual_demand", "predicted_demand")
        per_model_metrics.append({country_col: country, sku_col: sku, "split": split, **metric})

    per_model_overall = []
    for (country, sku), grp in eval_df.groupby([country_col, sku_col], sort=False):
        metric = _metrics_from_frame(grp, "actual_demand", "predicted_demand")
        per_model_overall.append(
            {country_col: country, sku_col: sku, "split": "all_non_forecast", **metric}
        )

    global_metrics_by_split = []
    for split, grp in eval_df.groupby("split", sort=False):
        metric = _metrics_from_frame(grp, "actual_demand", "predicted_demand")
        global_metrics_by_split.append({"split": split, **metric})

    global_metrics_overall = _metrics_from_frame(eval_df, "actual_demand", "predicted_demand")

    out_path = Path(data_cfg["output_dir"])
    out_path.mkdir(parents=True, exist_ok=True)

    pred_path = out_path / "predictions_by_sku_country.csv"
    imp_path = out_path / "feature_importance_by_sku_country.csv"
    global_by_sku_path = out_path / "global_predictions_by_sku.csv"
    global_all_path = out_path / "global_predictions_all.csv"
    metrics_json_path = out_path / "training_metrics.json"

    predictions_df.to_csv(pred_path, index=False)
    feature_importance_df.to_csv(imp_path, index=False)
    global_by_sku_df.to_csv(global_by_sku_path, index=False)
    global_all_df.to_csv(global_all_path, index=False)

    metrics_payload = {
        "summary": {
            "config_path": str(config_path),
            "n_rows_evaluated": int(len(eval_df)),
            "n_models_country_sku": int(eval_df[[country_col, sku_col]].drop_duplicates().shape[0]),
            "dataset_sizes": train_cfg["dataset_sizes"],
            "forecast_horizon": horizon,
            "min_history_length": min_history_length,
            "ridge_alpha": ridge_alpha,
            "non_negative_predictions": non_negative,
            "future_exogenous_strategy": future_exog_strategy,
        },
        "global_metrics_overall": global_metrics_overall,
        "global_metrics_by_split": global_metrics_by_split,
        "per_model_metrics_by_split": per_model_metrics,
        "per_model_metrics_overall": per_model_overall,
    }
    metrics_json_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    print(f"Saved: {pred_path}")
    print(f"Saved: {imp_path}")
    print(f"Saved: {global_by_sku_path}")
    print(f"Saved: {global_all_path}")
    print(f"Saved: {metrics_json_path}")

    return {
        "predictions": predictions_df,
        "feature_importance": feature_importance_df,
        "global_by_sku": global_by_sku_df,
        "global_all": global_all_df,
        "metrics": metrics_payload,
    }


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python train.py <config_path>")
    return train_and_predict(sys.argv[1])


if __name__ == "__main__":
    main()
