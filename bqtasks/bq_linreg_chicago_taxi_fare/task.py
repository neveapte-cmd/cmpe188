import os
import sys
import json
import random
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

TASK_ID = "bq_linreg_chicago_taxi_fare"
OUTPUT_DIR = os.path.join("tasks", TASK_ID, "artifacts")


def get_task_metadata() -> Dict:
    return {
        "task_id": TASK_ID,
        "series": "Linear Regression",
        "algorithm": "Linear Regression (BigQuery Chicago Taxi Fare)",
        "interface_protocol": "pytorch_task_v1",
        "bigquery_dataset": "bigquery-public-data.chicago_taxi_trips.taxi_trips",
    }


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_from_bigquery() -> pd.DataFrame:
    from google.cloud import bigquery
    client = bigquery.Client()
    query = '''
    SELECT
      trip_miles,
      trip_seconds,
      pickup_community_area,
      dropoff_community_area,
      payment_type,
      company,
      fare
    FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
    WHERE fare IS NOT NULL
      AND trip_miles IS NOT NULL
      AND trip_seconds IS NOT NULL
      AND trip_miles > 0
      AND trip_seconds > 0
      AND fare > 0
      AND DATE(trip_start_timestamp) BETWEEN '2018-01-01' AND '2018-12-31'
      AND RAND() < 0.0005
    LIMIT 6000
    '''
    return client.query(query).to_dataframe()


def _preprocess(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    df = df.copy()
    df["pickup_community_area"] = df["pickup_community_area"].fillna(-1)
    df["dropoff_community_area"] = df["dropoff_community_area"].fillna(-1)
    df["payment_type"] = df["payment_type"].fillna("UNKNOWN")
    df["company"] = df["company"].fillna("UNKNOWN")

    df = pd.get_dummies(df, columns=["payment_type", "company"], drop_first=False)

    y = df["fare"].astype(np.float32).values
    X = df.drop(columns=["fare"]).astype(np.float32).values

    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-6
    X = (X - mean) / std
    return X, y


def make_dataloaders(batch_size: int = 64):
    df = _load_from_bigquery()
    if len(df) < 500:
        raise RuntimeError("BigQuery query returned too few rows; increase sample size or relax filters.")

    X, y = _preprocess(df)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).view(-1, 1),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32).view(-1, 1),
    )

    return DataLoader(train_ds, batch_size=batch_size, shuffle=True), DataLoader(val_ds, batch_size=batch_size, shuffle=False)


def build_model(input_dim: int) -> nn.Module:
    return nn.Linear(input_dim, 1)


def train(model, train_loader, optimizer, criterion, device, epochs: int = 80) -> Dict:
    loss_history = []
    for _ in range(epochs):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
        loss_history.append(running / len(train_loader.dataset))
    return {"loss_history": loss_history}


def evaluate(model, loader, device) -> Dict:
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb.to(device)).cpu().numpy()
            ys.append(yb.numpy())
            ps.append(pred)
    y_true = np.vstack(ys).ravel()
    y_pred = np.vstack(ps).ravel()

    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    return {
        "mse": float(mse),
        "r2": float(r2),
        "rmse": rmse,
        "mae": mae,
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
    }


def predict(model, x: torch.Tensor, device) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return model(x.to(device))


def save_artifacts(train_result: Dict, train_metrics: Dict, val_metrics: Dict) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.figure()
    plt.plot(train_result["loss_history"])
    plt.title("Chicago Taxi Fare Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"))
    plt.close()

    plt.figure()
    plt.scatter(val_metrics["y_true"], val_metrics["y_pred"], alpha=0.35)
    mn = min(min(val_metrics["y_true"]), min(val_metrics["y_pred"]))
    mx = max(max(val_metrics["y_true"]), max(val_metrics["y_pred"]))
    plt.plot([mn, mx], [mn, mx], linestyle="--")
    plt.xlabel("True Fare")
    plt.ylabel("Predicted Fare")
    plt.title("Validation Predictions")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "val_pred_scatter.png"))
    plt.close()

    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"metadata": get_task_metadata(), "train_metrics": train_metrics, "val_metrics": val_metrics}, f, indent=2)


if __name__ == "__main__":
    try:
        set_seed(42)
        device = get_device()
        train_loader, val_loader = make_dataloaders(batch_size=64)

        sample_x, _ = next(iter(train_loader))
        model = build_model(sample_x.shape[1]).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        train_result = train(model, train_loader, optimizer, criterion, device, epochs=80)
        train_metrics = evaluate(model, train_loader, device)
        val_metrics = evaluate(model, val_loader, device)

        print("=== Train Metrics ===")
        for k, v in train_metrics.items():
            if k not in {"y_true", "y_pred"}:
                print(f"{k}: {v:.6f}")

        print("\n=== Validation Metrics ===")
        for k, v in val_metrics.items():
            if k not in {"y_true", "y_pred"}:
                print(f"{k}: {v:.6f}")

        save_artifacts(train_result, train_metrics, val_metrics)

        assert val_metrics["r2"] > 0.50, f"Validation R2 too low: {val_metrics['r2']:.4f}"
        assert val_metrics["rmse"] < 10.0, f"Validation RMSE too high: {val_metrics['rmse']:.4f}"

        print("\nTask completed successfully.")
        sys.exit(0)
    except Exception as e:
        print(f"\nTask failed: {e}")
        print("Hint: authenticate BigQuery first (for example, gcloud auth application-default login).")
        sys.exit(1)
