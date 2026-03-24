import os
import sys
import json
import random
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

TASK_ID = "bq_logreg_census_income"
OUTPUT_DIR = os.path.join("tasks", TASK_ID, "artifacts")


def get_task_metadata() -> Dict:
    return {
        "task_id": TASK_ID,
        "series": "Logistic Regression",
        "algorithm": "Logistic Regression (BigQuery Census Income)",
        "interface_protocol": "pytorch_task_v1",
        "bigquery_dataset": "bigquery-public-data.ml_datasets.census_adult_income",
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
      age,
      education_num,
      hours_per_week,
      capital_gain,
      capital_loss,
      workclass,
      marital_status,
      occupation,
      relationship,
      race,
      sex,
      native_country,
      income_bracket
    FROM `bigquery-public-data.ml_datasets.census_adult_income`
    WHERE age IS NOT NULL
      AND education_num IS NOT NULL
      AND hours_per_week IS NOT NULL
    LIMIT 10000
    '''
    return client.query(query).to_dataframe()


def _preprocess(df: pd.DataFrame):
    df = df.copy().fillna("UNKNOWN")
    y = (df["income_bracket"].astype(str).str.contains(">50K")).astype(np.float32).values
    X_df = df.drop(columns=["income_bracket"])
    cat_cols = X_df.select_dtypes(include=["object"]).columns.tolist()
    X_df = pd.get_dummies(X_df, columns=cat_cols, drop_first=False)

    X = X_df.astype(np.float32).values
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-6
    X = (X - mean) / std
    return X, y


def make_dataloaders(batch_size: int = 64):
    df = _load_from_bigquery()
    if len(df) < 1000:
        raise RuntimeError("BigQuery query returned too few rows; retry with more rows.")

    X, y = _preprocess(df)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

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


def train(model, train_loader, optimizer, criterion, device, epochs: int = 60) -> Dict:
    loss_history = []
    for _ in range(epochs):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
        loss_history.append(running / len(train_loader.dataset))
    return {"loss_history": loss_history}


def evaluate(model, loader, device) -> Dict:
    model.eval()
    y_true_list, y_prob_list, y_pred_list = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            probs = torch.sigmoid(model(xb.to(device))).cpu().numpy().ravel()
            preds = (probs >= 0.5).astype(np.float32)
            y_true_list.append(yb.numpy().ravel())
            y_prob_list.append(probs)
            y_pred_list.append(preds)

    y_true = np.concatenate(y_true_list)
    y_prob = np.concatenate(y_prob_list)
    y_pred = np.concatenate(y_pred_list)

    mse = mean_squared_error(y_true, y_prob)
    r2 = r2_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        "mse": float(mse),
        "r2": float(r2),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "y_true": y_true.tolist(),
        "y_prob": y_prob.tolist(),
        "y_pred": y_pred.tolist(),
    }


def predict(model, x: torch.Tensor, device) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(x.to(device)))


def save_artifacts(train_result: Dict, train_metrics: Dict, val_metrics: Dict) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.figure()
    plt.plot(train_result["loss_history"])
    plt.title("Census Income Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"))
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
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        train_result = train(model, train_loader, optimizer, criterion, device, epochs=60)
        train_metrics = evaluate(model, train_loader, device)
        val_metrics = evaluate(model, val_loader, device)

        print("=== Train Metrics ===")
        for k, v in train_metrics.items():
            if k not in {"y_true", "y_prob", "y_pred"}:
                print(f"{k}: {v:.6f}")

        print("\n=== Validation Metrics ===")
        for k, v in val_metrics.items():
            if k not in {"y_true", "y_prob", "y_pred"}:
                print(f"{k}: {v:.6f}")

        save_artifacts(train_result, train_metrics, val_metrics)

        assert val_metrics["accuracy"] > 0.80, f"Validation accuracy too low: {val_metrics['accuracy']:.4f}"
        assert val_metrics["f1"] > 0.60, f"Validation F1 too low: {val_metrics['f1']:.4f}"

        print("\nTask completed successfully.")
        sys.exit(0)
    except Exception as e:
        print(f"\nTask failed: {e}")
        print("Hint: authenticate BigQuery first (for example, gcloud auth application-default login).")
        sys.exit(1)
