import os
import sys
import json
import random
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


TASK_ID = "linreg_lvl5_huber_robust"
OUTPUT_DIR = os.path.join("tasks", TASK_ID, "artifacts")


def get_task_metadata() -> Dict:
    r"""
    Linear Regression with robust Huber loss.

    Huber loss:
    L_delta(r) = 0.5 r^2 if |r| <= delta, else delta (|r| - 0.5*delta)
    """
    return {
        "task_id": TASK_ID,
        "series": "Linear Regression",
        "algorithm": "Linear Regression (Robust Huber Loss)",
        "interface_protocol": "pytorch_task_v1",
    }


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_synthetic_data(n_samples: int = 600, n_features: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n_samples, n_features)).astype(np.float32)
    true_w = np.array([2.5, -1.2, 0.7, 0.0, 1.5], dtype=np.float32)
    true_b = -0.8
    y = X @ true_w + true_b + rng.normal(scale=0.6, size=n_samples).astype(np.float32)

    outlier_idx = rng.choice(n_samples, size=int(0.08 * n_samples), replace=False)
    y[outlier_idx] += rng.normal(loc=0.0, scale=12.0, size=outlier_idx.shape[0]).astype(np.float32)
    return X, y.astype(np.float32)


def make_dataloaders(batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    X, y = _make_synthetic_data()
    n = X.shape[0]
    n_train = int(0.8 * n)

    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def build_model(input_dim: int) -> nn.Module:
    return nn.Linear(input_dim, 1)


def train(model, train_loader, optimizer, criterion, device, epochs: int = 220) -> Dict:
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
            xb = xb.to(device)
            preds = model(xb).cpu().numpy()
            ys.append(yb.numpy())
            ps.append(preds)

    y_true = np.vstack(ys).ravel()
    y_pred = np.vstack(ps).ravel()

    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    median_ae = float(np.median(np.abs(y_true - y_pred)))

    return {
        "mse": float(mse),
        "r2": float(r2),
        "mae": mae,
        "median_absolute_error": median_ae,
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
    plt.title("Huber Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"))
    plt.close()

    plt.figure()
    plt.scatter(val_metrics["y_true"], val_metrics["y_pred"], alpha=0.6)
    mn = min(min(val_metrics["y_true"]), min(val_metrics["y_pred"]))
    mx = max(max(val_metrics["y_true"]), max(val_metrics["y_pred"]))
    plt.plot([mn, mx], [mn, mx], linestyle="--")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title("Validation Predictions")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pred_scatter.png"))
    plt.close()

    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"metadata": get_task_metadata(), "train_metrics": train_metrics, "val_metrics": val_metrics},
            f,
            indent=2,
        )


if __name__ == "__main__":
    try:
        set_seed(42)
        device = get_device()
        train_loader, val_loader = make_dataloaders(batch_size=32)

        sample_x, _ = next(iter(train_loader))
        model = build_model(sample_x.shape[1]).to(device)
        criterion = nn.HuberLoss(delta=1.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

        train_result = train(model, train_loader, optimizer, criterion, device, epochs=220)
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

        assert val_metrics["r2"] > 0.85, f"Validation R2 too low: {val_metrics['r2']:.4f}"
        assert val_metrics["mse"] < 25.0, f"Validation MSE too high: {val_metrics['mse']:.4f}"

        print("\nTask completed successfully.")
        sys.exit(0)
    except Exception as e:
        print(f"\nTask failed: {e}")
        sys.exit(1)
