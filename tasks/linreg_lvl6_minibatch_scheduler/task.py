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


TASK_ID = "linreg_lvl6_minibatch_scheduler"
OUTPUT_DIR = os.path.join("tasks", TASK_ID, "artifacts")


def get_task_metadata() -> Dict:
    return {
        "task_id": TASK_ID,
        "series": "Linear Regression",
        "algorithm": "Linear Regression (Mini-batch + Scheduler)",
        "interface_protocol": "pytorch_task_v1",
    }


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_synthetic_data(n_samples: int = 800, n_features: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(123)
    X = rng.normal(size=(n_samples, n_features)).astype(np.float32)
    w = np.array([1.5, -2.0, 0.8, 0.0, 1.2, -0.7, 0.5, 2.2], dtype=np.float32)
    b = 1.25
    y = X @ w + b + rng.normal(scale=0.4, size=n_samples).astype(np.float32)
    return X, y.astype(np.float32)


def make_dataloaders(batch_size: int = 64) -> Tuple[DataLoader, DataLoader]:
    X, y = _make_synthetic_data()
    n = X.shape[0]
    n_train = int(0.8 * n)

    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]

    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-6
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).view(-1, 1),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32).view(-1, 1),
    )

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
    )


def build_model(input_dim: int) -> nn.Module:
    return nn.Linear(input_dim, 1)


def train(model, train_loader, optimizer, criterion, device, scheduler=None, epochs: int = 180) -> Dict:
    loss_history = []
    lr_history = []

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

        epoch_loss = running / len(train_loader.dataset)
        loss_history.append(epoch_loss)
        lr_history.append(optimizer.param_groups[0]["lr"])

        if scheduler is not None:
            scheduler.step()

    return {"loss_history": loss_history, "lr_history": lr_history}


def evaluate(model, loader, device) -> Dict:
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            ys.append(yb.numpy())
            ps.append(pred)

    y_true = np.vstack(ys).ravel()
    y_pred = np.vstack(ps).ravel()
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mse))

    return {
        "mse": float(mse),
        "r2": float(r2),
        "rmse": rmse,
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
    plt.plot(train_result["loss_history"], label="loss")
    plt.title("Mini-batch Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"))
    plt.close()

    plt.figure()
    plt.plot(train_result["lr_history"], label="lr")
    plt.title("Learning Rate Schedule")
    plt.xlabel("Epoch")
    plt.ylabel("LR")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "lr_curve.png"))
    plt.close()

    plt.figure()
    plt.scatter(val_metrics["y_true"], val_metrics["y_pred"], alpha=0.55)
    mn = min(min(val_metrics["y_true"]), min(val_metrics["y_pred"]))
    mx = max(max(val_metrics["y_true"]), max(val_metrics["y_pred"]))
    plt.plot([mn, mx], [mn, mx], linestyle="--")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title("Validation Prediction Scatter")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "val_pred_scatter.png"))
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
        train_loader, val_loader = make_dataloaders(batch_size=64)

        sample_x, _ = next(iter(train_loader))
        model = build_model(sample_x.shape[1]).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

        train_result = train(model, train_loader, optimizer, criterion, device, scheduler=scheduler, epochs=180)
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

        assert val_metrics["r2"] > 0.90, f"Validation R2 too low: {val_metrics['r2']:.4f}"
        assert train_result["loss_history"][-1] < train_result["loss_history"][0] * 0.2, "Loss did not decrease enough"

        print("\nTask completed successfully.")
        sys.exit(0)
    except Exception as e:
        print(f"\nTask failed: {e}")
        sys.exit(1)
