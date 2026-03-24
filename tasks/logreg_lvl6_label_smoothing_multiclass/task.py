import os
import sys
import json
import random
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import matplotlib.pyplot as plt


TASK_ID = "logreg_lvl6_label_smoothing_multiclass"
OUTPUT_DIR = os.path.join("tasks", TASK_ID, "artifacts")


def get_task_metadata() -> Dict:
    return {
        "task_id": TASK_ID,
        "series": "Logistic Regression",
        "algorithm": "Softmax Regression (Label Smoothing)",
        "interface_protocol": "pytorch_task_v1",
    }


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_dataloaders(batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    X, y = make_blobs(
        n_samples=900,
        centers=[(-2, -1), (2, 0), (0, 3)],
        cluster_std=[0.8, 0.9, 0.85],
        random_state=42,
        n_features=2,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
    )


def build_model(input_dim: int) -> nn.Module:
    return nn.Linear(input_dim, 3)


def train(model, train_loader, optimizer, criterion, device, epochs: int = 120) -> Dict:
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
    y_true_list = []
    y_pred_list = []
    y_prob_list = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            y_true_list.append(yb.numpy())
            y_pred_list.append(preds)
            y_prob_list.append(probs)

    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)
    y_prob = np.vstack(y_prob_list)

    y_true_onehot = np.eye(3)[y_true]
    mse = mean_squared_error(y_true_onehot, y_prob)
    r2 = r2_score(y_true_onehot, y_prob)

    acc = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return {
        "mse": float(mse),
        "r2": float(r2),
        "accuracy": float(acc),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "y_prob": y_prob.tolist(),
    }


def predict(model, x: torch.Tensor, device) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        x = x.to(device)
        return torch.softmax(model(x), dim=1)


def save_artifacts(train_result: Dict, train_metrics: Dict, val_metrics: Dict) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plt.figure()
    plt.plot(train_result["loss_history"])
    plt.title("Multiclass Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"))
    plt.close()

    plt.figure()
    y_true = np.array(val_metrics["y_true"])
    y_pred = np.array(val_metrics["y_pred"])
    plt.scatter(np.arange(len(y_true)), y_true, s=10, label="true", alpha=0.7)
    plt.scatter(np.arange(len(y_pred)), y_pred, s=10, label="pred", alpha=0.7)
    plt.title("Validation True vs Predicted Classes")
    plt.xlabel("Sample Index")
    plt.ylabel("Class")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "val_classes.png"))
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
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        train_result = train(model, train_loader, optimizer, criterion, device, epochs=120)
        train_metrics = evaluate(model, train_loader, device)
        val_metrics = evaluate(model, val_loader, device)

        print("=== Train Metrics ===")
        for k, v in train_metrics.items():
            if k not in {"y_true", "y_pred", "y_prob"}:
                print(f"{k}: {v:.6f}")

        print("\n=== Validation Metrics ===")
        for k, v in val_metrics.items():
            if k not in {"y_true", "y_pred", "y_prob"}:
                print(f"{k}: {v:.6f}")

        save_artifacts(train_result, train_metrics, val_metrics)

        assert val_metrics["accuracy"] > 0.90, f"Validation accuracy too low: {val_metrics['accuracy']:.4f}"
        assert val_metrics["macro_f1"] > 0.90, f"Validation macro-F1 too low: {val_metrics['macro_f1']:.4f}"

        print("\nTask completed successfully.")
        sys.exit(0)
    except Exception as e:
        print(f"\nTask failed: {e}")
        sys.exit(1)
