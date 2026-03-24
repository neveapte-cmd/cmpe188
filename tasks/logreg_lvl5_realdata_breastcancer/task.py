import os
import sys
import json
import random
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

import matplotlib.pyplot as plt


TASK_ID = "logreg_lvl5_realdata_breastcancer"
OUTPUT_DIR = os.path.join("tasks", TASK_ID, "artifacts")


def get_task_metadata() -> Dict:
    return {
        "task_id": TASK_ID,
        "series": "Logistic Regression",
        "algorithm": "Logistic Regression (Breast Cancer Dataset)",
        "interface_protocol": "pytorch_task_v1",
        "description": "Binary classification on sklearn breast cancer dataset using PyTorch.",
    }


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_dataloaders(batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    X, y = load_breast_cancer(return_X_y=True)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def build_model(input_dim: int) -> nn.Module:
    return nn.Linear(input_dim, 1)


def train(model, train_loader, optimizer, criterion, device, epochs: int = 100) -> Dict:
    loss_history = []

    for _ in range(epochs):
        model.train()
        running_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        loss_history.append(epoch_loss)

    return {"loss_history": loss_history}


def evaluate(model, loader, device) -> Dict:
    model.eval()

    y_true_list = []
    y_prob_list = []
    y_pred_list = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
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
    auc = roc_auc_score(y_true, y_prob)

    return {
        "mse": float(mse),
        "r2": float(r2),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "auc": float(auc),
        "y_true": y_true.tolist(),
        "y_prob": y_prob.tolist(),
        "y_pred": y_pred.tolist(),
    }


def predict(model, x: torch.Tensor, device) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        x = x.to(device)
        logits = model(x)
        probs = torch.sigmoid(logits)
    return probs


def save_artifacts(train_result: Dict, train_metrics: Dict, val_metrics: Dict) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    loss_history = train_result["loss_history"]

    plt.figure()
    plt.plot(loss_history)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"))
    plt.close()

    y_true = np.array(val_metrics["y_true"])
    y_pred = np.array(val_metrics["y_pred"])
    cm = confusion_matrix(y_true, y_pred)

    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    plt.close()

    fpr, tpr, _ = roc_curve(y_true, np.array(val_metrics["y_prob"]))
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {val_metrics['auc']:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"))
    plt.close()

    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "task_metadata": get_task_metadata(),
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    try:
        set_seed(42)
        device = get_device()
        print(f"Using device: {device}")

        train_loader, val_loader = make_dataloaders(batch_size=32)
        sample_x, _ = next(iter(train_loader))
        input_dim = sample_x.shape[1]

        model = build_model(input_dim).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        train_result = train(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epochs=100,
        )

        train_metrics = evaluate(model, train_loader, device)
        val_metrics = evaluate(model, val_loader, device)

        print("\n=== Train Metrics ===")
        for k, v in train_metrics.items():
            if k not in {"y_true", "y_prob", "y_pred"}:
                print(f"{k}: {v:.6f}")

        print("\n=== Validation Metrics ===")
        for k, v in val_metrics.items():
            if k not in {"y_true", "y_prob", "y_pred"}:
                print(f"{k}: {v:.6f}")

        save_artifacts(train_result, train_metrics, val_metrics)

        assert val_metrics["accuracy"] > 0.93, "Validation accuracy below threshold"
        assert val_metrics["f1"] > 0.93, "Validation F1 below threshold"

        print("\nTask completed successfully.")
        sys.exit(0)

    except Exception as e:
        print(f"\nTask failed: {e}")
        sys.exit(1)
