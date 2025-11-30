# src/train.py
import os
import json
import torch
from torch import nn, optim
from torchvision import models
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.pytorch

from utils import load_params, set_seed, get_device, get_loaders


def build_model(num_classes):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    # freeze all
    for p in model.parameters():
        p.requires_grad = False

    # replace last layer
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


def main():
    params = load_params()
    set_seed(params["train"]["seed"])
    device = get_device()

    train_loader, val_loader, classes = get_loaders(params)

    mlflow_cfg = params["mlflow"]
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    model = build_model(len(classes)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.classifier[1].parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )

    with mlflow.start_run() as run:
        best_val_acc = 0
        num_epochs = params["train"]["num_epochs"]

        for epoch in range(1, num_epochs + 1):
            # training
            model.train()
            train_loss = 0
            train_preds = []
            train_labels = []

            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * imgs.size(0)
                train_preds.extend(outputs.argmax(1).cpu().tolist())
                train_labels.extend(labels.cpu().tolist())

            train_loss /= len(train_loader.dataset)
            train_acc = accuracy_score(train_labels, train_preds)

            # val
            model.eval()
            val_loss = 0
            val_preds = []
            val_labels = []

            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * imgs.size(0)
                    val_preds.extend(outputs.argmax(1).cpu().tolist())
                    val_labels.extend(labels.cpu().tolist())

            val_loss /= len(val_loader.dataset)
            val_acc = accuracy_score(val_labels, val_preds)

            print(f"Epoch {epoch}/{num_epochs} "
                  f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
                  f"Val Loss {val_loss:.4f} Acc {val_acc:.4f}")

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc
            }, step=epoch)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                os.makedirs("artifacts", exist_ok=True)
                torch.save(model.state_dict(), "artifacts/best_model.pt")

        # log to mlflow
        mlflow.pytorch.log_model(model, "model")

        with open("artifacts/classes.json", "w") as f:
            json.dump({"classes": classes}, f, indent=2)

        mlflow.log_artifacts("artifacts")


if __name__ == "__main__":
    main()
