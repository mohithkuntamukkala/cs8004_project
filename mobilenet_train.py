import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import mlflow.exceptions

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import time
from pathlib import Path

EXPERIMENT_NAME = "PlantVillage_Abilation"
DATA_DIR = "PlantVillage"
EPOCHS = 25
BATCH_SIZE = 16
LR = 0.0005
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mlflow.set_tracking_uri("file:" + str(Path("mlruns").absolute()))
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name="MobileNetV2_FT") as run:  
    transform = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    }

    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform["train"])
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform["val"])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(train_dataset.classes))
    model = model.to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total Params: {total_params:,}")
    print(f"Trainable Params: {trainable_params:,}")

    mlflow.log_params({
        "model": "MobileNetV2",
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "total_params": total_params,
        "trainable_params": trainable_params
    })

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=LR)

    best_acc = 0.0
    train_times = []
    inference_times = []

    for epoch in range(EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            step_start = time.time()

            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            train_times.append(time.time() - step_start)

        train_loss = running_loss / len(train_dataset)
        train_acc = 100 * correct / total

        model.eval()
        val_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:

                inf_start = time.time()

                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)

                inference_times.append(time.time() - inf_start)

                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_dataset)
        val_acc = 100 * correct / total

        mlflow.log_metrics({
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        }, step=epoch)

        print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss={train_loss:.4f} Train Acc={train_acc:.2f}% | "
              f"Val Loss={val_loss:.4f} Val Acc={val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            mlflow.pytorch.log_model(model, artifact_path="best_model")

    mlflow.log_metric("avg_train_step_time", sum(train_times) / len(train_times))
    mlflow.log_metric("avg_inference_step_time", sum(inference_times) / len(inference_times))
    mlflow.log_metric("best_val_acc", best_acc)

  
    mlflow.pytorch.log_model(model, artifact_path="final_model")

    print("\nTraining complete.")
    print(f"Best Val Acc: {best_acc:.2f}%")
    print(f"Avg Train Step Time: {sum(train_times)/len(train_times):.6f} sec")
    print(f"Avg Inference Time: {sum(inference_times)/len(inference_times):.6f} sec")


    client = MlflowClient()
    model_uri = f"runs:/{run.info.run_id}/final_model"

    try:
        client.create_registered_model("PlantVillage_MobileNet")
    except mlflow.exceptions.RestException:
        pass  

    mv = client.create_model_version(
        name="PlantVillage_MobileNet",
        source=model_uri,
        run_id=run.info.run_id,
    )

    client.transition_model_version_stage(
        name="PlantVillage_MobileNet",
        version=mv.version,
        stage="Production",
        archive_existing_versions=True,
    )
















# import mlflow
# import mlflow.pytorch
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, transforms, models
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# import os
# import time

# EXPERIMENT_NAME = "PlantVillage_Abilation"   
# DATA_DIR = "PlantVillage"
# EPOCHS = 25
# BATCH_SIZE = 16
# LR = 0.0005
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# mlflow.set_experiment(EXPERIMENT_NAME)

# with mlflow.start_run(run_name="MobileNetV2_FT"):

#     transform = {
#         'train': transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406],
#                                  [0.229, 0.224, 0.225])
#         ]),
#         'val': transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406],
#                                  [0.229, 0.224, 0.225])
#         ])
#     }

#     train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform["train"])
#     val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform["val"])
#     train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

#     # ⬇️ MobileNetV2 (TorchVision)
#     model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

#     # Freeze all backbone
#     for param in model.parameters():
#         param.requires_grad = False

#     # Replace classifier head
#     in_features = model.classifier[1].in_features
#     model.classifier[1] = nn.Linear(in_features, len(train_dataset.classes))

#     model = model.to(DEVICE)

#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

#     print(f"Total Params: {total_params:,}")
#     print(f"Trainable Params: {trainable_params:,}")

#     mlflow.log_params({
#         "model": "MobileNetV2",
#         "epochs": EPOCHS,
#         "batch_size": BATCH_SIZE,
#         "learning_rate": LR,
#         "total_params": total_params,
#         "trainable_params": trainable_params
#     })

#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.classifier.parameters(), lr=LR)

#     best_acc = 0.0
#     train_times = []
#     inference_times = []

#     for epoch in range(EPOCHS):
#         model.train()
#         running_loss, correct, total = 0.0, 0, 0

#         for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
#             step_start = time.time()

#             images, labels = images.to(DEVICE), labels.to(DEVICE)

#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item() * images.size(0)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#             train_times.append(time.time() - step_start)

#         train_loss = running_loss / len(train_dataset)
#         train_acc = 100 * correct / total

#         model.eval()
#         val_loss, correct, total = 0.0, 0, 0

#         with torch.no_grad():
#             for images, labels in val_loader:

#                 inf_start = time.time()

#                 images, labels = images.to(DEVICE), labels.to(DEVICE)
#                 outputs = model(images)

#                 inference_times.append(time.time() - inf_start)

#                 loss = criterion(outputs, labels)
#                 val_loss += loss.item() * images.size(0)

#                 _, predicted = torch.max(outputs.data, 1)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()

#         val_loss /= len(val_dataset)
#         val_acc = 100 * correct / total

#         mlflow.log_metrics({
#             "train_loss": train_loss,
#             "train_acc": train_acc,
#             "val_loss": val_loss,
#             "val_acc": val_acc
#         }, step=epoch)

#         print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss={train_loss:.4f} Train Acc={train_acc:.2f}% | "
#               f"Val Loss={val_loss:.4f} Val Acc={val_acc:.2f}%")

#         if val_acc > best_acc:
#             best_acc = val_acc
#             mlflow.pytorch.log_model(model, artifact_path="best_model")

#     mlflow.log_metric("avg_train_step_time", sum(train_times) / len(train_times))
#     mlflow.log_metric("avg_inference_step_time", sum(inference_times) / len(inference_times))
#     mlflow.log_metric("best_val_acc", best_acc)

#     mlflow.pytorch.log_model(model, artifact_path="final_model")

#     print("\nTraining complete.")
#     print(f"Best Val Acc: {best_acc:.2f}%")
#     print(f"Avg Train Step Time: {sum(train_times)/len(train_times):.6f} sec")
#     print(f"Avg Inference Time: {sum(inference_times)/len(inference_times):.6f} sec")
