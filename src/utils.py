# src/utils.py
import yaml
import torch
import random
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def load_params(path="params.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_loaders(params):
    data = params["data"]
    batch_size = params["train"]["batch_size"]

    train_transforms = transforms.Compose([
        transforms.Resize((data["img_size"], data["img_size"])),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((data["img_size"], data["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_ds = datasets.ImageFolder(data["train_dir"], transform=train_transforms)
    val_ds = datasets.ImageFolder(data["val_dir"], transform=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=data["num_workers"])

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=data["num_workers"])

    return train_loader, val_loader, train_ds.classes
