# src/drift.py
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import os


# ---------- CONFIG ----------
VAL_DIR = "Plantvillage/val"
REF_DIR = "Plantvillage/ref_val"   # <-- You MUST create this once
IMG_SIZE = 224
BATCH = 32
NUM_WORKERS = 4
# ----------------------------


def load_loader(folder):
    tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    ds = datasets.ImageFolder(folder, transform=tfms)
    return DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS)


def extract_embeddings(loader, model, device):
    model.eval()
    emb_list = []

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            feat = model.features(x)
            feat = F.adaptive_avg_pool2d(feat, (1, 1))
            feat = feat.view(feat.size(0), -1)
            emb_list.extend(feat.cpu().numpy())

    return pd.DataFrame(emb_list)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load MobileNet for embedding extraction
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.to(device)

    # Load datasets
    if not os.path.exists(REF_DIR):
        raise ValueError(
            f"Reference folder missing!\n"
            f"Create a STATIC reference dataset folder:\n\n   {REF_DIR}\n\n"
            "This folder will NEVER change. Copy your initial validation data there."
        )

    ref_loader = load_loader(REF_DIR)
    cur_loader = load_loader(VAL_DIR)

    # Extract embeddings
    ref_emb = extract_embeddings(ref_loader, model, device)
    cur_emb = extract_embeddings(cur_loader, model, device)

    # Run drift
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_emb, current_data=cur_emb)
    report.save_html("drift_report.html")

    print("Image drift report saved → drift_report.html")


if __name__ == "__main__":
    main()
