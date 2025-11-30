import json
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import streamlit as st

from src.utils import load_params, get_device


@st.cache_resource
def load_model():
    device = get_device()
    params = load_params()

    with open("artifacts/classes.json") as f:
        classes = json.load(f)["classes"]

    model = models.mobilenet_v2(weights=None)
    in_f = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_f, len(classes))

    model.load_state_dict(torch.load("artifacts/best_model.pt", map_location=device))
    model.to(device)
    model.eval()

    tfms = transforms.Compose([
        transforms.Resize((params["data"]["img_size"], params["data"]["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    return model, classes, tfms, device


st.title("PlantVillage MobileNet Classifier")

model, classes, tfms, device = load_model()

file = st.file_uploader("Upload leaf image", type=["jpg", "png", "jpeg"])
if file:
    img = Image.open(file).convert("RGB")
    st.image(img)

    with torch.no_grad():
        x = tfms(img).unsqueeze(0).to(device)
        out = model(x)
        probs = torch.softmax(out, 1)[0].cpu().numpy()

    pred = classes[probs.argmax()]
    st.subheader(f"Prediction: {pred}")
