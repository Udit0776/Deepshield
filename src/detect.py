import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms

# Define same model architecture
model = nn.Sequential(
    nn.Conv2d(3,16,3),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(16,32,3),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Flatten(),
    nn.Linear(32*30*30,128),
    nn.ReLU(),
    nn.Linear(128,2)
)

# Load trained model
model.load_state_dict(torch.load("models/deepshield_model.pth"))
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

# Load test image
img = cv2.imread("test1.jpg")

img = transform(img)
img = img.unsqueeze(0)

# Prediction
with torch.no_grad():
    output = model(img)
    _, predicted = torch.max(output,1)

if predicted.item() == 0:
    print("Prediction: FAKE")
else:
    print("Prediction: REAL")