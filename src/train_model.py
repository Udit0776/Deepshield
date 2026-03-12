import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

# Load dataset
dataset = datasets.ImageFolder("dataset", transform=transform)

loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Simple CNN model
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

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(5):

    for images, labels in loader:

        outputs = model(images)

        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Epoch", epoch+1, "completed")

# Save model
torch.save(model.state_dict(),"models/deepshield_model.pth")

print("Model training complete!")