import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
import pandas as pd
from datetime import datetime
import psutil
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import Subset

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
img_size = 12
batch_size = 16
epochs = 5
num_runs = 5

# Load Data
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor()
])

# CLASS_NAMES = ['A', 'B']
TARGET_CLASSES = [0, 1]


# Tải dữ liệu
train_dataset = datasets.MNIST(root='./Mnist',train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./Mnist',train=False, download=True, transform=transform)

# Lọc ảnh lấy theo target_classes
def filter_classes(dataset):
    indices = [i for i, (_, label) in enumerate(dataset) if label in TARGET_CLASSES]
    subset = Subset(dataset, indices)
    return subset

train_dataset = filter_classes(train_dataset)
test_dataset = filter_classes(test_dataset)

# Thay đổi label về 0, 1
def relabel(subset):
    subset.dataset.targets = np.array(subset.dataset.targets)
    for i in range(len(subset)):
        label = subset.dataset.targets[subset.indices[i]]
        if label == 0:
            subset.dataset.targets[subset.indices[i]] = 0  # 0
        elif label == 1:
            subset.dataset.targets[subset.indices[i]] = 1  # 1
    return subset

train_dataset = relabel(train_dataset)
test_dataset = relabel(test_dataset)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#model
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 12, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(12 * ((img_size - 2) // 2) ** 2, 52),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(52, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.classifier(x)
        return x

    

# Training and evaluation
def train_and_evaluate():
    model = CNN().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.float().to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}, loss: {loss}")

    model.eval()
    y_true, y_pred, losses = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.float().to(device).unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            losses.append(loss.item())
            y_true.extend(labels.cpu().numpy())
            y_pred.extend((outputs.cpu().numpy() > 0.5).astype(int).flatten())
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("Classification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    avg_loss = np.mean(losses)
    return acc, prec, f1, avg_loss

# Run experiments
results = {"accuracy_mean": [], "accuracy_std": [],
           "precision_mean": [], "precision_std": [], "f1_mean": [], "f1_std": [],
           "loss_mean": [], "loss_std": [], "Ram_usage": [], "Mean_time":[]}


accs, precs, f1s, losses = [], [], [], []
TimeList, RamList = [], []

for _ in range(num_runs):
    print(f"Num runs: {_}")
    start_time = datetime.now()
    acc, prec, f1, loss = train_and_evaluate()
    ram_after = psutil.Process(os.getpid()).memory_info().rss
    end_time = datetime.now()
    elapsed = end_time - start_time
    ram_delta_mb = (ram_after) / 1024 / 1024
    RamList.append(ram_delta_mb)
    accs.append(acc)
    precs.append(prec)
    f1s.append(f1)
    losses.append(loss)
    TimeList.append(elapsed)
results["accuracy_mean"].append(np.mean(accs))
results["accuracy_std"].append(np.std(accs))
results["precision_mean"].append(np.mean(precs))
results["precision_std"].append(np.std(precs))
results["f1_mean"].append(np.mean(f1s))
results["f1_std"].append(np.std(f1s))
results["loss_mean"].append(np.mean(losses))
results["loss_std"].append(np.std(losses))
results["Mean_time"].append(str(np.mean(TimeList)))
results["Ram_usage"].append(np.mean(RamList))

Df = pd.DataFrame(results)
Df.to_excel("ResultMNIST_CNN.xlsx")