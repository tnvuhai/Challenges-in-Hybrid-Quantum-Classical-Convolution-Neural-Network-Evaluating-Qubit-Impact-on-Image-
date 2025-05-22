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
# epochs = 20
# num_runs = 5
# qubit_list = [2, 4, 6, 8]

epochs = 10
num_runs = 1
qubit_list = [6]


# Load Data
transform = transforms.Compose([
    # transforms.Grayscale(),
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor()
    # transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,))
])

TARGET_CLASSES = [0, 8]
CLASS_NAMES = ['airplane', 'ship']

# Tải dữ liệu
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Lọc chỉ lấy ảnh chó và mèo
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
            subset.dataset.targets[subset.indices[i]] = 0  # airplane
        elif label == 8:
            subset.dataset.targets[subset.indices[i]] = 1  # ship
    return subset

train_dataset = relabel(train_dataset)
test_dataset = relabel(test_dataset)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Quantum layer
class QuantumLayer(nn.Module):
    def __init__(self, n_qubits):
        super().__init__()
        dev = qml.device("default.qubit", wires=n_qubits)

        # weight_shapes = {"weights": (1, n_qubits)}

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            for i in range(n_qubits):
                qml.RX(np.pi * inputs[i], wires=i)
                qml.RZ(np.pi * inputs[i], wires=i)
            qml.templates.BasicEntanglerLayers(weights, wires=range(n_qubits))
            # for i in range(n_qubits - 1):
            #     qml.CNOT(wires=[i, i + 1])
            # qml.CNOT(wires=[n_qubits - 1, 0])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        # def circuit(inputs, weights):
        #     qml.AngleEmbedding(inputs,wires=range(n_qubits))
        #     qml.templates.BasicEntanglerLayers(weights, wires=range(n_qubits))
        #     return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        self.qnode = circuit
        self.weight = nn.Parameter(torch.rand((1, n_qubits)))  # This will be trained!
        

    def forward(self, x):
        outputs = []
        for i in range(x.shape[0]):
            out = self.qnode(x[i], self.weight)
            #out_tensor = out.to(dtype=torch.float32, device=x.device)
            out_tensor = torch.tensor(out, dtype=torch.float32).to(x.device)
            outputs.append(out_tensor)
        return torch.stack(outputs)


# Model
class HQCNN(nn.Module):
    def __init__(self, n_qubits):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 12, 5),
            nn.BatchNorm2d(12),
            nn.ReLU(),

            nn.Conv2d(12, 24, 5),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(),
            nn.Flatten(),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Linear(48, n_qubits),
            nn.Tanh()
        )
        self.q_layer = QuantumLayer(n_qubits)
        self.classifier = nn.Sequential(
            nn.Linear(n_qubits, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.q_layer(x)
        x = self.classifier(x)
        return x

# Model
# class HQCNN(nn.Module):
#     def __init__(self, n_qubits):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(3, 32, 3),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#             nn.Conv2d(32, 64, 3),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#             nn.Flatten(),
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.Linear(32, 16),
#             nn.ReLU()
#         )
#         self.classifier = nn.Sequential(
#             nn.Linear(16, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.classifier(x)
#         return x
    
# class HQCNN(nn.Module):
#     def __init__(self,  n_qubits):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#             nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#             nn.Flatten(),
#             nn.Linear(64 * 8 * 8, 128), nn.ReLU(),
#             nn.Linear(128, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         return self.net(x)

# Training and evaluation
def train_and_evaluate(n_qubits):
    model = HQCNN(n_qubits).to(device)
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
results = {"qubits": [], "accuracy_mean": [], "accuracy_std": [],
           "precision_mean": [], "precision_std": [], "f1_mean": [], "f1_std": [],
           "loss_mean": [], "loss_std": [], "Ram_usage": [], "Mean_time":[]}

for nq in qubit_list:
    accs, precs, f1s, losses = [], [], [], []
    TimeList, RamList = [], []
    print(f"Qubits: {nq}")
    for _ in range(num_runs):
        print(f"Num runs: {_}")
        start_time = datetime.now()
        acc, prec, f1, loss = train_and_evaluate(nq)
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
    print(RamList)
    results["qubits"].append(nq)
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
Df.to_excel("ResultCIFAR10.xlsx")

# Plotting
plt.errorbar(results["qubits"], results["accuracy_mean"], yerr=results["accuracy_std"], label="Accuracy", capsize=5)
plt.errorbar(results["qubits"], results["precision_mean"], yerr=results["precision_std"], label="Precision", capsize=5)
plt.errorbar(results["qubits"], results["loss_mean"], yerr=results["loss_std"], label="Loss", capsize=5)
plt.xlabel("Number of Qubits")
plt.ylabel("Metric Value")
plt.title("Hybrid QCNN Performance versus Number of Qubits in custom CIFAR 10")
plt.legend()
plt.grid(True)
plt.savefig("ResultCIFAR10.png",dpi=300)
plt.show()