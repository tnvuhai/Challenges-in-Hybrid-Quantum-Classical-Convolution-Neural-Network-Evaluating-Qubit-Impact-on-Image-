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
img_size = 12
qubit_list = [2, 4, 6, 8]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Quantum layer
class QuantumLayer(nn.Module):
    def __init__(self, n_qubits):
        super().__init__()
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            for i in range(n_qubits):
                qml.RX(np.pi * inputs[i], wires=i)
                qml.RZ(np.pi * inputs[i], wires=i)
            qml.templates.BasicEntanglerLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        self.qnode = circuit
        self.weight = nn.Parameter(torch.rand((1, n_qubits))) 
        

    def forward(self, x):
        outputs = []
        for i in range(x.shape[0]):
            out = self.qnode(x[i], self.weight)
            out_tensor = torch.tensor(out, dtype=torch.float32).to(x.device)
            outputs.append(out_tensor)
        return torch.stack(outputs)



class HQCNN(nn.Module):
    def __init__(self, n_qubits):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 12, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(12 * ((img_size - 2) // 2) ** 2, n_qubits),
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

def param_report(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    conv_params = sum(p.numel() for p in model.conv.parameters())
    q_params    = sum(p.numel() for p in model.q_layer.parameters())   
    clf_params  = sum(p.numel() for p in model.classifier.parameters())

    q_detail = {n: p.shape for n, p in model.q_layer.named_parameters()}

    print("== Parameter counts ==")
    print(f"Total: {total}")
    print(f"Conv: {conv_params}")
    print(f"QuantumLayer: {q_params}  | detail: {q_detail}")
    print(f"Classifier: {clf_params}")
    return {
        "total": total,
        "conv": conv_params,
        "quantum": q_params,
        "quantum_detail": q_detail,
        "classifier": clf_params,
    }

def param_report_cnn(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())

    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Module):
        clf_params = sum(p.numel() for p in model.classifier.parameters())
    else:
        clf_params = 0

    if hasattr(model, "features") and isinstance(model.features, nn.Module):
        feat_params = sum(p.numel() for p in model.features.parameters())
    elif hasattr(model, "conv") and isinstance(model.conv, nn.Module):
        feat_params = sum(p.numel() for p in model.conv.parameters())
    else:
        feat_params = total - clf_params

    if hasattr(model, "features") and isinstance(model.features, nn.Module):
        cnn_named = dict(model.features.named_parameters())
    elif hasattr(model, "conv") and isinstance(model.conv, nn.Module):
        cnn_named = dict(model.conv.named_parameters())
    else:
        cnn_named = {n: p for n, p in model.named_parameters()
                     if not (hasattr(model, "classifier") and n.startswith("classifier"))}

    cnn_detail = {n: tuple(p.shape) for n, p in cnn_named.items()}

    print("== CNN Parameter counts ==")
    print(f"Total: {total}")
    print(f"CNN block: {feat_params}")
    print(f"Classifier: {clf_params}")
    return {
        "total": total,
        "cnn": feat_params,
        "classifier": clf_params,
        "cnn_detail": cnn_detail,
    }


if __name__ == "__main__":
    print("Device:", device)

    modelCNN = CNN().to(device)
    rpt_cnn = param_report_cnn(modelCNN)
    print(f"Total CNN params: {rpt_cnn['total']}")
    for nq in qubit_list:
        model = HQCNN(nq).to(device)
        print(f"\n[Qubits = {nq}]")
        rpt = param_report(model)

        print(f"Total params: {rpt['total']}")
        print(f"QuantumLayer params: {rpt['quantum']}  details: {rpt['quantum_detail']}")
