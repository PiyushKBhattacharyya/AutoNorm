import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from thop import profile
from torchvision import transforms
from configs import config
from data import get_dataloaders

# Ensure results directory exists
os.makedirs("results", exist_ok=True)

# --- Gradient Norm Computation ---
def compute_grad_norms(model):
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.norm().item()
    return grad_norms

# --- Classification Evaluation ---
def evaluate(model, loader, device, task_name):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x, task_name)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

# --- Regression Evaluation ---
def evaluate_regression(model, loader, device, task_name):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x, task_name)
            preds.append(out.cpu())
            targets.append(y.cpu())
    preds = torch.cat(preds)
    targets = torch.cat(targets)
    rmse = torch.sqrt(torch.mean((preds - targets) ** 2)).item()
    mae = torch.mean(torch.abs(preds - targets)).item()
    return rmse, mae

# --- Accuracy/Loss Plots ---
def plot_accuracy_loss(logs, prefix, output_dir="results"):
    plt.figure(figsize=(8, 5))
    plt.plot(logs["val_acc"], label="Val Accuracy", color='green', linewidth=2)
    plt.plot(logs["train_acc"], label="Train Accuracy", color='blue', linestyle='--')
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{prefix}_accuracy.png"))
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(logs["val_loss"], label="Val Loss", color='red', linewidth=2)
    plt.plot(logs["train_loss"], label="Train Loss", color='orange', linestyle='--')
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{prefix}_loss.png"))
    plt.close()

# --- RMSE/Loss Plots for Regression ---
def plot_rmse_loss(logs, prefix, output_dir="results"):
    plt.figure(figsize=(8, 5))
    plt.plot(logs["val_rmse"], label="Val RMSE", color='green', linewidth=2)
    plt.plot(logs["train_rmse"], label="Train RMSE", color='blue', linestyle='--')
    plt.title("RMSE over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{prefix}_rmse.png"))
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(logs["val_loss"], label="Val Loss", color='red', linewidth=2)
    plt.plot(logs["train_loss"], label="Train Loss", color='orange', linestyle='--')
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{prefix}_loss.png"))
    plt.close()

# --- Gradient Heatmap ---
def plot_gradient_heatmap(grads, title, save_path):
    layers = list(grads[0].keys())
    data = [[g[layer] for layer in layers] for g in grads]
    plt.figure(figsize=(10, 6))
    sns.heatmap(data, xticklabels=layers)
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

# --- FLOPs + Latency ---
def estimate_flops_and_latency(model, task, input_shape=(3, 32, 32), device='cpu'):
    model.eval()
    dummy_input = torch.randn(1, *input_shape).to(device)
    flops, params = profile(model, inputs=(dummy_input, task), verbose=False)
    torch.cuda.empty_cache()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(100):
            _ = model(dummy_input, task)
    end = time.perf_counter()
    latency = (end - start) / 100
    return flops, latency

from torch.nn.parameter import UninitializedParameter

def load_pretrained_partial(model, ckpt_path):
    if not os.path.exists(ckpt_path):
        print(f"[WARN] Checkpoint {ckpt_path} not found.")
        return
    print(f"[INFO] Loading checkpoint from {ckpt_path}")
    try:
        with torch.serialization.safe_globals([UninitializedParameter]):
            state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception as e:
        print(f"[WARN] weights_only load failed ({e}), retrying with weights_only=False")
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=False)


# --- Corruption Loaders ---
def add_noise_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(noise_tensor_clamp),
        transforms.Normalize((0.5,), (0.5,))
    ])

def noise_tensor_clamp(tensor):
    return torch.clamp(tensor + 0.2 * torch.randn_like(tensor), 0, 1)

def rotation_transform():
    return transforms.Compose([
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

def get_corrupted_loaders():
    return {
        "noise": get_dataloaders("CIFAR10", config['batch_size'], custom_transform=add_noise_transform())[1],
        "rotation": get_dataloaders("CIFAR10", config['batch_size'], custom_transform=rotation_transform())[1]
    }