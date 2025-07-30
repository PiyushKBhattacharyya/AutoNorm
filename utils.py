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

#   Gradient Norm Computation
def compute_grad_norms(model):
    """
    Compute L2 norms of gradients for each parameter in the model.
    Useful for analyzing which layers are learning the most.
    """
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.norm().item()
    return grad_norms

# Accuracy Evaluation
def evaluate(model, loader, device):
    """
    Evaluate the model accuracy on a given dataloader.
    """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

# Accuracy & Loss Visualization
def plot_accuracy_loss(logs, prefix, output_dir="results"):
    """
    Plot and save training/validation accuracy and loss curves.
    """
    # Accuracy Plot
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

    # Loss Plot
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

# Heatmap of Gradient Norms Over Time
def plot_gradient_heatmap(grads, title, save_path):
    """
    Plot heatmap of gradient norms over time for each layer.
    """
    layers = list(grads[0].keys())
    data = [[g[layer] for layer in layers] for g in grads]

    plt.figure(figsize=(10, 6))
    sns.heatmap(data, xticklabels=layers)
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

# FLOPs + Latency Estimation
def estimate_flops_and_latency(model, input_shape=(3, 32, 32), device='cpu'):
    """
    Estimate model FLOPs and inference latency.
    """
    model.eval()
    dummy_input = torch.randn(1, *input_shape).to(device)

    # FLOPs & parameter count
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)

    # Latency: average over 100 runs
    torch.cuda.empty_cache()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(100):
            _ = model(dummy_input)
    end = time.perf_counter()

    latency = (end - start) / 100  # seconds
    return flops, latency

# Load Partial Pretrained Weights
def load_pretrained_partial(model, checkpoint_path):
    """
    Load a pretrained checkpoint and filter out unmatched layers.
    Useful for transfer learning or model surgery.
    """
    pretrained_dict = torch.load(checkpoint_path)
    model_dict = model.state_dict()

    filtered_dict = {
        k: v for k, v in pretrained_dict.items()
        if k in model_dict and model_dict[k].shape == v.shape
    }

    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)
    print(f"Loaded {len(filtered_dict)}/{len(pretrained_dict)} parameters from {checkpoint_path}")
    

def plot_attention_heads(attn_weights, save_path):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    if attn_weights is None:
        print(f"[WARN] Attention weights not available. Skipping attention visualization.")
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(15, 4))
    for h in range(min(attn_weights.shape[1], 4)):
        plt.subplot(1, 4, h + 1)
        sns.heatmap(attn_weights[0, h].detach().cpu().numpy(), cmap="viridis")
        plt.title(f"Head {h}")
        plt.xlabel("Key")
        plt.ylabel("Query")
    plt.suptitle("Attention Maps")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
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