import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from tabulate import tabulate
from torchvision import transforms

from model import *
from train import train_model
from utils import *
from data import get_dataloaders
from configs import config
from factory import model_variants, make_only_dyt, make_only_ln

def main():
    os.makedirs(config['save_dir'], exist_ok=True)
    results = []

    print("\n[Phase 1] Pretraining AutoNorm on MNIST")
    mnist_train, mnist_test = get_dataloaders("MNIST", config['batch_size'])
    base_model = TransformerWithAutoNorm(input_dim=784).to(config['device'])
    optimizer = optim.Adam(base_model.parameters(), lr=config['learning_rate'])
    logs_mnist = train_model(base_model, mnist_train, mnist_test, nn.CrossEntropyLoss(), optimizer, config)
    torch.save(base_model.state_dict(), os.path.join(config['save_dir'], "pretrained_mnist.pth"))
    plot_accuracy_loss(logs_mnist, prefix="MNIST_Pretrain")

    print("\n[Phase 2] Transfer to CIFAR10")
    cifar_train, cifar_test = get_dataloaders("CIFAR10", config['batch_size'])
    corrupted_loaders = get_corrupted_loaders()

    for model_name, model_fn in model_variants.items():
        strategy = "finetune"
        print(f"\n[Model: {model_name} | Strategy: {strategy.upper()}]")
        model = model_fn().to(config['device'])

        if model_name == "AutoNorm":
            load_pretrained_partial(model, os.path.join(config['save_dir'], "pretrained_mnist.pth"))

        optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=config['learning_rate'])
        logs = train_model(model, cifar_train, cifar_test, nn.CrossEntropyLoss(), optimizer, config)

        prefix = f"{model_name}_{strategy}"
        plot_accuracy_loss(logs, prefix=prefix)
        if 'grad_norms' in logs:
            plot_gradient_heatmap(logs['grad_norms'], title=f"Gradient Norms: {prefix}", save_path=os.path.join(config['save_dir'], f"{prefix}_grad.png"))

        corruption_results = {}
        for corruption, loader in corrupted_loaders.items():
            try:
                corrupted_acc = evaluate(model, loader, config['device'])
            except Exception as e:
                corrupted_acc = None
                print(f"[WARN] {corruption} evaluation failed: {e}")
            corruption_results[corruption] = corrupted_acc

        flops, latency = estimate_flops_and_latency(model, input_shape=(3, 32, 32), device=config['device'])
        results.append([
            f"{model_name}-{strategy}",
            logs.get("train_acc", [-1])[-1],
            logs.get("train_loss", [-1])[-1],
            logs.get("val_acc", [-1])[-1],
            latency,
            flops,
            corruption_results.get("noise", None),
            corruption_results.get("rotation", None),
        ])

    print("\n[AutoML Norm Search Comparison]")
    for label, model_fn in {"OnlyDyT": make_only_dyt, "OnlyLayerNorm": make_only_ln}.items():
        model = model_fn().to(config['device'])
        optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=config['learning_rate'])
        logs = train_model(model, cifar_train, cifar_test, nn.CrossEntropyLoss(), optimizer, config)
        flops, latency = estimate_flops_and_latency(model, input_shape=(3, 32, 32), device=config['device'])
        corruption_results = {}
        for corruption, loader in corrupted_loaders.items():
            try:
                corrupted_acc = evaluate(model, loader, config['device'])
            except:
                corrupted_acc = None
            corruption_results[corruption] = corrupted_acc

        results.append([
            label,
            logs.get("train_acc", [-1])[-1],
            logs.get("train_loss", [-1])[-1],
            logs.get("val_acc", [-1])[-1],
            latency,
            flops,
            corruption_results.get("noise", None),
            corruption_results.get("rotation", None),
        ])

    # === Save results ===
    headers = ["Method", "Train Acc", "Train Loss", "Val Acc", "Latency", "FLOPs", "Noise Acc", "Rotation Acc"]
    with open(os.path.join(config['save_dir'], "summary_results.csv"), "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(results)

    print("\n" + tabulate(results, headers=headers, tablefmt="github"))


if __name__ == "__main__":
    main()