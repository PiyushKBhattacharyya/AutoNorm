import os
import csv
import torch
import torch.nn as nn
from tabulate import tabulate

from model import TransformerWithAutoNorm
from train import train_model
from utils import (
    plot_accuracy_loss, plot_rmse_loss,
    estimate_flops_and_latency, evaluate, evaluate_regression,
    get_corrupted_loaders, load_pretrained_partial
)
from data import get_dataloaders
from configs import config
from factory import model_variants

# Wrapper class to provide fixed 'task' during profiling
class ModelWithTaskWrapper(nn.Module):
    def __init__(self, model, task):
        super().__init__()
        self.model = model
        self.task = task

    def forward(self, *args, **kwargs):
        # Ignore passed task argument, always use self.task internally
        # Assume first arg is input tensor
        x = args[0]
        # Forward with x and self.task
        return self.model(x, self.task)

def main():
    os.makedirs(config['save_dir'], exist_ok=True)

    # --- Dataset Info ---
    datasets_info = {
        "MNIST": {"type": "classification", "finetune": True},
        "CIFAR10": {"type": "classification", "finetune": True},
        "FashionMNIST": {"type": "classification", "finetune": True},
        "CIFAR100": {"type": "classification", "pretrain": True},
        "SVHN": {"type": "classification", "finetune": True},
        "CaliforniaHousing": {"type": "regression", "pretrain": True},
        "EnergyEfficiency": {"type": "regression", "finetune": True}
    }

    classification_results = []
    regression_results = []

    # --- Pretrain Phase ---
    for dataset_name, meta in datasets_info.items():
        if meta.get("pretrain", False):
            print(f"\n[Phase 1] Pretraining on {dataset_name}")
            train_loader, test_loader = get_dataloaders(dataset_name, config['batch_size'])
            model = TransformerWithAutoNorm(input_dim=train_loader.dataset[0][0].numel()).to(config['device'])
            logs = train_model(model, train_loader, test_loader, config, dataset_name, meta["type"])
            ckpt_path = os.path.join(config['save_dir'], f"pretrained_{dataset_name}.pth")
            torch.save(model.state_dict(), ckpt_path)
            if meta["type"] == "classification":
                plot_accuracy_loss(logs, prefix=f"{dataset_name}_Pretrain")
            else:
                plot_rmse_loss(logs, prefix=f"{dataset_name}_Pretrain")

    # --- Finetune Phase ---
    corrupted_loaders = get_corrupted_loaders()

    for dataset_name, meta in datasets_info.items():
        if meta.get("finetune", False):
            print(f"\n[Phase 2] Finetuning on {dataset_name} ({meta['type']})")
            train_loader, test_loader = get_dataloaders(dataset_name, config['batch_size'])

            # Pick pretrain checkpoint based on type
            if meta["type"] == "classification":
                pretrain_ckpt = os.path.join(config['save_dir'], "pretrained_CIFAR100.pth")
            else:
                pretrain_ckpt = os.path.join(config['save_dir'], "pretrained_CaliforniaHousing.pth")

            for model_name, model_fn in model_variants.items():
                print(f"\n[Model: {model_name} | Task: {dataset_name}]")
                model = model_fn().to(config['device'])
                load_pretrained_partial(model, pretrain_ckpt)

                logs = train_model(model, train_loader, test_loader, config, dataset_name, meta["type"])

                prefix = f"{model_name}_{dataset_name}"
                if meta["type"] == "classification":
                    plot_accuracy_loss(logs, prefix=prefix)
                else:
                    plot_rmse_loss(logs, prefix=prefix)

                # Wrap model to provide 'task' for flops/latency estimation
                wrapped_model = ModelWithTaskWrapper(model, task=dataset_name)

                flops, latency = estimate_flops_and_latency(
                    wrapped_model,
                    input_shape=(train_loader.dataset[0][0].numel(),),
                    device=config['device'],
                    task=dataset_name
                )

                if meta["type"] == "classification":
                    corruption_results = {}
                    for corruption, loader in corrupted_loaders.items():
                        try:
                            corrupted_acc = evaluate(model, loader, config['device'], dataset_name)
                        except Exception as e:
                            corrupted_acc = None
                            print(f"[WARN] {corruption} eval failed: {e}")
                        corruption_results[corruption] = corrupted_acc

                    classification_results.append([
                        f"{model_name}-{dataset_name}",
                        logs.get("train_acc", [-1])[-1],
                        logs.get("train_loss", [-1])[-1],
                        logs.get("val_acc", [-1])[-1],
                        latency, flops,
                        corruption_results.get("noise", None),
                        corruption_results.get("rotation", None)
                    ])

                else:  # regression
                    val_rmse = logs.get("val_rmse", [-1])[-1]
                    val_mae = logs.get("val_mae", [-1])[-1]
                    regression_results.append([
                        f"{model_name}-{dataset_name}",
                        logs.get("train_rmse", [-1])[-1],
                        val_rmse,
                        logs.get("train_mae", [-1])[-1],
                        val_mae,
                        logs.get("train_loss", [-1])[-1],
                        logs.get("val_loss", [-1])[-1],
                        latency, flops
                    ])

    # --- Save CSVs ---
    cls_headers = ["Method", "Train Acc", "Train Loss", "Val Acc", "Latency", "FLOPs", "Noise Acc", "Rotation Acc"]
    reg_headers = ["Method", "Train RMSE", "Val RMSE", "Train MAE", "Val MAE", "Train Loss", "Val Loss", "Latency", "FLOPs"]

    with open(os.path.join(config['save_dir'], "summary_results_classification.csv"), "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(cls_headers)
        writer.writerows(classification_results)

    with open(os.path.join(config['save_dir'], "summary_results_regression.csv"), "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(reg_headers)
        writer.writerows(regression_results)

    # --- Print Tables ---
    print("\n[Classification Results]")
    print(tabulate(classification_results, headers=cls_headers, tablefmt="github"))

    print("\n[Regression Results]")
    print(tabulate(regression_results, headers=reg_headers, tablefmt="github"))


if __name__ == "__main__":
    main()