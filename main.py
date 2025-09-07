import os
import csv
import copy
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


# ------------------------------
# Helper functions (top-level, Windows-safe)
# ------------------------------

def _is_head_param(name: str) -> bool:
    name = name.lower()
    head_keys = ("head", "heads", "classifier", "regressor", "fc", "out_proj", "out", "final")
    return any(k in name for k in head_keys)


def freeze_except_head(model: nn.Module) -> None:
    """
    Freeze all parameters except those that are part of the heads.
    Explicit and conservative: turn off grads for all, then unfreeze heads.* entries.
    """
    # First freeze everything
    for _, p in model.named_parameters():
        p.requires_grad = False

    # Then explicitly unfreeze known head parameters
    found = False
    for name, p in model.named_parameters():
        # explicit moduledict 'heads.' prefix is primary target
        if name.startswith("heads.") or _is_head_param(name):
            p.requires_grad = True
            found = True

    if not found:
        # As a safety-net: if no head-like params found, leave classifier modules unfrozen by pattern
        for name, p in model.named_parameters():
            if ".head" in name or "classifier" in name or "regressor" in name:
                p.requires_grad = True


def unfreeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = True


def clone_and_update(base: dict, updates: dict) -> dict:
    new_cfg = copy.deepcopy(base)
    new_cfg.update(updates)
    return new_cfg


# ------------------------------
# Progressive fine-tune (head-only -> full)
# ------------------------------

def progressive_finetune(model: nn.Module,
                         train_loader,
                         test_loader,
                         base_config: dict,
                         dataset_name: str,
                         task_type: str):
    # Phase 1: head-only
    freeze_except_head(model)
    head_epochs = base_config.get("head_only_epochs",
                                  max(3, base_config.get('num_epochs', 5) // 4))
    head_lr = base_config.get("lr_head", base_config.get('learning_rate', 1e-3))

    head_cfg = clone_and_update(base_config, {
        "phase": "head_only",
        "learning_rate": head_lr,
        "num_epochs": head_epochs,
        "early_stop_patience": base_config.get('early_stop_patience', 5),
    })

    logs_head = train_model(model, train_loader, test_loader, head_cfg, dataset_name, task_type)

    # Phase 2: unfreeze all
    unfreeze_all(model)
    full_epochs = base_config.get('full_finetune_epochs', base_config.get('num_epochs'))
    full_lr = base_config.get('lr_finetune', base_config.get('learning_rate', 1e-3) * 0.5)

    full_cfg = clone_and_update(base_config, {
        "phase": "full_finetune",
        "learning_rate": full_lr,
        "num_epochs": full_epochs,
        "early_stop_patience": base_config.get('early_stop_patience', 7),
    })

    logs_full = train_model(model, train_loader, test_loader, full_cfg, dataset_name, task_type)

    merged_logs = {}
    if isinstance(logs_head, dict):
        merged_logs.update(logs_head)
    if isinstance(logs_full, dict):
        merged_logs.update(logs_full)
    return merged_logs


# ------------------------------
# Main orchestration
# ------------------------------

def main():
    os.makedirs(config['save_dir'], exist_ok=True)

    # Dataset map
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


    # --- Pretrain phase ---
    for dataset_name, meta in datasets_info.items():
        if meta.get('pretrain', False):
            print(f"\n[Phase 1] Pretraining on {dataset_name}")
            train_loader, test_loader = get_dataloaders(dataset_name, config['batch_size'])

            # Create model: choose image vs vector input handling
            is_img = dataset_name in ("CIFAR10", "CIFAR100", "SVHN", "MNIST", "FashionMNIST")
            if is_img:
                model = TransformerWithAutoNorm(is_cifar=True).to(config['device'])
            else:
                # compute input_dim for tabular/vector datasets and pass it in
                sample = train_loader.dataset[0][0]
                input_dim = sample.numel() if hasattr(sample, 'numel') else int(getattr(sample, 'shape', (1,))[0])
                model = TransformerWithAutoNorm(input_dim=input_dim, is_cifar=False).to(config['device'])

            pre_cfg = clone_and_update(config, {
                'phase': 'pretrain',
                'num_epochs': config.get('max_epochs_pretrain', config.get('num_epochs')),
                'learning_rate': config.get('pretrain_lr', config.get('learning_rate'))
            })

            logs = train_model(model, train_loader, test_loader, pre_cfg, dataset_name, meta['type'])
            ckpt_path = os.path.join(config['save_dir'], f"pretrained_{dataset_name}.pth")
            torch.save(model.state_dict(), ckpt_path)
            if meta['type'] == 'classification':
                plot_accuracy_loss(logs, prefix=f"{dataset_name}_Pretrain")
            else:
                plot_rmse_loss(logs, prefix=f"{dataset_name}_Pretrain")

    # --- Finetune phase (progressive) ---
    corrupted_loaders = get_corrupted_loaders()

    for dataset_name, meta in datasets_info.items():
        if meta.get('finetune', False):
            print(f"\n[Phase 2] Finetuning on {dataset_name} ({meta['type']})")
            train_loader, test_loader = get_dataloaders(dataset_name, config['batch_size'])

            # Pick pretrain checkpoint
            if meta['type'] == 'classification':
                pretrain_ckpt = os.path.join(config['save_dir'], 'pretrained_CIFAR100.pth')
            else:
                pretrain_ckpt = os.path.join(config['save_dir'], 'pretrained_CaliforniaHousing.pth')

            for model_name, model_fn in model_variants.items():
                print(f"\n[Model: {model_name} | Task: {dataset_name}]")

                # Instantiate model; prefer passing input_dim for non-image tasks
                is_img = dataset_name in ("CIFAR10", "CIFAR100", "SVHN", "MNIST", "FashionMNIST")
                try:
                    # If factory accepts kwargs (is_cifar), pass it
                    model = model_fn(is_cifar=is_img)
                except TypeError:
                    model = model_fn()
                    if not is_img:
                        # set input_dim for non-image models where possible
                        sample = train_loader.dataset[0][0]
                        input_dim = sample.numel() if hasattr(sample, 'numel') else int(getattr(sample, 'shape', (1,))[0])
                        try:
                            if hasattr(model, 'linear1') and isinstance(model.linear1, nn.LazyLinear):
                                # replace LazyLinear with a concrete Linear to avoid uninitialized params
                                model.linear1 = nn.Linear(input_dim, model.linear1.out_features)
                        except Exception:
                            pass
                    # try set flag if attribute exists
                    if hasattr(model, 'is_cifar'):
                        model.is_cifar = is_img

                model = model.to(config['device'])

                # Load pretrain weights (partial load if heads differ)
                load_pretrained_partial(model, pretrain_ckpt)

                # Progressive fine-tune
                finetune_cfg = clone_and_update(config, {
                    'phase': 'finetune',
                    'learning_rate': config.get('learning_rate'),
                    'num_epochs': config.get('num_epochs')
                })

                logs = progressive_finetune(model, train_loader, test_loader, finetune_cfg, dataset_name, meta['type'])

                prefix = f"{model_name}_{dataset_name}"
                if meta['type'] == 'classification':
                    plot_accuracy_loss(logs, prefix=prefix)
                else:
                    plot_rmse_loss(logs, prefix=prefix)

                # Estimate flops/latency: choose representative input shape
                sample_input = train_loader.dataset[0][0]
                if hasattr(sample_input, 'ndim') and sample_input.ndim == 3:
                    input_shape = tuple(sample_input.shape)
                else:
                    input_shape = (sample_input.numel(),)

                # Wrapper that calls model(x, task)
                class _Wrap(nn.Module):
                    def __init__(self, m, task):
                        super().__init__()
                        self.m = m
                        self.task = task

                    def forward(self, x, task=None):
                        # Forward call: some of your models may accept (x, task)
                        try:
                            return self.m(x, task or self.task)
                        except TypeError:
                            return self.m(x)


                wrapped_model = _Wrap(model, dataset_name)

                flops, latency = estimate_flops_and_latency(
                    wrapped_model,
                    dataset_name,  # <- goes in the 2nd position (task)
                    input_shape=input_shape,
                    device=config['device']
                )

                if meta['type'] == 'classification':
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
                        logs.get('train_acc', [-1])[-1],
                        logs.get('train_loss', [-1])[-1],
                        logs.get('val_acc', [-1])[-1],
                        latency, flops,
                        corruption_results.get('noise', None),
                        corruption_results.get('rotation', None)
                    ])
                else:
                    regression_results.append([
                        f"{model_name}-{dataset_name}",
                        logs.get('train_rmse', [-1])[-1],
                        logs.get('val_rmse', [-1])[-1],
                        logs.get('train_mae', [-1])[-1],
                        logs.get('val_mae', [-1])[-1],
                        logs.get('train_loss', [-1])[-1],
                        logs.get('val_loss', [-1])[-1],
                        latency, flops
                    ])

    # Save CSVs
    cls_headers = ["Method", "Train Acc", "Train Loss", "Val Acc", "Latency", "FLOPs", "Noise Acc", "Rotation Acc"]
    reg_headers = ["Method", "Train RMSE", "Val RMSE", "Train MAE", "Val MAE", "Train Loss", "Val Loss", "Latency", "FLOPs"]

    with open(os.path.join(config['save_dir'], "summary_results_classification.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(cls_headers)
        writer.writerows(classification_results)

    with open(os.path.join(config['save_dir'], "summary_results_regression.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(reg_headers)
        writer.writerows(regression_results)

    # Print tables
    print('\n[Classification Results]')
    print(tabulate(classification_results, headers=cls_headers, tablefmt='github'))

    print('\n[Regression Results]')
    print(tabulate(regression_results, headers=reg_headers, tablefmt='github'))


if __name__ == '__main__':
    main()
