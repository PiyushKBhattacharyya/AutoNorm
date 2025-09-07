import math
import random
import copy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from contextlib import contextmanager

from utils import *  # ensure utils.py does NOT import train.py to avoid circular imports


# ------------------------------
# Helper: Exponential Moving Average of weights (EMA)
# ------------------------------
class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999, device=None):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.device = device
        if device is not None:
            self.ema.to(device)

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if k in msd:
                v.copy_(v * d + msd[k] * (1.0 - d))


# ------------------------------
# Helper: MixUp / CutMix (classification only)
# ------------------------------

def _one_hot(labels: torch.Tensor, num_classes: int, smoothing: float = 0.0):
    with torch.no_grad():
        y = torch.zeros((labels.size(0), num_classes), device=labels.device)
        y.scatter_(1, labels.unsqueeze(1), 1.0)
        if smoothing > 0.0:
            y = y * (1.0 - smoothing) + smoothing / num_classes
    return y


def mixup(inputs, targets, num_classes, alpha=0.2, smoothing=0.0):
    if alpha <= 0.0:
        return inputs, _one_hot(targets, num_classes, smoothing), 1.0
    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    index = torch.randperm(inputs.size(0), device=inputs.device)
    mixed_x = lam * inputs + (1 - lam) * inputs[index, :]
    targets1 = _one_hot(targets, num_classes, smoothing)
    targets2 = _one_hot(targets[index], num_classes, smoothing)
    mixed_y = lam * targets1 + (1 - lam) * targets2
    return mixed_x, mixed_y, lam


def rand_bbox(W, H, lam):
    cut_rat = math.sqrt(max(0.0, 1.0 - lam))
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = random.randint(0, max(0, W - 1))
    cy = random.randint(0, max(0, H - 1))
    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y2 = min(cy + cut_h // 2, H)
    return x1, y1, x2, y2


def cutmix(inputs, targets, num_classes, alpha=1.0, smoothing=0.0):
    if alpha <= 0.0:
        return inputs, _one_hot(targets, num_classes, smoothing), 1.0
    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    B, C, H, W = inputs.size()
    index = torch.randperm(B, device=inputs.device)
    x1, y1, x2, y2 = rand_bbox(W, H, lam)
    inputs_clone = inputs.clone()
    # guard against degenerate boxes
    if x2 > x1 and y2 > y1:
        inputs_clone[:, :, y1:y2, x1:x2] = inputs[index, :, y1:y2, x1:x2]
    lam_adjusted = 1.0 - ((x2 - x1) * (y2 - y1) / (W * H)) if (x2 > x1 and y2 > y1) else 1.0
    targets1 = _one_hot(targets, num_classes, smoothing)
    targets2 = _one_hot(targets[index], num_classes, smoothing)
    mixed_y = lam_adjusted * targets1 + (1 - lam_adjusted) * targets2
    return inputs_clone, mixed_y, lam_adjusted


# ------------------------------
# Helper: simple in-loop CIFAR aug (flip + small shifts)
# ------------------------------

def augment_cifar_batch(x):
    if random.random() < 0.5:
        x = torch.flip(x, dims=[3])
    if random.random() < 0.5:
        v = random.randint(-2, 2)
        x = torch.roll(x, shifts=v, dims=2)
        v = random.randint(-2, 2)
        x = torch.roll(x, shifts=v, dims=3)
    return x


# ------------------------------
# Helper: Test-Time Augmentation (TTA) for classification
# ------------------------------
@torch.no_grad()
def tta_logits(model, x, task, device):
    x = x.to(device)
    outs = []
    outs.append(model(x, task))
    outs.append(model(torch.flip(x, dims=[3]), task))
    outs.append(model(torch.roll(x, 1, dims=2), task))
    outs.append(model(torch.roll(x, -1, dims=3), task))
    return torch.stack(outs, dim=0).mean(0)


# ------------------------------
# Dummy autocast (no-op) for CPU runs
# ------------------------------
@contextmanager
def dummy_autocast(enabled: bool = True):
    yield


# ------------------------------
# Main training function (AMP-free CPU-friendly)
# ------------------------------

def train_model(model, train_loader, test_loader, config, task_name, task_type, distill_teacher=None):
    logs = {'train_loss': [], 'val_loss': []}

    # Flags & defaults
    is_cifar = task_name in ["CIFAR10", "CIFAR100"]
    num_classes = 100 if task_name == "CIFAR100" else 10

    label_smoothing = float(config.get('label_smoothing', 0.1 if not is_cifar else 0.2))
    mixup_alpha = float(config.get('mixup_alpha', 0.2 if is_cifar else 0.0))
    cutmix_alpha = float(config.get('cutmix_alpha', 1.0 if is_cifar else 0.0))
    mix_prob = float(config.get('mix_prob', 0.7 if is_cifar else 0.0))
    use_ema = bool(config.get('ema', True if is_cifar else False))
    ema_decay = float(config.get('ema_decay', 0.9999))
    use_tta = bool(config.get('tta', True if is_cifar else False))

    # Criterion
    if task_type == "classification":
        logs.update({'train_acc': [], 'val_acc': []})
        base_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    else:
        logs.update({'train_rmse': [], 'val_rmse': [], 'train_mae': [], 'val_mae': []})
        base_criterion = nn.MSELoss()

    device = config['device']
    model.to(device)

    # ------------------------
    # Initialize lazy modules immediately (before creating optimizer / EMA)
    # ------------------------
    try:
        model.eval()
        with torch.no_grad():
            # try to get a sample from train_loader to initialize Lazy modules
            if hasattr(train_loader, '__iter__'):
                for sample_x, _ in train_loader:
                    sample_x = sample_x.to(device)
                    # run one small batch forward to initialize lazy params
                    _ = model(sample_x[:1], task_name)
                    break
            else:
                # fallback if train_loader isn't iterable
                sample = train_loader.dataset[0][0]
                if hasattr(sample, 'shape'):
                    dummy = torch.randn(1, *sample.shape, device=device)
                else:
                    dummy = torch.randn(1, sample.numel(), device=device)
                _ = model(dummy, task_name)
    except Exception as e:
        # don't crash here; warn and continue (initialization might still fail later)
        print(f"[WARN] lazy-init forward failed: {e}")
    finally:
        model.train()

    # Now collect parameters for optimizer
    params = [p for p in model.parameters() if p.requires_grad]

    # If params is empty (common when freeze_except_head accidentally froze everything),
    # attempt to unfreeze heads explicitly and recompute params.
    if len(params) == 0:
        print("[WARN] No trainable params found — unfreezing head parameters explicitly.")
        # explicit attempt to find heads.* parameters and unfreeze them
        found = False
        for name, p in model.named_parameters():
            if name.startswith("heads.") or ".head" in name or "classifier" in name or "regressor" in name:
                p.requires_grad = True
                found = True
        params = [p for p in model.parameters() if p.requires_grad]
        if not found or len(params) == 0:
            # as a last resort, unfreeze all parameters to avoid optimizer error
            print("[WARN] No head parameters found — unfreezing all parameters to proceed.")
            for p in model.parameters():
                p.requires_grad = True
            params = [p for p in model.parameters() if p.requires_grad]

    # Create optimizer now that params are known
    optimizer = torch.optim.AdamW(
        params,
        lr=config['learning_rate'],
        weight_decay=5e-4 if is_cifar else 1e-4
    )

    # Cosine LR with warmup (per-step)
    total_steps = max(1, config['num_epochs'] * len(train_loader))
    warmup_steps = int(float(config.get('warmup_ratio', 0.05)) * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Prepare EMA AFTER lazy init and optimizer creation
    if use_ema:
        try:
            ema = ModelEMA(model, decay=ema_decay, device=device)
        except Exception as e:
            print(f"[WARN] EMA creation failed (uninitialized params?). Continuing without EMA: {e}")
            ema = None
    else:
        ema = None

    # (rest of the training loop follows exactly as before)
    best_val_loss = float('inf')
    best_val_metric = 0.0 if task_type == "classification" else float('inf')
    patience_counter = 0
    best_model_state = None
    global_step = 0

    for epoch in range(config['num_epochs']):
        model.train()
        epoch_loss, train_metric_sum, train_mae_sum, total = 0.0, 0.0, 0.0, 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            if is_cifar and data.ndim == 4:
                data = augment_cifar_batch(data)

            optimizer.zero_grad(set_to_none=True)

            if task_type == "classification" and data.ndim == 4:
                use_mix = random.random() < mix_prob and (mixup_alpha > 0 or cutmix_alpha > 0)
                if use_mix:
                    if cutmix_alpha > 0 and random.random() < 0.5:
                        inputs, targets_soft, _ = cutmix(data, target, num_classes, alpha=cutmix_alpha, smoothing=label_smoothing)
                    else:
                        inputs, targets_soft, _ = mixup(data, target, num_classes, alpha=mixup_alpha, smoothing=label_smoothing)
                else:
                    inputs = data
                    targets_soft = None
            else:
                inputs = data
                targets_soft = None

            with dummy_autocast(enabled=False):
                outputs = model(inputs, task_name)
                if task_type == "classification":
                    if targets_soft is not None:
                        loss = (-targets_soft * F.log_softmax(outputs, dim=1)).sum(dim=1).mean()
                    else:
                        loss = base_criterion(outputs, target.long())
                else:
                    loss = base_criterion(outputs, target)

                if distill_teacher is not None and task_type == "classification":
                    with torch.no_grad():
                        t_out = distill_teacher(inputs, task_name)
                    kd = F.kl_div(F.log_softmax(outputs, dim=1), F.softmax(t_out, dim=1), reduction='batchmean')
                    lam = float(config.get('distill_lambda', 0.2))
                    loss = (1 - lam) * loss + lam * kd

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            if ema is not None:
                try:
                    ema.update(model)
                except Exception as e:
                    print(f"[WARN] EMA update failed: {e}")

            epoch_loss += loss.item() * inputs.size(0)
            total += inputs.size(0)

            if task_type == "classification":
                with torch.no_grad():
                    preds = outputs.argmax(dim=1)
                    train_metric_sum += preds.eq(target).sum().item()
            else:
                with torch.no_grad():
                    preds = outputs
                    rmse = torch.sqrt(torch.mean((preds - target) ** 2)).item()
                    mae = torch.mean(torch.abs(preds - target)).item()
                    train_metric_sum += rmse
                    train_mae_sum += mae * inputs.size(0)

            global_step += 1

        logs['train_loss'].append(epoch_loss / max(1, total))
        if task_type == "classification":
            logs['train_acc'].append(train_metric_sum / max(1, total))
        else:
            logs['train_rmse'].append(train_metric_sum / max(1, len(train_loader)))
            logs['train_mae'].append(train_mae_sum / max(1, total))

        # Validation (EMA + optional TTA)
        model.eval()
        eval_model = ema.ema if (ema is not None) else model

        with torch.no_grad():
            if task_type == "classification":
                if use_tta and is_cifar:
                    correct, tot = 0, 0
                    val_losses = []
                    for x, y in test_loader:
                        logits = tta_logits(eval_model, x, task_name, device)
                        loss = base_criterion(logits, y.to(device).long()).item()
                        val_losses.append(loss)
                        preds = logits.argmax(dim=1)
                        correct += (preds.cpu() == y).sum().item()
                        tot += y.size(0)
                    val_acc = correct / max(1, tot)
                    val_loss = sum(val_losses) / max(1, len(val_losses))
                else:
                    val_acc = evaluate(eval_model, test_loader, device, task_name)
                    val_loss = sum(base_criterion(eval_model(x.to(device), task_name), y.to(device).long()).item()
                                   for x, y in test_loader) / len(test_loader)
                logs['val_acc'].append(val_acc)
            else:
                val_rmse, val_mae = evaluate_regression(eval_model, test_loader, device, task_name)
                logs['val_rmse'].append(val_rmse)
                logs['val_mae'].append(val_mae)
                val_loss = sum(base_criterion(eval_model(x.to(device), task_name), y.to(device)).item()
                               for x, y in test_loader) / len(test_loader)

        logs['val_loss'].append(val_loss)

        def fmt(v):
            try:
                return f"{float(v):.4f}"
            except (TypeError, ValueError):
                return str(v)

        print(f"Epoch {epoch+1}: Train Loss={fmt(logs['train_loss'][-1])}, "
              f"Val Loss={fmt(val_loss)}, "
              f"{'Train Acc' if task_type=='classification' else 'Train RMSE'}="
              f"{fmt(logs['train_acc'][-1] if task_type=='classification' else logs['train_rmse'][-1])}, "
              f"{'Val Acc' if task_type=='classification' else 'Val RMSE'}="
              f"{fmt(logs['val_acc'][-1] if task_type=='classification' else logs['val_rmse'][-1])}")

        # Early stopping
        if task_type == "classification":
            metric_improved = logs['val_acc'][-1] > best_val_metric
            if metric_improved:
                best_val_metric = logs['val_acc'][-1]
        else:
            metric_improved = logs['val_rmse'][-1] < best_val_metric
            if metric_improved:
                best_val_metric = logs['val_rmse'][-1]

        if val_loss < best_val_loss or metric_improved:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(eval_model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.get('early_stop_patience', 10):
                print(f">> Early stopping triggered at epoch {epoch + 1}")
                break

    # Load best weights (respect EMA)
    if best_model_state is not None:
        if ema is not None:
            try:
                ema.ema.load_state_dict(best_model_state)
                model.load_state_dict(ema.ema.state_dict(), strict=False)
            except Exception as e:
                print(f"[WARN] Loading EMA weights failed: {e}")
                model.load_state_dict(best_model_state)
        else:
            model.load_state_dict(best_model_state)

    # Align log lengths
    min_len = min(len(v) for v in logs.values())
    for k in logs:
        logs[k] = logs[k][:min_len]

    # Save logs
    if config.get('log_csv', False):
        tag = config.get("experiment_tag", f"{task_name}_{task_type}")
        df = pd.DataFrame(logs)
        os.makedirs('logs', exist_ok=True)
        df.to_csv(f"logs/{tag}_metrics.csv", index=False)
        print(f">> Saved logs to logs/{tag}_metrics.csv")

    return logs
