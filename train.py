import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import compute_grad_norms, evaluate, evaluate_regression
import pandas as pd
import copy

def train_model(model, train_loader, test_loader, config, task_name, task_type, distill_teacher=None):
    logs = {
        'train_loss': [],
        'val_loss': []
    }

    # Setup criterion with label smoothing for classification
    if task_type == "classification":
        logs.update({'train_acc': [], 'val_acc': []})
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    else:  # regression
        logs.update({'train_rmse': [], 'val_rmse': [], 'train_mae': [], 'val_mae': []})
        criterion = nn.MSELoss()

    model.to(config['device'])
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config['learning_rate'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max' if task_type=='classification' else 'min', patience=3, factor=0.5)

    best_val_loss = float('inf')
    best_val_metric = 0.0 if task_type == "classification" else float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(config['num_epochs']):
        model.train()
        epoch_loss = 0
        train_metric_sum = 0
        train_mae_sum = 0
        total = 0

        for data, target in train_loader:
            data, target = data.to(config['device']), target.to(config['device'])
            optimizer.zero_grad()

            output = model(data, task_name)
            loss = criterion(output, target if task_type == "regression" else target.long())

            # Distillation (optional)
            if distill_teacher:
                with torch.no_grad():
                    teacher_out = distill_teacher(data, task_name)
                distill_loss = nn.KLDivLoss()(
                    F.log_softmax(output, dim=1),
                    F.softmax(teacher_out, dim=1)
                )
                loss = (1 - config['distill_lambda']) * loss + config['distill_lambda'] * distill_loss

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_loss += loss.item() * data.size(0)
            total += data.size(0)

            if task_type == "classification":
                preds = output.argmax(dim=1)
                train_metric_sum += preds.eq(target).sum().item()
            else:
                preds = output.detach()
                rmse = torch.sqrt(torch.mean((preds - target) ** 2)).item()
                mae = torch.mean(torch.abs(preds - target)).item()
                train_metric_sum += rmse
                train_mae_sum += mae * data.size(0)

        logs['train_loss'].append(epoch_loss / total)

        if task_type == "classification":
            logs['train_acc'].append(train_metric_sum / total)
        else:
            avg_rmse = train_metric_sum / len(train_loader)
            avg_mae = train_mae_sum / total
            logs['train_rmse'].append(avg_rmse)
            logs['train_mae'].append(avg_mae)

        model.eval()
        with torch.no_grad():
            if task_type == "classification":
                val_acc = evaluate(model, test_loader, config['device'], task_name)
                logs['val_acc'].append(val_acc)
                val_loss = sum(criterion(model(x.to(config['device']), task_name), y.to(config['device']).long()).item()
                               for x, y in test_loader) / len(test_loader)
            else:
                val_rmse, val_mae = evaluate_regression(model, test_loader, config['device'], task_name)
                logs['val_rmse'].append(val_rmse)
                logs['val_mae'].append(val_mae)
                val_loss = sum(criterion(model(x.to(config['device']), task_name), y.to(config['device'])).item()
                               for x, y in test_loader) / len(test_loader)

        logs['val_loss'].append(val_loss)

        print(f"Epoch {epoch+1}: "
              f"Train Loss={logs['train_loss'][-1]:.4f}, "
              f"Val Loss={val_loss:.4f}, "
              f"{'Train Acc' if task_type=='classification' else 'Train RMSE'}="
              f"{logs['train_acc'][-1] if task_type=='classification' else logs['train_rmse'][-1]:.4f}, "
              f"{'Val Acc' if task_type=='classification' else 'Val RMSE'}="
              f"{logs['val_acc'][-1] if task_type=='classification' else logs['val_rmse'][-1]:.4f}")

        # Early stopping logic
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
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.get('early_stop_patience', 10):
                print(f">> Early stopping triggered at epoch {epoch + 1}")
                break

        # Step scheduler based on validation metric
        if task_type == "classification":
            scheduler.step(logs['val_acc'][-1])
        else:
            scheduler.step(logs['val_loss'][-1])

    if best_model_state:
        model.load_state_dict(best_model_state)

    # Fix for mismatched log lengths before saving CSV:
    min_len = min(len(v) for v in logs.values())
    for k in logs:
        logs[k] = logs[k][:min_len]

    # Save logs
    if config.get('log_csv', False):
        tag = config.get("experiment_tag", f"{task_name}_{task_type}")
        df = pd.DataFrame(logs)
        df.to_csv(f"logs/{tag}_metrics.csv", index=False)
        print(f">> Saved logs to logs/{tag}_metrics.csv")

    return logs