import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
import pandas as pd

def train_model(model, train_loader, test_loader, criterion, optimizer, config, distill_teacher=None):
    logs = {
        'train_loss': [],
        'train_acc': [],
        'val_acc': [],
        'val_loss': [],
        'grad_norms': [],
        'norm_weights': []
    }

    model.to(config['device'])

    #  Freeze selector if ablation flag is set
    if config.get('freeze_selector', False):
        print(">> Selector frozen during training.")
        if hasattr(model, 'selector'):
            for param in model.selector.parameters():
                param.requires_grad = False
        else:
            print(" Warning: model has no attribute 'selector' to freeze.")

    for epoch in range(config['num_epochs']):
        model.train()
        epoch_loss = 0
        train_correct = 0
        total = 0

        for data, target in train_loader:
            data, target = data.to(config['device']), target.to(config['device'])
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            if distill_teacher:
                with torch.no_grad():
                    teacher_out = distill_teacher(data)
                distill_loss = nn.KLDivLoss()(F.log_softmax(output, dim=1), F.softmax(teacher_out, dim=1))
                loss = (1 - config['distill_lambda']) * loss + config['distill_lambda'] * distill_loss

            loss.backward()
            logs['grad_norms'].append(compute_grad_norms(model))
            optimizer.step()

            epoch_loss += loss.item()

            preds = output.argmax(dim=1)
            train_correct += preds.eq(target).sum().item()
            total += target.size(0)

        logs['train_loss'].append(epoch_loss / len(train_loader))
        logs['train_acc'].append(train_correct / total)

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(config['device']), target.to(config['device'])
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                pred = output.argmax(dim=1)
                val_correct += pred.eq(target).sum().item()
                val_total += target.size(0)

        logs['val_loss'].append(val_loss / len(test_loader))
        logs['val_acc'].append(val_correct / val_total)

        # Log dynamic norm weights if applicable
        if hasattr(model, "show_norm_weights"):
            logs['norm_weights'].append(model.show_norm_weights())

    #  Save logs to CSV if enabled
    if config.get('log_csv', False):
        tag = config.get("experiment_tag", "default_run")
        df = pd.DataFrame({
            "train_loss": logs['train_loss'],
            "train_acc": logs['train_acc'],
            "val_loss": logs['val_loss'],
            "val_acc": logs['val_acc']
        })
        df.to_csv(f"logs/{tag}_metrics.csv", index=False)
        print(f">> Saved logs to logs/{tag}_metrics.csv")

    return logs