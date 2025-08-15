import torch

config = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "batch_size": 64,
    "learning_rate": 1e-3,
    "num_epochs": 200,
    "distill_lambda": 0.3,
    "save_dir": "results",
    'use_amp': True,
    'early_stop_patience': 5,
    # Ablation Flags
    'disable_selector': False,          # Use constant (equal) weights instead of learning
    'random_selector': False,           # Use random weights every forward pass
    # Logging & Debug
    'log_csv': True,
    'max_epochs_finetune' : 70,
    'max_epochs_pretrain' : 150
}