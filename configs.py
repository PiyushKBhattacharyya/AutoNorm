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
    'disable_selector': False,
    'random_selector': False,
    'log_csv': True,
    'max_epochs_finetune' : 75,
    'max_epochs_pretrain' : 125,
    "label_smoothing": 0.2,
    "mixup_alpha": 0.2,
    "cutmix_alpha": 1.0,
    "mix_prob": 0.7,
    "ema": True, "ema_decay": 0.9999,
    "tta": True,
    "warmup_ratio": 0.05,
}