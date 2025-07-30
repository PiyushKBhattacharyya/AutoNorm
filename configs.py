import torch

config = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "batch_size": 64,
    "learning_rate": 1e-3,
    "num_epochs": 1,
    "distill_lambda": 0.3,
    "save_dir": "results",
    
    # Ablation Flags
    'freeze_selector': False,           # Freeze norm selector during finetuning
    'disable_selector': False,          # Use constant (equal) weights instead of learning
    'random_selector': False,           # Use random weights every forward pass
    'remove_regression_head': False,    # Disable regression head (for multitask setup)
    'attention_plot_epoch': 1,
    # Logging & Debug
    'log_csv': True,
}