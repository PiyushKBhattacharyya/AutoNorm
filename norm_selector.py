import torch
import torch.nn as nn
import torch.nn.functional as F


class NormSelector(nn.Module):
    def __init__(self, hidden_dim, disable_selector=False, random_selector=False, use_gumbel=False, tau=1.0):
        super().__init__()
        self.disable_selector = disable_selector
        self.random_selector = random_selector
        self.use_gumbel = use_gumbel
        self.tau = tau  # Gumbel temperature

        # Shared MLP gate for both modes
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)  # two choices: DyT and LN
        )

    def forward(self, x, ln_out, dyt_out):
        B, C, _ = x.shape  # x shape: (B, C, 1) after unsqueeze

        if self.disable_selector:
            # Use LayerNorm only
            return ln_out
        elif self.random_selector:
            # Random alpha per sample
            alpha = torch.rand(B, 1, 1, device=x.device)
            return alpha * dyt_out + (1 - alpha) * ln_out

        # Learnable selection
        pooled = self.pool(x.transpose(1, 2))  # (B, C, 1)
        logits = self.mlp(pooled)  # (B, 2)

        if self.use_gumbel:
            weights = F.gumbel_softmax(logits, tau=self.tau, hard=True)  # (B, 2)
        else:
            weights = F.softmax(logits, dim=-1)

        # weights[:, 0] → DyT weight, weights[:, 1] → LN weight
        dyt_weight = weights[:, 0].unsqueeze(1).unsqueeze(2)
        ln_weight = weights[:, 1].unsqueeze(1).unsqueeze(2)

        return dyt_weight * dyt_out + ln_weight * ln_out