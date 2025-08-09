import torch
import torch.nn as nn
from norm_selector import NormSelector

class DyT(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, dim))

    def forward(self, x):
        return x * self.alpha

class ResidualMLPBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
            nn.Dropout(0.1),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.mlp(x))

class TransformerWithAutoNorm(nn.Module):
    def __init__(self, input_dim=784, dim=128, depth=6, disable_selector=False, random_selector=False):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, dim)

        self.dyt = DyT(dim)
        self.ln = nn.LayerNorm(dim)
        self.norm_selector = NormSelector(
            hidden_dim=dim,
            disable_selector=False,
            random_selector=False
        )

        self.blocks = nn.Sequential(*[ResidualMLPBlock(dim) for _ in range(depth)])
        self.out = nn.Linear(dim, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.linear1(x)

        dyt_out = self.dyt(x)
        ln_out = self.ln(x)
        x = self.norm_selector(x.unsqueeze(1), ln_out.unsqueeze(1), dyt_out.unsqueeze(1)).squeeze(1)

        x = self.blocks(x)
        return self.out(x)

class TeacherTransformer(nn.Module):
    def __init__(self, input_dim=784, dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, 10)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)


class FrozenDyTTransformer(TransformerWithAutoNorm):
    def __init__(self, input_dim=784, dim=128):
        super().__init__(input_dim=input_dim, dim=dim)
        for p in self.ln.parameters():
            p.requires_grad = False


class FrozenLNTransformer(TransformerWithAutoNorm):
    def __init__(self, input_dim=784, dim=128):
        super().__init__(input_dim=input_dim, dim=dim)
        for p in self.dyt.parameters():
            p.requires_grad = False