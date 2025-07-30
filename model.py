import torch
import torch.nn as nn
from norm_selector import NormSelector


class DyT(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, dim))

    def forward(self, x):
        return x * self.alpha


class MLPBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * 2, dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        return self.dropout(self.fc2(self.act(self.fc1(x))))

class AttentionBlock(nn.Module):
    def __init__(self, dim, heads=4, return_attention=False):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.return_attention = return_attention
        self.attn_weights = None

    def forward(self, x):
        attn_out, attn_weights = self.attn(x, x, x, need_weights=True)
        if self.return_attention:
            self.attn_weights = attn_weights.detach()
        return self.norm(x + attn_out)

class TransformerWithAutoNorm(nn.Module):
    def __init__(self, input_dim=784, dim=192, depth=2, disable_selector=False, random_selector=False, use_gumbel=True, tau=0.5):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, dim)
        self.dyt = DyT(dim)
        self.ln = nn.LayerNorm(dim)
        self.norm_selector = NormSelector(dim, disable_selector, random_selector, use_gumbel, tau)

        self.blocks = nn.Sequential(
            *[nn.Sequential(AttentionBlock(dim), MLPBlock(dim)) for _ in range(depth)]
        )

        self.out = nn.Linear(dim, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        dyt_out = self.dyt(x)
        ln_out = self.ln(x)
        x = self.norm_selector(x.unsqueeze(1), ln_out.unsqueeze(1), dyt_out.unsqueeze(1))  # (B, 1, C)
        x = self.blocks(x)
        return self.out(x.squeeze(1))


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