import torch
import torch.nn as nn
import torch.nn.functional as F
from norm_selector import NormSelector


# ------------------------------
# Utility Modules (Windows-safe: no lambdas/locals)
# ------------------------------
class DropPath(nn.Module):
    """Stochastic Depth (per sample)."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor)
        return x.div(keep_prob) * random_tensor


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=4, emb_dim=128):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # (B, C, H, W) -> (B, D, H', W')
        return self.proj(x)


class DyT(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, dim))

    def forward(self, x):
        return x * self.alpha


class SE(nn.Module):
    """Squeeze-and-Excitation on channel dimension for token features."""
    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        hidden = max(8, dim // reduction)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        # x: (B, N, D)
        s = x.mean(dim=1)  # (B, D)
        s = F.gelu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))  # (B, D)
        return x * s.unsqueeze(1)


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim, dropout=0.1, drop_path=0.0, layerscale_init=1e-4, use_se=True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path)
        self.use_se = use_se
        self.se = SE(dim) if use_se else nn.Identity()
        self.gamma = nn.Parameter(layerscale_init * torch.ones(dim))  # LayerScale

    def forward(self, x):
        # x: (B, N, D)
        residual = x
        y = self.mlp(x)
        y = self.se(y)
        y = y * self.gamma  # broadcast over tokens
        y = self.drop_path(y)
        out = self.norm(residual + y)
        return out


class TransformerWithAutoNorm(nn.Module):
    def __init__(self,
                 input_dim=784,
                 dim=128,
                 depth=12,
                 disable_selector=False,
                 random_selector=False,
                 is_cifar=False,
                 patch_size=4,
                 drop_path_rate=0.1,
                 dropout=0.1):
        super().__init__()
        self.is_cifar = is_cifar
        self.dim = dim
        self.patch_size = patch_size

        # Always create an image patch embed / pos_embed / cls_token so forward() won't crash
        # Use in_channels=3; single-channel inputs (MNIST) are expanded in _image_forward.
        self.patch_embed = PatchEmbedding(in_channels=3, patch_size=patch_size, emb_dim=dim)
        # default positional map for 32x32 with patch_size -> 8x8 grid; will be interpolated if needed
        self.pos_embed_2d = nn.Parameter(torch.randn(1, dim, 8, 8) * 0.02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # keep a LazyLinear for non-image inputs
        self.linear1 = nn.LazyLinear(dim)

        self.dyt = DyT(dim)
        self.ln = nn.LayerNorm(dim)
        self.norm_selector = NormSelector(
            hidden_dim=dim,
            disable_selector=disable_selector,
            random_selector=random_selector
        )

        # Stochastic depth schedule
        dprates = [drop_path_rate * i / float(max(1, depth - 1)) for i in range(depth)] if depth > 1 else [0.0]

        self.blocks = nn.ModuleList([
            ResidualMLPBlock(dim, dropout=dropout if is_cifar else 0.3, drop_path=dprates[i])
            for i in range(depth)
        ])

        self.heads = nn.ModuleDict({
            "MNIST": nn.Linear(dim, 10),
            "CIFAR10": nn.Linear(dim, 10),
            "FashionMNIST": nn.Linear(dim, 10),
            "CIFAR100": nn.Linear(dim, 100),
            "SVHN": nn.Linear(dim, 10),
            "CaliforniaHousing": nn.Linear(dim, 1),
            "EnergyEfficiency": nn.Linear(dim, 2)
        })

    def _image_forward(self, x):
        # x: (B, C, H, W)
        # If grayscale (C==1), expand to 3 channels so patch_embed (in_ch=3) works.
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        x = self.patch_embed(x)  # (B, D, H', W')
        B, D, Hp, Wp = x.shape

        pos = self.pos_embed_2d
        if pos.shape[2] != Hp or pos.shape[3] != Wp:
            pos = F.interpolate(pos, size=(Hp, Wp), mode='bilinear', align_corners=False)

        x = x + pos  # broadcast add: (1, D, Hp, Wp) with (B, D, Hp, Wp)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)

        if self.cls_token is not None:
            cls = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)  # (B, 1+N, D)
        return x

    def _vector_forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        return x.unsqueeze(1)

    def forward(self, x, task):
        is_image_input = (x.ndim == 4 and x.size(1) in (1, 3))
        if self.is_cifar or is_image_input:
            x = self._image_forward(x)
        else:
            x = self._vector_forward(x)

        dyt_out = self.dyt(x)
        ln_out = self.ln(x)
        x = self.norm_selector(x, ln_out, dyt_out)

        for blk in self.blocks:
            x = blk(x)

        if self.is_cifar or is_image_input:
            # if cls_token exists we use it; otherwise fallback to mean pooling
            x = x[:, 0, :] if self.cls_token is not None else x.mean(dim=1)
        else:
            x = x.squeeze(1)

        if task not in self.heads:
            raise ValueError(f"Task '{task}' not found in model heads. Available: {list(self.heads.keys())}")
        return self.heads[task](x)
class TeacherTransformer(nn.Module):
    def __init__(self, input_dim=784, dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.LazyLinear(dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, 10)
        )

    def forward(self, x, task=None):
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