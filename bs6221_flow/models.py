"""Frozen definitions exported with the analysis model weights."""
import math
import torch
from torch import nn
from torch.nn import functional as F

class TimeEmbedding(nn.Module):

    def __init__(self, dim=64):
        super().__init__()
        self.register_buffer('frequencies', torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), dim // 2)))
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t):
        angles = t[:, None] * self.frequencies[None, :]
        return self.mlp(torch.cat([angles.sin(), angles.cos()], dim=1))

class TimeBlock(nn.Module):

    def __init__(self, cin, cout, tdim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.norm1 = nn.GroupNorm(4, cout)
        self.norm2 = nn.GroupNorm(4, cout)
        self.time = nn.Linear(tdim, cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, embedding):
        h = self.conv1(x)
        h = F.silu(self.norm1(h) + self.time(embedding)[:, :, None, None])
        h = F.silu(self.norm2(self.conv2(h)))
        return h + self.skip(x)

class ConditionalTinyUNet(nn.Module):

    def __init__(self, channels=24):
        super().__init__()
        c = channels
        self.embedding = TimeEmbedding()
        self.label_embedding = nn.Embedding(10, 64)
        self.encoder1 = TimeBlock(1, c)
        self.encoder2 = TimeBlock(c, 2 * c)
        self.middle = TimeBlock(2 * c, 4 * c)
        self.decoder2 = TimeBlock(6 * c, 2 * c)
        self.decoder1 = TimeBlock(3 * c, c)
        self.out = nn.Conv2d(c, 1, 1)

    def forward(self, x, t, y):
        embedding = self.embedding(t) + self.label_embedding(y)
        a = self.encoder1(x, embedding)
        b = self.encoder2(F.avg_pool2d(a, 2), embedding)
        h = self.middle(F.avg_pool2d(b, 2), embedding)
        h = F.interpolate(h, size=b.shape[-2:], mode='nearest')
        h = self.decoder2(torch.cat([h, b], dim=1), embedding)
        h = F.interpolate(h, size=a.shape[-2:], mode='nearest')
        h = self.decoder1(torch.cat([h, a], dim=1), embedding)
        return self.out(h)

def cosine_schedule(steps):
    u = torch.linspace(0, 1, steps + 1, dtype=torch.float64)
    cumulative = torch.cos((u + 0.008) / 1.008 * math.pi / 2).square()
    cumulative = cumulative / cumulative[0]
    betas = (1 - cumulative[1:] / cumulative[:-1]).clamp(1e-06, 0.999).float()
    alphas = 1 - betas
    abar = alphas.cumprod(0)
    previous = torch.cat([torch.ones(1), abar[:-1]])
    return {'betas': betas, 'alphas': alphas, 'abar': abar, 'posterior_var': betas * (1 - previous) / (1 - abar), 'coef_x0': betas * previous.sqrt() / (1 - abar), 'coef_xt': (1 - previous) * alphas.sqrt() / (1 - abar)}

class DigitClassifier(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(), nn.Linear(32 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 10))

    def forward(self, x):
        return self.net(x)
