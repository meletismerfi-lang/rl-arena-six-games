"""CNN (NatureCNN) vs a small ViT-style transformer: cost per PPO update at our batch size."""
import time

import torch as th
import torch.nn as nn


class Cnn(th.nn.Module):
    """SB3 NatureCNN trunk, 4x84x84 -> 512."""

    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
            nn.Flatten(),
        )
        self.linear = nn.Sequential(nn.Linear(3136, 512), nn.ReLU())

    def forward(self, x):
        return self.linear(self.cnn(x.float() / 255.0))


class TinyViT(th.nn.Module):
    """Patch 8 transformer over 84x84x4 -> 512, same output width for a fair comparison."""

    def __init__(self, dim=256, depth=4, heads=4, patch=12):
        super().__init__()
        n_patches = (84 // patch) ** 2
        self.patch = patch
        self.proj = nn.Conv2d(4, dim, patch, patch)
        self.cls = nn.Parameter(th.zeros(1, 1, dim))
        self.pos = nn.Parameter(th.zeros(1, n_patches + 1, dim))
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, depth)
        self.head = nn.Sequential(nn.Linear(dim, 512), nn.ReLU())

    def forward(self, x):
        x = self.proj(x.float() / 255.0).flatten(2).transpose(1, 2)
        x = th.cat([self.cls.expand(x.shape[0], -1, -1), x], dim=1) + self.pos
        return self.head(self.enc(x)[:, 0])


def bench(model, batch=1024, iters=12):
    x = th.randint(0, 255, (batch, 4, 84, 84), dtype=th.uint8, device="cuda")
    opt = th.optim.Adam(model.parameters(), lr=1e-4)
    for _ in range(3):
        loss = model(x).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    th.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        loss = model(x).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    th.cuda.synchronize()
    ms = (time.time() - t0) / iters * 1000
    n = sum(p.numel() for p in model.parameters())
    return ms, n


if __name__ == "__main__":
    for name, m in [("CNN (NatureCNN)", Cnn().cuda()), ("ViT dim256 d4 patch12", TinyViT().cuda())]:
        ms, n = bench(m)
        print(f"{name:24s} {ms:7.1f} ms/update  ({n:,} params)", flush=True)
