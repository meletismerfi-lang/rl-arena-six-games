"""What does a ~30M-parameter trunk cost per update on this GPU? Measured, not estimated."""
import time

import torch as th
import torch.nn as nn


class Wide(nn.Module):
    """NatureCNN scaled up: wider convs + bigger FC, sized to hit ~30M params."""

    def __init__(self, c1=96, c2=192, c3=192, fc=2048, out=512):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(4, c1, 8, 4), nn.ReLU(),
            nn.Conv2d(c1, c2, 4, 2), nn.ReLU(),
            nn.Conv2d(c2, c3, 3, 1), nn.ReLU(),
            nn.Flatten(),
        )
        flat = c3 * 7 * 7
        self.linear = nn.Sequential(nn.Linear(flat, fc), nn.ReLU(), nn.Linear(fc, out), nn.ReLU())
        self.head = nn.Linear(out, 9)

    def forward(self, x):
        return self.head(self.linear(self.cnn(x.float() / 255.0)))


def bench(model, batch, iters=8):
    model = model.cuda()
    opt = th.optim.Adam(model.parameters(), lr=1e-4)
    x = th.randint(0, 255, (batch, 4, 84, 84), dtype=th.uint8, device="cuda")
    for _ in range(3):
        loss = model(x).pow(2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    th.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        loss = model(x).pow(2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    th.cuda.synchronize()
    ms = (time.time() - t0) / iters * 1000
    n = sum(p.numel() for p in model.parameters())
    return ms, n


if __name__ == "__main__":
    th.backends.cuda.matmul.allow_tf32 = True
    print(f"device {th.cuda.get_device_name(0)} | VRAM {th.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
    for batch in (32, 64):
        for tag, m in [
            ("current 1.7M (NatureCNN)", Wide(c1=32, c2=64, c3=64, fc=512)),
            ("~4M (wider FC)", Wide(c1=48, c2=96, c3=96, fc=1024)),
            ("~30M target", Wide(c1=96, c2=192, c3=192, fc=2048)),
        ]:
            try:
                ms, n = bench(m, batch)
                print(f"batch {batch:>3} | {tag:26s} {n/1e6:5.1f}M params  {ms:7.1f} ms/update", flush=True)
            except RuntimeError as e:
                print(f"batch {batch:>3} | {tag:26s} FAILED: {str(e)[:60]}", flush=True)
