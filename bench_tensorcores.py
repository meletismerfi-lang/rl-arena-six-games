"""Is the policy update using tensor cores? Time one NatureCNN update under each precision path."""
import time

import torch as th
import torch.nn as nn


class Trunk(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
            nn.Flatten(),
        )
        self.linear = nn.Sequential(nn.Linear(3136, 512), nn.ReLU())
        self.head = nn.Linear(512, 9)

    def forward(self, x):
        return self.head(self.linear(self.cnn(x.float() / 255.0)))


def bench(tag, dtype=None, tf32_matmul=False, tf32_cudnn=True, iters=10, batch=1024):
    th.backends.cuda.matmul.allow_tf32 = tf32_matmul
    th.backends.cudnn.allow_tf32 = tf32_cudnn
    m = Trunk().cuda()
    opt = th.optim.Adam(m.parameters(), lr=1e-4)
    x = th.randint(0, 255, (batch, 4, 84, 84), dtype=th.uint8, device="cuda")

    def step():
        if dtype:
            with th.autocast("cuda", dtype=dtype):
                loss = m(x).pow(2).mean()
        else:
            loss = m(x).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()

    for _ in range(3):
        step()
    th.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        step()
    th.cuda.synchronize()
    ms = (time.time() - t0) / iters * 1000
    print(f"{tag:46s} {ms:7.1f} ms/update   ({1000 / ms * batch:,.0f} samples/s)", flush=True)


if __name__ == "__main__":
    print(f"torch {th.__version__} | device {th.cuda.get_device_name(0)}")
    bench("fp32, TF32 matmul OFF (pure CUDA cores)", tf32_matmul=False, tf32_cudnn=False)
    bench("fp32, cuDNN TF32 ON (default)", tf32_matmul=False, tf32_cudnn=True)
    bench("fp32, TF32 matmul ON (tensor cores for FC)", tf32_matmul=True, tf32_cudnn=True)
    bench("bf16 autocast (tensor cores)", dtype=th.bfloat16, tf32_matmul=True)
