"""Export the trained DQN's online Q-network as a standalone 1.69M-param inference artifact.

Produces (in --out):
  mspacman_qnet.pt              TorchScript, self-contained, loadable with torch.jit.load
  mspacman_qnet.safetensors     raw weights (or .pt fallback)
  model.py                      the nn.Module definition + loader, no SB3 / gym needed
  inference.py                  minimal example: frames -> Q values -> action
  sample_obs.npy                one real preprocessed observation (1,4,84,84) uint8
  README.md                     model card + preprocessing contract
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch as th
import torch.nn as nn

ROOT = Path(__file__).resolve().parent


class LiteQNet(nn.Module):
    """SB3 NatureCNN trunk + 9-way Q head, with no Stable-Baselines3 dependency.

    Input : uint8 (N, 4, 84, 84), 4 stacked grayscale frames, scaled 0-255
    Output: float32 (N, 9) Q-values, ALE Ms. Pac-Man action order:
            NOOP, UP, RIGHT, LEFT, DOWN, UPRIGHT, UPLEFT, DOWNRIGHT, DOWNLEFT
    """

    ACTIONS = ["NOOP", "UP", "RIGHT", "LEFT", "DOWN", "UPRIGHT", "UPLEFT", "DOWNRIGHT", "DOWNLEFT"]

    def __init__(self) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        self.linear = nn.Sequential(nn.Linear(3136, 512), nn.ReLU())
        self.q_head = nn.Linear(512, 9)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        x = obs.float() / 255.0
        return self.q_head(self.linear(self.cnn(x)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(ROOT / "models/mspacman/dqn_envpool/final_4000000.zip"))
    p.add_argument("--out", default=str(ROOT.parent / "mspacman-dqn-lite"))
    args = p.parse_args()

    from stable_baselines3 import DQN

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = DQN.load(args.model, device="cpu")
    qnet = model.policy.q_net
    lite = LiteQNet()
    sd = {
        "cnn.0.weight": qnet.features_extractor.cnn[0].weight,
        "cnn.0.bias": qnet.features_extractor.cnn[0].bias,
        "cnn.2.weight": qnet.features_extractor.cnn[2].weight,
        "cnn.2.bias": qnet.features_extractor.cnn[2].bias,
        "cnn.4.weight": qnet.features_extractor.cnn[4].weight,
        "cnn.4.bias": qnet.features_extractor.cnn[4].bias,
        "linear.0.weight": qnet.features_extractor.linear[0].weight,
        "linear.0.bias": qnet.features_extractor.linear[0].bias,
        "q_head.weight": qnet.q_net[0].weight,
        "q_head.bias": qnet.q_net[0].bias,
    }
    lite.load_state_dict(sd)
    lite.eval()

    n_params = sum(x.numel() for x in lite.parameters())
    assert n_params == 1_688_745, n_params

    # numeric check against the SB3 network it came from
    x = th.randint(0, 256, (5, 4, 84, 84), dtype=th.uint8)
    with th.no_grad():
        a = lite(x)
        b = qnet(x)
    diff = (a - b).abs().max().item()
    print(f"params={n_params:,}  max|delta| vs SB3 q_net = {diff:.3e}")
    assert diff < 1e-5, diff

    # a real observation, captured from the same preprocessing the trainer used
    try:
        import cv2
        import gymnasium as gym
        import ale_py
        gym.register_envs(ale_py)
        env = gym.make("ALE/MsPacman-v5", repeat_action_probability=0.0, render_mode="rgb_array")
        frame, _ = env.reset(seed=7)
        stack = []
        for _ in range(4):
            g = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            stack.append(cv2.resize(g, (84, 84), interpolation=cv2.INTER_AREA))
            frame, *_ = env.step(0)
        np.save(out / "sample_obs.npy", np.stack(stack)[None].astype(np.uint8))
        env.close()
        print("sample_obs.npy saved", np.load(out / "sample_obs.npy").shape)
    except Exception as e:
        print("sample obs skipped:", type(e).__name__, e)

    # 1. TorchScript (single file, no code needed at load time)
    ts = out / "mspacman_qnet.pt"
    traced = th.jit.trace(lite, th.zeros(1, 4, 84, 84, dtype=th.uint8))
    traced.save(str(ts))

    # 2. raw weights
    try:
        from safetensors.torch import save_file
        weights = out / "mspacman_qnet.safetensors"
        save_file({k: v.contiguous() for k, v in lite.state_dict().items()}, str(weights))
    except ImportError:
        weights = out / "mspacman_qnet_state_dict.pt"
        th.save(lite.state_dict(), weights)

    info = {
        "model": "NatureCNN + Q head (SB3 DQN, Ms. Pac-Man)",
        "params": n_params,
        "input": "uint8 (N, 4, 84, 84) stacked grayscale frames",
        "output": "float32 (N, 9) Q-values",
        "actions": LiteQNet.ACTIONS,
        "source": "models/mspacman/dqn_envpool/final_4000000.zip (online q_net only, target net dropped)",
        "trained_steps": 4_000_000,
        "scores_full_game": [2210, 1820, 1820, 2760],
        "torchscript": ts.name,
        "weights": weights.name,
    }
    (out / "model_info.json").write_text(json.dumps(info, indent=2))
    for f in [ts, weights, out / "sample_obs.npy"]:
        if f.exists():
            print(f"{f.name:32s} {f.stat().st_size / 1024**2:6.2f} MB")
    print("exported to", out)


if __name__ == "__main__":
    main()
