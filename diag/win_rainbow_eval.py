"""Score the Rainbow C51 model on Windows with EnvPool's exact observation pipeline.

The Windows/gymnasium path historically scored ~280 against EnvPool's 3,530 for the same
checkpoint. Dumping (RGB frame, 84x84 obs) pairs from a lockstep pair of EnvPool instances showed
the order of operations is what differs: EnvPool grayscales *then* resizes with INTER_AREA
(99.6% of pixels exact), while gymnasium's AtariPreprocessing resizes the colour frame first
(92.0%). This harness implements the EnvPool order and max-pools the last two of the four skipped
frames, then reports the score with both pipelines so the gap is measurable rather than assumed.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import ale_py
import cv2
import gymnasium as gym
import numpy as np
import torch as th

sys.path.insert(0, r"C:\Users\melet\AI\rl-games")
from rainbow_pacman_3 import DuelingC51  # noqa: E402

gym.register_envs(ale_py)
CKPT = r"C:\Users\melet\AI\rl-games\models\rainbow_scale_10m\best.pt"
VMIN, VMAX, ATOMS = -1.0, 25.0, 51
N_ACT = 9


def load_net(device: str) -> DuelingC51:
    net = DuelingC51(4, N_ACT, ATOMS, width=64).to(device).eval()
    sd = th.load(CKPT, map_location=device, weights_only=True)
    net.load_state_dict(sd)
    return net


def prep_envpool(rgb: np.ndarray) -> np.ndarray:
    """EnvPool order: grayscale, then resize with INTER_AREA."""
    return cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (84, 84), interpolation=cv2.INTER_AREA)


def prep_gymnasium(rgb: np.ndarray) -> np.ndarray:
    """gymnasium AtariPreprocessing order: resize the colour frame, then grayscale."""
    return cv2.cvtColor(cv2.resize(rgb, (84, 84), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)


def run_episode(net, device: str, seed: int, prep, max_decisions: int = 4000):
    env = gym.make("ALE/MsPacman-v5", frameskip=1, repeat_action_probability=0.0)
    frame, _ = env.reset(seed=seed)
    support = th.linspace(VMIN, VMAX, ATOMS, device=device)

    small = prep(frame)
    stack = [small] * 4
    score, decisions = 0.0, 0
    while decisions < max_decisions:
        obs = np.asarray(stack, dtype=np.uint8)[None]
        with th.no_grad():
            q = net.q(th.as_tensor(obs, device=device), support)
        a = int(q.argmax(1).item())

        frames, rew, done = [], 0.0, False
        for _ in range(4):                                    # frameskip 4, ALE-native frames
            frame, r, term, trunc, _ = env.step(a)
            frames.append(frame)
            rew += float(r)
            if term or trunc:
                done = True
                break
        pooled = np.maximum(frames[-1], frames[-2]) if len(frames) >= 2 else frames[-1]
        stack = stack[1:] + [prep(pooled)]
        score += rew
        decisions += 1
        if done:
            break
    env.close()
    return score, decisions


def main() -> None:
    device = "cuda" if th.cuda.is_available() else "cpu"
    net = load_net(device)
    print(f"device {device} | params {sum(p.numel() for p in net.parameters()):,} | "
          f"support [{VMIN}, {VMAX}] atoms {ATOMS}", flush=True)

    seeds = [1, 2, 3]
    for label, prep in (("envpool order", prep_envpool), ("gymnasium order", prep_gymnasium)):
        scores = []
        for s in seeds:
            sc, dec = run_episode(net, device, s, prep)
            scores.append(sc)
            print(f"  {label:16s} seed {s}: {sc:6.0f} pts over {dec:4d} decisions", flush=True)
        print(f"{label:18s} mean {np.mean(scores):7.1f}   (EnvPool reference: 3,530)", flush=True)


if __name__ == "__main__":
    main()
