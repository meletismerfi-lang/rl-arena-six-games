"""Evaluate a whole checkpoint sweep and report the real peak.

  .venv/Scripts/python.exe eval_sweep.py --glob "models/mspacman/dqn_envpool/ckpt/continue_8000000_*000000_steps.zip" --episodes 6

Training saves every 1M steps and keeps no eval, so the "final" model is not necessarily the
best. This walks the checkpoints, plays full games with the EnvPool-compatible preprocessing
(no noop-reset, no FIRE), and prints a table — the honest way to pick which checkpoint to ship.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import ale_py
import gymnasium as gym
import numpy as np
import torch as th
from gymnasium.wrappers import AtariPreprocessing, FrameStack
from stable_baselines3 import DQN

gym.register_envs(ale_py)
ENV_ID = "ALE/MsPacman-v5"


def make_env(seed: int):
    e = gym.make(ENV_ID, frameskip=1, repeat_action_probability=0.0, render_mode=None,
                 max_episode_steps=20_000)
    e = AtariPreprocessing(e, frame_skip=4, screen_size=84, grayscale_obs=True,
                           grayscale_newaxis=False, scale_obs=False, terminal_on_life_loss=False)
    e = FrameStack(e, 4)
    e.reset(seed=seed)
    return e


def evaluate(path: Path, episodes: int, device: str) -> tuple[float, float, float]:
    model = DQN.load(str(path), device="cpu")
    model.replay_buffer = None
    model.policy.to(device)
    scores = []
    for ep in range(episodes):
        env = make_env(7 + ep)
        obs, _ = env.reset(seed=7 + ep)
        done, total = False, 0.0
        while not done:
            action, _ = model.predict(np.asarray(obs)[None].astype(np.uint8), deterministic=True)
            obs, reward, term, trunc, _ = env.step(int(np.asarray(action).reshape(-1)[0]))
            total += float(reward)
            done = term or trunc
        env.close()
        scores.append(total)
    return float(np.mean(scores)), float(np.min(scores)), float(np.max(scores))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--device", default="cuda" if th.cuda.is_available() else "cpu")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.glob))
    if not paths:
        raise SystemExit(f"no checkpoints matched {args.glob}")
    print(f"{len(paths)} checkpoints × {args.episodes} episodes\n")
    rows = []
    best = (-1.0, None)
    for p in paths:
        m, lo, hi = evaluate(Path(p), args.episodes, args.device)
        rows.append((Path(p).name, m, lo, hi))
        best = max(best, (m, p))
        print(f"{Path(p).name:44s} mean {m:7.1f}  min {lo:6.1f}  max {hi:6.1f}", flush=True)

    print("\nranked:")
    for name, m, lo, hi in sorted(rows, key=lambda r: -r[1]):
        print(f"  {m:7.1f}  {name}")
    print(f"\nbest checkpoint: {best[1]}  ({best[0]:.1f})")


if __name__ == "__main__":
    main()
