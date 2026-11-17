"""Train the Flappy bot on the 6-float state view — learns indefinite play in ~2 minutes.

  .venv/Scripts/python.exe train_flappy_state.py --steps 400000 --envs 16
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from flappy_env import StateFlappyEnv

ROOT = Path(__file__).resolve().parent


class Progress(BaseCallback):
    def __init__(self, model, every=25_000):
        super().__init__()
        self.model, self.every, self.last, self.t, self.best = model, every, 0, time.time(), -1e9

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last >= self.every:
            env = StateFlappyEnv()
            scores = []
            for _ in range(5):
                obs, _ = env.reset()
                done = False
                total = 0.0
                while not done:
                    act, _ = self.model.predict(obs, deterministic=True)
                    obs, r, term, trunc, info = env.step(int(act))
                    total += r
                    done = term or trunc
                scores.append(env.score)
            now = time.time()
            mean = float(np.mean(scores))
            self.best = max(self.best, mean)
            print(f"[{self.num_timesteps:>8,}] {(self.num_timesteps - self.last)/(now-self.t):7,.0f} steps/s "
                  f"| pipes mean {mean:6.1f} max {max(scores):4d} (best {self.best:.1f})", flush=True)
            self.last, self.t = self.num_timesteps, now
        return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400_000)
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    th.set_num_threads(1)
    out = ROOT / "models" / "flappy" / "ppo_state"
    out.mkdir(parents=True, exist_ok=True)
    vec = SubprocVecEnv([lambda i=i: StateFlappyEnv() for i in range(args.envs)]) if args.envs > 1 \
        else DummyVecEnv([lambda: StateFlappyEnv()])
    model = PPO("MlpPolicy", vec, verbose=0, device=args.device,
                seed=args.seed, n_steps=256, batch_size=1024, n_epochs=6,
                learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, ent_coef=0.0,
                policy_kwargs={"net_arch": [64, 64]})
    model.learn(total_timesteps=args.steps, callback=Progress(model))
    model.env = None
    model.save(out / "final.zip")
    print("saved", out / "final.zip")


if __name__ == "__main__":
    main()
