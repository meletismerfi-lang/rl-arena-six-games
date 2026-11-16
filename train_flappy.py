"""PPO on the Flappy Bird env. ~500k steps is enough to play indefinitely.

  .venv/Scripts/python.exe train_flappy.py --steps 600000 --envs 16
"""
import argparse
from pathlib import Path

import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecTransposeImage

from flappy_env import FlappyEnv

ROOT = Path(__file__).resolve().parent


class Speed(BaseCallback):
    def __init__(self, every: int = 50_000):
        super().__init__()
        self.every, self.n, self.t = every, 0, __import__("time").time()

    def _on_step(self) -> bool:
        import time
        if self.num_timesteps - self.n >= self.every:
            now = time.time()
            print(f"[{self.num_timesteps:>8,}] {(self.num_timesteps - self.n) / (now - self.t):8,.0f} steps/s",
                  flush=True)
            self.n, self.t = self.num_timesteps, now
        return True


def make_env(rank: int):
    def _init():
        return FlappyEnv()
    return _init


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=float, default=600_000)
    p.add_argument("--envs", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-steps", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    th.set_num_threads(1)
    th.backends.cuda.matmul.allow_tf32 = True
    print(f"torch {th.__version__} cuda={th.cuda.is_available()}")

    out = ROOT / "models" / "flappy" / "ppo"
    out.mkdir(parents=True, exist_ok=True)

    vec_cls = SubprocVecEnv if args.envs > 1 else DummyVecEnv
    train_env = vec_cls([make_env(i) for i in range(args.envs)])
    train_env = VecTransposeImage(VecFrameStack(train_env, n_stack=4))
    eval_env = VecTransposeImage(VecFrameStack(DummyVecEnv([make_env(0)]), n_stack=4))
    print(f"obs={train_env.observation_space} actions={train_env.action_space.n} envs={args.envs}")

    model = PPO("CnnPolicy", train_env, verbose=0, device="cuda", seed=args.seed,
                learning_rate=args.lr, n_steps=args.n_steps, batch_size=args.batch_size,
                n_epochs=args.n_epochs, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, tensorboard_log=str(ROOT / "logs" / "flappy"))

    steps = int(args.steps)
    model.learn(total_timesteps=steps, callback=[
        Speed(max(steps // 12, 25_000)),
        EvalCallback(eval_env, best_model_save_path=str(out), log_path=str(ROOT / "logs" / "flappy" / "eval"),
                     eval_freq=max(25_000 // args.envs, 500), n_eval_episodes=5, deterministic=True),
    ])
    env0 = train_env
    model.env = None
    model.save(out / "final.zip")
    print("saved", out / "final.zip")


if __name__ == "__main__":
    main()
