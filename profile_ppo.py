"""Split the PPO wall-clock: env stepping vs gradient updates. Run under WSL.

  wsl -d jaxgpu -u root -- /root/gpurl/bin/python \
      /mnt/c/Users/melet/AI/rl-games/profile_ppo.py --envs 64 --n-steps 128 --batch-size 1024 --epochs 3
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch as th
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from envpool_vec import EnvPoolAtari  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--envs", type=int, default=64)
    p.add_argument("--n-steps", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--reps", type=int, default=6)
    args = p.parse_args()

    env = EnvPoolAtari("MsPacman-v5", num_envs=args.envs, seed=3)
    model = PPO("CnnPolicy", env, device="cuda", n_steps=args.n_steps,
                batch_size=args.batch_size, n_epochs=args.epochs, verbose=0)
    model.learn(total_timesteps=args.envs * args.n_steps)  # warmup + fill rollout buffer

    # (a) environment stepping only
    model._last_obs = env.reset()
    acts = np.zeros(args.envs, dtype=np.int32)
    for _ in range(20):
        env.step(acts)
    t0 = time.time()
    for _ in range(200):
        env.step(acts)
    env_dt = time.time() - t0
    env_steps = 200 * args.envs
    print(f"env step:   {env_steps / env_dt:9,.0f} steps/s   ({env_dt / 200 * 1000:.2f} ms per vec step)", flush=True)

    # (b) gradient updates only (rollout buffer already full)
    model.train()  # one clean timing pass
    th.cuda.synchronize()
    t0 = time.time()
    for _ in range(args.reps):
        model.train()
    th.cuda.synchronize()
    train_dt = (time.time() - t0) / args.reps
    updates = args.epochs * (args.n_steps * args.envs // args.batch_size)
    print(f"ppo update: {train_dt * 1000:9.1f} ms per train() call "
          f"({updates} minibatch updates, {train_dt / updates * 1000:.1f} ms each)", flush=True)

    rollout = args.n_steps * args.envs
    print(f"\nper rollout of {rollout:,} env steps: env {env_dt / 200 * (rollout / args.envs):.2f}s "
          f"+ train {train_dt:.2f}s  ->  {rollout / (env_dt / 200 * (rollout / args.envs) + train_dt):,.0f} steps/s end-to-end", flush=True)
    print(f"share of wall clock: env {env_dt / 200 * (rollout / args.envs) / (env_dt / 200 * (rollout / args.envs) + train_dt) * 100:.0f}% "
          f"/ train {train_dt / (env_dt / 200 * (rollout / args.envs) + train_dt) * 100:.0f}%")


if __name__ == "__main__":
    main()
