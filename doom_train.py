"""PPO (Stable-Baselines3) on ViZDoom scenarios.

  python doom_train.py --scenario defend_the_center --steps 600000 --n-envs 8
  python doom_train.py --scenario deadly_corridor --steps 2000000
"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch as th
from stable_baselines3 import PPO
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecTransposeImage

from doom_env import DoomEnv

class SpeedCallback(BaseCallback):
    """Local copy: importing train.py drags in ale_py, which the WSL env doesn't need."""

    def __init__(self, every: int = 5_000):
        super().__init__()
        self.every, self.last, self.t = every, 0, time.time()

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last >= self.every:
            now = time.time()
            print(f"[{self.num_timesteps:>9,}] {(self.num_timesteps - self.last)/(now - self.t):8,.0f} steps/s "
                  f"| ep_rew_mean {self.model.ep_info_buffer and float(np.mean([e['r'] for e in self.model.ep_info_buffer])) or float('nan'):.2f}",
                  flush=True)
            self.last, self.t = self.num_timesteps, now
        return True


ROOT = Path(__file__).resolve().parent


def make_env(scenario: str, frame_skip: int, seed: int, rank: int):
    def _init():
        return DoomEnv(scenario=scenario, frame_skip=frame_skip, seed=seed + rank)
    return _init


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="defend_the_center")
    p.add_argument("--steps", type=float, default=600_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--frame-skip", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--n-steps", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=20_000)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--tag", default="")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    if args.device == "cuda":
        if not th.cuda.is_available():
            raise SystemExit("CUDA not available — refusing to silently run on CPU.")
        name = th.cuda.get_device_name(0)
        print(f"GPU: {name} ({th.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB VRAM)", flush=True)
        if any(k in name.lower() for k in ("intel", "uhd", "iris", "radeon", "arc", "microsoft basic")):
            raise SystemExit(f"Refusing to train on '{name}' (iGPU).")

    run_name = f"ppo{args.tag}"
    out = ROOT / "models" / "doom" / args.scenario / run_name
    logs = ROOT / "logs" / "doom" / args.scenario / run_name
    out.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    vec_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    train_env = SubprocVecEnv([make_env(args.scenario, args.frame_skip, args.seed, i) for i in range(args.n_envs)]) \
        if args.n_envs > 1 else DummyVecEnv([make_env(args.scenario, args.frame_skip, args.seed, 0)])
    train_env = VecTransposeImage(VecFrameStack(train_env, n_stack=4))
    eval_env = DummyVecEnv([make_env(args.scenario, args.frame_skip, args.seed + 500, 0)])
    eval_env = VecTransposeImage(VecFrameStack(eval_env, n_stack=4))
    print(f"scenario={args.scenario} obs={train_env.observation_space} actions={train_env.action_space.n} "
          f"n_envs={args.n_envs}", flush=True)

    model = PPO(
        "CnnPolicy", train_env, verbose=0, device=args.device, seed=args.seed,
        learning_rate=args.lr, n_steps=args.n_steps, batch_size=args.batch_size, n_epochs=args.n_epochs,
        gamma=0.99, gae_lambda=0.95, clip_range=0.1, ent_coef=args.ent_coef, vf_coef=0.5,
        max_grad_norm=0.5, tensorboard_log=str(logs),
    )

    steps = int(args.steps)
    callbacks = [
        SpeedCallback(every=max(steps // 50, 5_000)),
        EvalCallback(eval_env, best_model_save_path=str(out), log_path=str(logs / "eval"),
                     eval_freq=max(args.eval_every // args.n_envs, 1), n_eval_episodes=args.eval_episodes,
                     deterministic=True),
        CheckpointCallback(save_freq=max(100_000 // args.n_envs, 10_000), save_path=str(out / "ckpt"), name_prefix="ppo"),
    ]

    t0 = time.time()
    model.learn(total_timesteps=steps, callback=callbacks, tb_log_name="run")
    final = out / "final.zip"
    model.save(final)
    meta = {"scenario": args.scenario, "algo": "ppo", "steps": steps, "n_envs": args.n_envs,
            "seconds": round(time.time() - t0, 1), "final_model": str(final)}
    (out / "train_meta.json").write_text(json.dumps(meta, indent=2))
    print("TRAINED", json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
