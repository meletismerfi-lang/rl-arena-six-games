"""DQN (off-policy) on EnvPool Atari, for the PPO vs DQN comparison.

  wsl -d jaxgpu -u root -- /root/gpurl/bin/python \
      /mnt/c/Users/melet/AI/rl-games/dqn_pacman.py --envs 32 --steps 1000000
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch as th
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

sys.path.insert(0, str(Path(__file__).resolve().parent))
from envpool_vec import EnvPoolAtari  # noqa: E402
from replay_gpu import TorchReplayBuffer  # noqa: E402

ROOT = Path("/mnt/c/Users/melet/AI/rl-games")


class Speed(BaseCallback):
    def __init__(self, every: int):
        super().__init__()
        self.every, self.n, self.t = every, 0, time.time()

    def _on_step(self) -> bool:
        if self.num_timesteps - self.n >= self.every:
            now = time.time()
            print(f"[{self.num_timesteps:>9,}] {(self.num_timesteps - self.n) / max(now - self.t, 1e-9):8,.0f} steps/s",
                  flush=True)
            self.n, self.t = self.num_timesteps, now
        return True


def safe_save(model, path: Path) -> None:
    """Save weights only.

    Two things must be detached or the write fails / explodes:
      - model.env: cloudpickle cannot serialise AtariGymnasiumEnvPool -> raises mid-write,
        leaving a 0-byte .zip behind.
      - model.replay_buffer: 200k frames in VRAM -> the .zip becomes ~7 GB and the pickle
        can exhaust memory.
    """
    env, buf = model.env, getattr(model, "replay_buffer", None)
    model.env = None
    if buf is not None:
        model.replay_buffer = None
    try:
        model.save(str(path))
    finally:
        model.env = env
        if buf is not None:
            model.replay_buffer = buf


class SafeCheckpoint(BaseCallback):
    def __init__(self, every_steps: int, save_path: Path, prefix: str, n_envs: int):
        super().__init__()
        self.every, self.dir, self.prefix, self.n_envs = every_steps, save_path, prefix, n_envs
        self.dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps and self.num_timesteps % self.every < self.n_envs:
            p = self.dir / f"{self.prefix}_{self.num_timesteps}_steps.zip"
            if not p.exists():
                safe_save(self.model, p)
                print(f"checkpoint: {p.name}", flush=True)
        return True


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--game", default="MsPacman")
    p.add_argument("--envs", type=int, default=32)
    p.add_argument("--steps", type=float, default=1_000_000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--train-freq", type=int, default=4)
    p.add_argument("--gradient-steps", type=int, default=2)
    p.add_argument("--learning-starts", type=int, default=20_000)
    p.add_argument("--buffer-size", type=int, default=200_000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", default=None, help="checkpoint to continue from (weights only; replay starts empty)")
    p.add_argument("--ckpt-every", type=int, default=500_000, help="env steps between checkpoints")
    args = p.parse_args()

    th.set_num_threads(1)
    # tensor cores for the FC layers too: cuDNN's TF32 covers convs by default, but PyTorch
    # keeps matmul on FP32. Measured 16.5 -> 15.2 ms per update (batch 1024) on this GPU.
    th.backends.cuda.matmul.allow_tf32 = True
    th.backends.cudnn.allow_tf32 = True
    th.backends.cudnn.benchmark = True   # autotune conv algos for the fixed 84x84 input
    print(f"torch {th.__version__} cuda={th.cuda.is_available()}", flush=True)
    game_key = args.game.lower()
    out = ROOT / "models" / game_key / "dqn_envpool"
    out.mkdir(parents=True, exist_ok=True)

    env = EnvPoolAtari(f"{args.game}-v5", num_envs=args.envs, seed=args.seed, reward_clip=False)
    rows = max(2000, args.buffer_size // args.envs)
    print(f"obs={env.observation_space} actions={env.action_space.n} envs={args.envs} "
          f"buffer={rows * args.envs:,} transitions (~{rows * args.envs * 4 * 84 * 84 / 1024**3:.2f} GB VRAM)", flush=True)

    if args.resume:
        print(f"resuming weights from {args.resume}", flush=True)
        model = DQN.load(args.resume, env=env, device=args.device,
                         replay_buffer_class=TorchReplayBuffer, buffer_size=rows,
                         learning_starts=args.learning_starts)
        model.replay_buffer = model.replay_buffer_class(
            rows, env.observation_space, env.action_space, device=model.device, n_envs=args.envs,
        )
        model.exploration_initial_eps, model.exploration_final_eps = 1.0, 0.01
        model.exploration_rate = 0.0     # already explored in the earlier phase; exploit now
        model.exploration_schedule = lambda _: model.exploration_final_eps
    else:
        model = DQN(
            "CnnPolicy", env, verbose=0, device=args.device, seed=args.seed,
            learning_rate=args.lr, buffer_size=rows, learning_starts=args.learning_starts,
            batch_size=args.batch_size, gamma=args.gamma, train_freq=args.train_freq,
            gradient_steps=args.gradient_steps, target_update_interval=1000,
            exploration_fraction=0.10, exploration_final_eps=0.01,
            replay_buffer_class=TorchReplayBuffer,
        )

    steps = int(args.steps)
    t0 = time.time()
    model.learn(
        total_timesteps=steps,
        callback=[
            Speed(every=max(steps // 20, 25_000)),
            SafeCheckpoint(args.ckpt_every, out / "ckpt", f"continue_{steps}", args.envs),
        ],
        tb_log_name=f"dqn_{steps}",
    )
    safe_save(model, out / f"final_{steps}.zip")
    meta = {"algo": "dqn", "steps": steps, "envs": args.envs, "seconds": round(time.time() - t0, 1),
            "steps_per_s": round(steps / max(time.time() - t0, 1e-9), 1),
            "resumed_from": args.resume}
    (out / f"meta_{steps}.json").write_text(json.dumps(meta, indent=2))
    print("TRAINED", json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
