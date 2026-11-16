"""Sample-efficient alternative to PPO: QR-DQN (distributional, off-policy) + EnvPool.

Off-policy + replay reuses every frame far more than PPO's single gradient pass, and the
distributional head (QR-DQN) is the biggest single win in the Rainbow ablation after PER.
Replay lives in VRAM so the 100k-frame buffer doesn't eat 5.6 GB of system RAM.

  wsl -d jaxgpu -u root -- /root/gpurl/bin/python \
      /mnt/c/Users/melet/AI/rl-games/qrdqn_pacman.py --envs 32 --steps 4000000
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch as th
from sb3_contrib import QRDQN
from stable_baselines3.common.callbacks import BaseCallback

sys.path.insert(0, str(Path(__file__).resolve().parent))
from envpool_vec import EnvPoolAtari  # noqa: E402
from replay_gpu import TorchReplayBuffer  # noqa: E402

ROOT = Path("/mnt/c/Users/melet/AI/rl-games")


class Speed(BaseCallback):
    def __init__(self, every: int):
        super().__init__()
        self.every, self.n, self.t, self.rewards = every, 0, time.time(), []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.rewards = (self.rewards + [info["episode"]["r"]])[-50:]
        if self.num_timesteps - self.n >= self.every:
            now = time.time()
            sps = (self.num_timesteps - self.n) / max(now - self.t, 1e-9)
            mean = sum(self.rewards) / len(self.rewards) if self.rewards else float("nan")
            print(f"[{self.num_timesteps:>10,}] {sps:8,.0f} steps/s | ep_rew_mean(last {len(self.rewards)}) = {mean:8.2f}",
                  flush=True)
            self.n, self.t = self.num_timesteps, now
        return True


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--game", default="MsPacman")
    p.add_argument("--envs", type=int, default=32)
    p.add_argument("--steps", type=float, default=4_000_000)
    p.add_argument("--buffer-size", type=int, default=200_000, help="transitions (kept in VRAM)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--n-step", type=int, default=3)
    p.add_argument("--n-quantiles", type=int, default=51)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--train-freq", type=int, default=4)
    p.add_argument("--gradient-steps", type=int, default=1)
    p.add_argument("--learning-starts", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bf16", action="store_true", help="bfloat16 autocast for the CNN forward/backward")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    print(f"torch {th.__version__} cuda={th.cuda.is_available()}", flush=True)
    game_key = args.game.lower()
    tag = "qrdqn_envpool"
    out = ROOT / "models" / game_key / tag
    logs = ROOT / "logs" / game_key / tag
    out.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    env = EnvPoolAtari(f"{args.game}-v5", num_envs=args.envs, seed=args.seed)
    rows = max(2000, args.buffer_size // args.envs)
    print(f"obs={env.observation_space} actions={env.action_space.n} envs={args.envs} "
          f"buffer={rows * args.envs:,} transitions (~{rows * args.envs * 4 * 84 * 84 / 1024**3:.2f} GB VRAM)", flush=True)

    model = QRDQN(
        "CnnPolicy", env, verbose=0, device=args.device, seed=args.seed,
        learning_rate=args.lr, buffer_size=rows, learning_starts=args.learning_starts,
        batch_size=args.batch_size, gamma=args.gamma, train_freq=args.train_freq,
        gradient_steps=args.gradient_steps, target_update_interval=1000,
        exploration_fraction=0.10, exploration_final_eps=0.01,
        policy_kwargs={"n_quantiles": args.n_quantiles},
        replay_buffer_class=TorchReplayBuffer,
        tensorboard_log=str(logs),
    )

    steps = int(args.steps)
    if args.bf16:
        orig_forward = model.policy.forward

        def bf16_forward(obs, deterministic=False, episode_start=None):
            with th.autocast("cuda", dtype=th.bfloat16):
                return orig_forward(obs, deterministic, episode_start)

        model.policy.forward = bf16_forward
        print("bf16 autocast: ON", flush=True)

    t0 = time.time()
    model.learn(total_timesteps=steps, callback=Speed(every=max(steps // 50, 25_000)), tb_log_name="run")
    model.save(out / "final.zip")
    meta = {"game": game_key, "env_id": f"ALE/{args.game}-v5", "algo": "qrdqn", "steps": steps,
            "n_envs": args.envs, "n_step": args.n_step, "seconds": round(time.time() - t0, 1),
            "final_model": str(out / "final.zip")}
    (out / "train_meta.json").write_text(json.dumps(meta, indent=2))
    print("TRAINED", json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
