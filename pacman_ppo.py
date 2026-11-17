"""PPO on EnvPool Atari (fast). Runs under WSL where EnvPool has wheels.

  wsl -d jaxgpu -u root -- /root/gpurl/bin/python \
     /mnt/c/Users/melet/AI/rl-games/pacman_ppo.py --game MsPacman --envs 64 --steps 20000000
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

sys.path.insert(0, str(Path(__file__).resolve().parent))
from envpool_vec import EnvPoolAtari  # noqa: E402

ROOT = Path("/mnt/c/Users/melet/AI/rl-games")


class Speed(BaseCallback):
    def __init__(self, every: int):
        super().__init__()
        self.every = every
        self.n = 0
        self.t = time.time()
        self.rewards = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.rewards = (self.rewards + [info["episode"]["r"]])[-50:]
        if self.num_timesteps - self.n >= self.every:
            now = time.time()
            sps = (self.num_timesteps - self.n) / max(now - self.t, 1e-9)
            mean = sum(self.rewards) / len(self.rewards) if self.rewards else float("nan")
            print(f"[{self.num_timesteps:>10,}] {sps:8,.0f} steps/s | "
                  f"ep_rew_mean(last {len(self.rewards)}) = {mean:8.2f} | {now - self.t:5.1f}s", flush=True)
            self.n, self.t = self.num_timesteps, now
        return True


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--game", default="MsPacman")
    p.add_argument("--envs", type=int, default=64)
    p.add_argument("--steps", type=float, default=20_000_000)
    p.add_argument("--n-steps", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--clip", type=float, default=0.1)
    p.add_argument("--eval-every", type=int, default=1_000_000)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bf16", action="store_true", help="(alias) bfloat16 autocast")
    p.add_argument("--amp", default="none", choices=["none", "bf16", "fp16"],
                   help="autocast dtype; fp16 also enables loss scaling")
    p.add_argument("--loss-scale", type=float, default=0.0,
                   help="initial loss scale (0 = off for bf16, 65536 for fp16, dynamic)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-reward-clip", action="store_true",
                   help="keep the real score deltas (dot=10, ghost=200-1600) instead of +/-1")
    p.add_argument("--lr-schedule", default="linear", choices=["none", "linear"])
    p.add_argument("--torch-threads", type=int, default=1)
    args = p.parse_args()

    print(f"torch {th.__version__} cuda={th.cuda.is_available()} dev={th.cuda.get_device_name(0) if th.cuda.is_available() else 'cpu'}",
          flush=True)
    th.set_num_threads(args.torch_threads)   # leave cores to EnvPool's workers
    th.backends.cuda.matmul.allow_tf32 = True   # tensor cores for the FC matmul (convs already TF32)
    th.backends.cudnn.allow_tf32 = True

    game_key = args.game.lower()
    tag = "ppo_envpool"
    out = ROOT / "models" / game_key / tag
    logs = ROOT / "logs" / game_key / tag
    out.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    train_env = EnvPoolAtari(f"{args.game}-v5", num_envs=args.envs, seed=args.seed,
                             reward_clip=not args.no_reward_clip)
    eval_env = EnvPoolAtari(f"{args.game}-v5", num_envs=4, seed=args.seed + 999,
                            reward_clip=not args.no_reward_clip)
    print(f"obs={train_env.observation_space} actions={train_env.action_space.n} envs={args.envs}", flush=True)

    model = PPO(
        "CnnPolicy", train_env, verbose=0, device=args.device, seed=args.seed,
        learning_rate=(lambda remaining: remaining * args.lr) if args.lr_schedule == "linear" else args.lr, n_steps=args.n_steps, batch_size=args.batch_size, n_epochs=args.n_epochs,
        gamma=0.99, gae_lambda=0.95, clip_range=args.clip, ent_coef=args.ent_coef, vf_coef=0.5,
        max_grad_norm=0.5, tensorboard_log=str(logs),
    )

    steps = int(args.steps)
    amp_dtype = "bf16" if args.bf16 else args.amp
    amp_state = None
    if amp_dtype != "none":
        from amp_scaling import enable_amp
        amp_state = enable_amp(model, dtype=amp_dtype, scale=args.loss_scale)

    callbacks = [
        Speed(every=max(steps // 50, 50_000)),
        EvalCallback(eval_env, best_model_save_path=str(out), log_path=str(logs / "eval"),
                     eval_freq=max(args.eval_every // args.envs, 1), n_eval_episodes=args.eval_episodes,
                     deterministic=True),
    ]

    t0 = time.time()
    model.learn(total_timesteps=steps, callback=callbacks, tb_log_name="run")
    if amp_state is not None:
        from amp_scaling import disable_amp
        disable_amp(amp_state)
        print(f"amp summary: dtype={amp_state['dtype']} skipped_steps={amp_state['skipped']} "
              f"final_scale={amp_state['scale']:.0f}", flush=True)
    model.save(out / "final.zip")
    meta = {"game": game_key, "env_id": f"ALE/{args.game}-v5", "algo": "ppo", "steps": steps,
            "n_envs": args.envs, "seconds": round(time.time() - t0, 1), "final_model": str(out / "final.zip")}
    (out / "train_meta.json").write_text(json.dumps(meta, indent=2))
    print("TRAINED", json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
