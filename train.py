"""Generic Atari RL trainer (Stable-Baselines3 + ALE).

Examples
--------
  python train.py --game pong --steps 600000
  python train.py --game breakout --algo ppo --steps 2000000 --lr 2.5e-4
  python train.py --game space_invaders --resume models/space_invaders/dqn/best.zip
"""
import argparse
import json
import os
import time
from pathlib import Path

# Pin the discrete NVIDIA GPU before CUDA initialises; never fall back to an iGPU.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import ale_py
import gymnasium as gym
import torch as th
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecTransposeImage

gym.register_envs(ale_py)

ROOT = Path(__file__).resolve().parent


class SpeedCallback(BaseCallback):
    """Prints steps/sec and a rolling episode-reward mean every `every` steps."""

    def __init__(self, every=10000, t0=None):
        super().__init__()
        self.every = every
        self.t0 = t0 or time.time()
        self.last_n = 0
        self.last_t = self.t0
        self.rewards = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.rewards = (self.rewards + [info["episode"]["r"]])[-50:]
        if self.num_timesteps - self.last_n >= self.every:
            now = time.time()
            sps = (self.num_timesteps - self.last_n) / max(now - self.last_t, 1e-9)
            mean_r = sum(self.rewards) / len(self.rewards) if self.rewards else float("nan")
            print(
                f"[{self.num_timesteps:>9,}] {sps:7.1f} steps/s | "
                f"ep_rew_mean(last {len(self.rewards)} eps) = {mean_r:7.2f} | "
                f"elapsed {now - self.t0:6.1f}s",
                flush=True,
            )
            self.last_n, self.last_t = self.num_timesteps, now
        return True


def build_game(game: str) -> str:
    game = game.strip().lower().replace(" ", "_").replace("-", "_")
    env_id = f"ALE/{game.capitalize()}-v5" if game.capitalize() in {
        k.split("/")[1].rsplit("-", 1)[0] for k in gym.registry if k.startswith("ALE/")
    } else None
    if env_id is None:
        # fall back to exact match search (handles MsPacman, SpaceInvaders, ...)
        hits = [k for k in gym.registry if k.lower().startswith(f"ale/{game.lower()}-v5")]
        if not hits:
            names = sorted({k.split("/")[1].rsplit("-", 1)[0] for k in gym.registry if k.startswith("ALE/")})
            raise SystemExit(f"Unknown game '{game}'. Try one of:\n  " + ", ".join(names))
        env_id = hits[0]
    return env_id


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--game", default="pong", help="e.g. pong, breakout, space_invaders, mspacman")
    p.add_argument("--algo", default="dqn", choices=["dqn", "ppo", "a2c"])
    p.add_argument("--steps", type=float, default=500_000, help="total environment steps (1e6 ok)")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-envs", type=int, default=1)
    p.add_argument("--vec-env", default="subproc", choices=["dummy", "subproc"])
    p.add_argument("--eval-every", type=int, default=25_000)
    p.add_argument("--eval-episodes", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=None, help="DQN/PPO minibatch (raise it to load the GPU)")
    p.add_argument("--buffer-size", type=int, default=100_000)
    p.add_argument("--gradient-steps", type=int, default=1)
    p.add_argument("--gpu-buffer", action="store_true",
                   help="store replay frames in VRAM (torch buffer) instead of ~GBs of system RAM")
    p.add_argument("--resume", default=None, help="path to a .zip model to continue from")
    p.add_argument("--tag", default="", help="suffix for the run dir (keeps comparisons separate)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--torch-threads", type=int, default=1)
    p.add_argument("--no-eval", action="store_true", help="skip periodic eval (fastest wall-clock)")
    args = p.parse_args()

    # leave the CPU cores to the emulators; the CNN batches are tiny
    th.set_num_threads(args.torch_threads)
    os.environ.setdefault("OMP_NUM_THREADS", str(args.torch_threads))

    env_id = build_game(args.game)
    steps = int(args.steps)

    if args.device == "cuda":
        if not th.cuda.is_available():
            raise SystemExit("CUDA not available — refusing to silently run on CPU.")
        gpu_name = th.cuda.get_device_name(0)
        bad = [k for k in ("intel", "uhd", "iris", "radeon", "arc", "microsoft basic") if k in gpu_name.lower()]
        print(f"GPU: {gpu_name} ({th.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB VRAM)", flush=True)
        if bad:
            raise SystemExit(f"Refusing to train on '{gpu_name}' (iGPU). Force the RTX in Windows graphics settings.")
    game_key = env_id.split("/")[1].rsplit("-", 1)[0].lower()
    run_name = f"{args.algo}{args.tag}"
    out = ROOT / "models" / game_key / run_name
    logs = ROOT / "logs" / game_key / run_name
    out.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    atari_kw = {"frameskip": 1, "repeat_action_probability": 0.0,
                "disable_env_checker": True, "max_episode_steps": None}  # ALE handles its own timeout
    vec_cls = SubprocVecEnv if (args.vec_env == "subproc" and args.n_envs > 1) else DummyVecEnv
    train_env = make_atari_env(env_id, n_envs=args.n_envs, seed=args.seed, vec_env_cls=vec_cls,
                               monitor_dir=str(logs / "monitor"), env_kwargs=atari_kw)
    eval_env = make_atari_env(env_id, n_envs=1, seed=args.seed + 123, monitor_dir=None, env_kwargs=atari_kw)
    # SB3's make_atari_env no longer stacks frames: do it explicitly (4 frames -> (4, 84, 84))
    train_env = VecTransposeImage(VecFrameStack(train_env, n_stack=4))
    eval_env = VecTransposeImage(VecFrameStack(eval_env, n_stack=4))
    print(f"env={env_id} obs={train_env.observation_space} actions={train_env.action_space.n}", flush=True)

    algo_cls = {"dqn": DQN, "ppo": PPO, "a2c": A2C}[args.algo]
    if args.resume:
        print(f"resuming from {args.resume}", flush=True)
        model = algo_cls.load(args.resume, env=train_env, device=args.device, tensorboard_log=str(logs))
        model.learning_rate = args.lr or model.learning_rate
    else:
        if args.algo == "dqn":
            buf_kw = {}
            buf_rows = args.buffer_size
            if args.gpu_buffer and args.device == "cuda":
                from replay_gpu import TorchReplayBuffer
                # SB3 buffers hold `buffer_size` rows x n_envs transitions per row.
                # --buffer-size is in transitions, so keep the VRAM footprint bounded.
                buf_rows = max(2000, args.buffer_size // args.n_envs)
                buf_kw = {"replay_buffer_class": TorchReplayBuffer}
                print(f"replay buffer: {buf_rows * args.n_envs:,} transitions in VRAM "
                      f"(~{buf_rows * args.n_envs * 4 * 84 * 84 / 1024**3:.2f} GB)", flush=True)
            model = DQN(
                "CnnPolicy", train_env, verbose=0, device=args.device, seed=args.seed,
                learning_rate=args.lr or 1e-4, buffer_size=buf_rows, learning_starts=20_000,
                batch_size=args.batch_size or 32, tau=1.0, gamma=0.99, train_freq=4, gradient_steps=args.gradient_steps,
                target_update_interval=1_000, exploration_fraction=0.10, exploration_final_eps=0.01,
                optimize_memory_usage=False, **buf_kw,
                tensorboard_log=str(logs),
            )
        elif args.algo == "ppo":
            model = PPO(
                "CnnPolicy", train_env, verbose=0, device=args.device, seed=args.seed,
                learning_rate=args.lr or 2.5e-4, n_steps=128, batch_size=args.batch_size or 256, n_epochs=4,
                gamma=0.99, gae_lambda=0.95, clip_range=0.1, ent_coef=0.01,
                tensorboard_log=str(logs),
            )
        else:
            model = A2C("CnnPolicy", train_env, verbose=0, device=args.device, seed=args.seed,
                        learning_rate=args.lr or 7e-4, n_steps=5, tensorboard_log=str(logs))

    callbacks = [
        SpeedCallback(every=max(steps // 40, 5_000)),
        CheckpointCallback(save_freq=max(100_000 // args.n_envs, 25_000), save_path=str(out / "ckpt"), name_prefix=args.algo),
    ]
    if not args.no_eval:
        callbacks.insert(1, EvalCallback(
            eval_env, best_model_save_path=str(out), log_path=str(logs / "eval"),
            eval_freq=max(args.eval_every // args.n_envs, 1), n_eval_episodes=args.eval_episodes,
            deterministic=True, render=False,
        ))

    t0 = time.time()
    model.learn(total_timesteps=steps, callback=callbacks, tb_log_name="run", progress_bar=False)
    final = out / "final.zip"
    model.save(final)
    meta = {"game": game_key, "env_id": env_id, "algo": args.algo, "steps": steps,
            "seconds": round(time.time() - t0, 1), "final_model": str(final)}
    (out / "train_meta.json").write_text(json.dumps(meta, indent=2))
    print("TRAINED", json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
