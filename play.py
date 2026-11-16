"""Record (or watch live) a trained Atari agent playing.

  python play.py --model models/pong/dqn/best.zip --episodes 2
  python play.py --model models/pong/dqn/best.zip --live
"""
import argparse
import os
from pathlib import Path

# Pin the discrete NVIDIA GPU before CUDA initialises.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import ale_py
import gymnasium as gym
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage

gym.register_envs(ale_py)
ROOT = Path(__file__).resolve().parent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--live", action="store_true", help="open a window instead of recording mp4")
    p.add_argument("--out", default=None)
    p.add_argument("--env-id", default=None, help="override, e.g. ALE/MsPacman-v5")
    p.add_argument("--envpool-compat", action="store_true",
                   help="match EnvPool preprocessing (no noop-reset, no FIRE at reset)")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    model_path = Path(args.model)
    meta_path = model_path.parent / "train_meta.json"
    env_id = args.env_id
    if env_id is None and meta_path.exists():
        import json
        env_id = json.loads(meta_path.read_text())["env_id"]
    if env_id is None:
        # fall back to <...>/models/<game>/<run>/file.zip
        parts = model_path.resolve().parts
        if "models" in parts:
            from train import build_game
            env_id = build_game(parts[parts.index("models") + 1])
        else:
            raise SystemExit("cannot infer env — pass --env-id ALE/MsPacman-v5")
    game = env_id.split("/")[1].rsplit("-", 1)[0].lower()

    videos = Path(args.out) if args.out else ROOT / "videos" / game
    videos.mkdir(parents=True, exist_ok=True)

    def make_venv(seed: int):
        if args.live:
            e = gym.make(env_id, frameskip=1, repeat_action_probability=0.0, render_mode="human")
        else:
            e = gym.make(env_id, frameskip=1, repeat_action_probability=0.0,
                         render_mode="rgb_array", max_episode_steps=20_000)
            e = gym.wrappers.RecordVideo(e, video_folder=str(videos), episode_trigger=lambda ep: True,
                                         name_prefix=f"{game}_seed{seed}_{model_path.parent.parent.name}_{model_path.stem}")
            e.reset(seed=seed)
        if args.envpool_compat:
            from gymnasium.wrappers import AtariPreprocessing, FrameStack
            e = AtariPreprocessing(e, frame_skip=4, screen_size=84, grayscale_obs=True,
                                   grayscale_newaxis=False, scale_obs=False, terminal_on_life_loss=False)
            e = FrameStack(e, 4)
            return DummyVecEnv([lambda: e])  # FrameStack already gives (4, 84, 84)
        e = AtariWrapper(e, frame_skip=4, clip_reward=False, terminal_on_life_loss=False)
        return VecTransposeImage(VecFrameStack(DummyVecEnv([lambda: e]), n_stack=4))

    cls = {"dqn": DQN, "ppo": PPO, "a2c": A2C}
    algo = "dqn"
    for k in cls:
        if k in str(model_path).lower():
            algo = k
    venv = make_venv(args.seed)
    model = cls[algo].load(model_path, env=venv, device="cpu")
    import torch as th
    model.replay_buffer = None            # inference needs no replay buffer (it is GBs of VRAM)
    model.policy.to(args.device)
    th.cuda.empty_cache() if th.cuda.is_available() else None

    scores = []
    for ep in range(args.episodes):
        venv.close()                      # fresh env per episode -> different ALE seed
        venv = make_venv(args.seed + ep)
        obs = venv.reset()
        done, total, steps = False, 0.0, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = venv.step(action)
            total += float(reward[0])
            steps += 1
            done = bool(dones[0])
        scores.append(total)
        print(f"episode {ep + 1}: reward={total:.1f} steps={steps}")
    venv.close()
    print("scores:", scores, "mean:", sum(scores) / len(scores))
    if not args.live:
        import json
        f = videos / "scores.json"
        data = json.loads(f.read_text()) if f.exists() else {}
        data[str(model_path)] = {"scores": scores, "mean": sum(scores) / len(scores),
                                 "n": len(scores), "episodes_steps": None}
        f.write_text(json.dumps(data, indent=2))
        print("videos:", videos)


if __name__ == "__main__":
    main()
