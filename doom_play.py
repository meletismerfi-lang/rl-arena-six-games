"""Record or watch a trained ViZDoom PPO agent.

  python doom_play.py --model models/doom/defend_the_center/ppo/best_model.zip --episodes 3
  python doom_play.py --model ... --live
"""
import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage

from doom_env import DoomEnv

ROOT = Path(__file__).resolve().parent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--scenario", default=None)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--live", action="store_true")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    model_path = Path(args.model)
    meta_file = model_path.parent / "train_meta.json"
    scenario = args.scenario
    if scenario is None:
        scenario = json.loads(meta_file.read_text())["scenario"] if meta_file.exists() else "defend_the_center"

    env = DoomEnv(scenario=scenario, render_mode="human" if args.live else None,
                  screen_format="RGB24", seed=args.seed)
    venv = VecTransposeImage(VecFrameStack(DummyVecEnv([lambda: env]), n_stack=4))
    model = PPO.load(model_path, env=venv, device=args.device)

    writer = None
    if not args.live:
        import imageio.v2 as imageio
        out_dir = ROOT / "videos" / "doom"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{scenario}_ppo_{model_path.stem}.mp4"

    results = []
    for ep in range(args.episodes):
        obs = venv.reset()
        done, total, steps, kills = False, 0.0, 0, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = venv.step(action)
            total += float(reward[0])
            steps += 1
            info = infos[0] if isinstance(infos, (list, tuple)) else {}
            kills = int(info.get("kills", kills))
            done = bool(dones[0])
            if not args.live and writer is None:
                h, w = env.rgb_frame().shape[:2]
                writer = imageio.get_writer(str(out_file), fps=args.fps, macro_block_size=1)
            if not args.live:
                import cv2
                fr = env.rgb_frame()
                if fr.size:
                    writer.append_data(cv2.resize(fr, (320, 240), interpolation=cv2.INTER_NEAREST))
        results.append(total)
        print(f"episode {ep + 1}: return={total:.1f} steps={steps}")

    if writer is not None:
        writer.close()
        print("video:", out_file)
    print("returns:", results, "mean:", sum(results) / len(results))
    venv.close()


if __name__ == "__main__":
    main()
