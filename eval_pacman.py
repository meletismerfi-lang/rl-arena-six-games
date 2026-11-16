"""Evaluate a trained policy inside EnvPool (its own training env), real unclipped scores."""
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from envpool_vec import EnvPoolAtari  # noqa: E402


def main() -> None:
    model_path = sys.argv[1] if len(sys.argv) > 1 else "/mnt/c/Users/melet/AI/rl-games/models/mspacman/ppo_envpool/final.zip"
    n_eps = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    env = EnvPoolAtari("MsPacman-v5", num_envs=8, seed=123, reward_clip=False)
    try:
        model = PPO.load(model_path, env=env, device="cuda")
    except TypeError:
        from stable_baselines3 import DQN
        model = DQN.load(model_path, env=env, device="cuda")

    obs = env.reset()
    scores = np.zeros(8)
    finished = []
    steps = 0
    while len(finished) < n_eps:
        actions, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(actions)
        scores += rewards
        steps += 1
        for i, d in enumerate(dones):
            if d:
                finished.append(float(scores[i]))
                scores[i] = 0.0
        if steps > 20000:
            break
    env.close()
    print("episodes:", len(finished))
    print("scores:", [round(s) for s in finished])
    print(f"mean={np.mean(finished):.1f} median={np.median(finished):.1f} max={np.max(finished):.0f}")


if __name__ == "__main__":
    main()
