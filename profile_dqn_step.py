"""Where does a DQN vec-step go? Time predict, env.step, and buffer.add separately.

Run inside WSL (needs envpool):  /root/gpurl/bin/python profile_dqn_step.py --envs 32
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch as th
from stable_baselines3 import DQN

sys.path.insert(0, str(Path(__file__).resolve().parent))
from envpool_vec import EnvPoolAtari  # noqa: E402
from replay_gpu import TorchReplayBuffer  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--envs", type=int, default=32)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--model", default="/mnt/c/Users/melet/AI/rl-games/models/mspacman/dqn_envpool/final_4000000.zip")
    args = p.parse_args()

    th.set_num_threads(1)
    env = EnvPoolAtari("MsPacman-v5", num_envs=args.envs, seed=0, reward_clip=False)
    model = DQN.load(args.model, env=env, device="cuda")
    buf = TorchReplayBuffer(50000 // args.envs, env.observation_space, env.action_space,
                            device="cuda", n_envs=args.envs)
    obs = env.reset()

    def timeit(tag, fn, n=args.iters):
        for _ in range(5):
            fn()
        th.cuda.synchronize()
        t0 = time.time()
        for _ in range(n):
            fn()
        th.cuda.synchronize()
        ms = (time.time() - t0) / n * 1000
        print(f"{tag:34s} {ms:6.2f} ms/vec-step   ({args.envs / (ms / 1000):,.0f} steps/s)", flush=True)
        return ms

    acts = np.zeros(args.envs, dtype=np.int32)
    t_env = timeit("env.step only", lambda: env.step(acts))
    t_pred = timeit("model.predict only", lambda: model.predict(obs, deterministic=False))
    dummy_next = obs
    t_add = timeit("replay.add only", lambda: buf.add(obs, dummy_next, acts, np.zeros(args.envs), np.zeros(args.envs), [{}] * args.envs))

    def full():
        a, _ = model.predict(obs, deterministic=False)
        o, r, d, i = env.step(a)
        buf.add(obs, o, a, r, d, i)
    t_full = timeit("predict + step + add (the real loop)", full)
    print(f"\nserial sum: {t_env + t_pred + t_add:.2f} ms | measured loop: {t_full:.2f} ms")


if __name__ == "__main__":
    main()
