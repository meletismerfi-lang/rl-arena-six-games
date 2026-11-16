"""Compare PPO update paths: baseline fp32 vs channels_last vs torch.compile vs bf16 autocast."""
import argparse
import sys
import time
from pathlib import Path

import torch as th
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from envpool_vec import EnvPoolAtari  # noqa: E402


def build(envs=64, n_steps=128, batch=1024, epochs=3):
    env = EnvPoolAtari("MsPacman-v5", num_envs=envs, seed=3)
    model = PPO("CnnPolicy", env, device="cuda", n_steps=n_steps, batch_size=batch, n_epochs=epochs, verbose=0)
    model.learn(total_timesteps=envs * n_steps)
    return model


def time_train(model, reps=5):
    model.train()
    th.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps):
        model.train()
    th.cuda.synchronize()
    return (time.time() - t0) / reps * 1000


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="base", choices=["base", "channels_last", "compile", "bf16"])
    args = p.parse_args()

    model = build()
    fe = model.policy.features_extractor
    name = args.variant

    if name == "channels_last":
        fe.cnn = fe.cnn.to(memory_format=th.channels_last)
        fe.linear = fe.linear
    elif name == "compile":
        fe.cnn = th.compile(fe.cnn, mode="max-autotune")
    elif name == "bf16":
        orig = model.policy.forward

        def fwd(obs, deterministic=False, episode_start=None):
            with th.autocast("cuda", dtype=th.bfloat16):
                return orig(obs, deterministic, episode_start)
        model.policy.forward = fwd

    ms = time_train(model)
    print(f"{name:14s} train() = {ms:7.1f} ms   ({ms / 24 * 1000:.1f} ms per minibatch update)", flush=True)


if __name__ == "__main__":
    main()
