"""GPU-resident RL: MJX/Brax simulation + PPO training, both on the NVIDIA GPU via JAX/XLA.

Nothing leaves the GPU: batched env state, rollouts, and gradients are all XLA buffers
in VRAM. (Compare: Atari/ViZDoom step one CPU core per env instance.)

Run inside WSL (JAX has no official Windows GPU wheels):
  wsl -d jaxgpu -- /root/gpurl/bin/python /mnt/c/Users/melet/AI/rl-games/brax_ppo.py \
      --env ant --envs 4096 --steps 10000000

Subcommands / flags:
  --bench          only measure GPU env throughput
  --env            brax env name: ant, humanoid, halfcheetah, hopper, walker2d, reacher, ...
  --envs           parallel environments resident in VRAM
  --steps          total PPO timesteps
"""
import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp

OUT = Path("/mnt/c/Users/melet/AI/rl-games/models/brax")


def check_devices() -> None:
    devs = jax.devices()
    print(f"jax {jax.__version__} | devices: {devs}", flush=True)
    if not any(d.platform == "gpu" for d in devs):
        raise SystemExit("No GPU backend in JAX — refusing to run this as a CPU job.")
    for d in devs:
        if d.platform == "gpu":
            print(f"gpu: {d.device_kind}", flush=True)


def bench_env(env_name: str, n_envs: int, steps: int = 200) -> float:
    """Pure-GPU throughput: batched reset + step of N envs entirely in XLA."""
    from brax import envs

    env = envs.get_environment(env_name, backend="mjx")
    reset = jax.jit(jax.vmap(env.reset))
    step = jax.jit(jax.vmap(env.step))
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, n_envs)
    state = reset(keys)
    act = jnp.zeros((n_envs, env.action_size))
    state = step(state, act)  # warmup/compile
    jax.block_until_ready(state.obs)

    t0 = time.time()
    for _ in range(steps):
        state = step(state, act)
    jax.block_until_ready(state.obs)
    dt = time.time() - t0
    sps = n_envs * steps / dt
    print(f"bench {env_name}: {n_envs} envs x {steps} steps in {dt:.2f}s = {sps:,.0f} env steps/s", flush=True)
    return sps


def train(env_name: str, n_envs: int, timesteps: int, seed: int = 0) -> dict:
    from brax import envs
    from brax.training.agents.ppo import train as ppo

    env = envs.get_environment(env_name, backend="mjx")
    OUT.mkdir(parents=True, exist_ok=True)

    times = []
    rewards = []

    def progress(num_steps, metrics):
        times.append((num_steps, time.time()))
        rewards.append((num_steps, float(metrics.get("eval/episode_reward", float("nan")))))
        if len(times) % 10 == 0 or len(times) <= 3:
            print(f"[{num_steps:>10,}] eval_reward={rewards[-1][1]:9.2f}", flush=True)

    t0 = time.time()
    make_inference_fn, params, metrics = ppo.train(
        environment=env,
        num_timesteps=timesteps,
        num_envs=n_envs,
        episode_length=1000,
        batch_size=min(1024, n_envs),
        num_minibatches=32,
        num_updates_per_batch=4,
        learning_rate=3e-4,
        entropy_cost=1e-2,
        discounting=0.97,
        unroll_length=20,
        num_evals=10,
        normalize_observations=True,
        reward_scaling=0.1,
        seed=seed,
        progress_fn=progress,
    )
    dt = time.time() - t0
    result = {
        "env": env_name,
        "n_envs": n_envs,
        "timesteps": timesteps,
        "seconds": round(dt, 1),
        "steps_per_s": round(timesteps / dt, 1),
        "final_eval_reward": rewards[-1][1] if rewards else None,
        "first_eval_reward": rewards[0][1] if rewards else None,
    }
    tag = f"{env_name}_{n_envs}env_{timesteps}steps"
    (OUT / f"{tag}.json").write_text(json.dumps({**result, "curve": rewards}, indent=2))
    print("TRAINED", json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="ant")
    p.add_argument("--envs", type=int, default=4096)
    p.add_argument("--steps", type=float, default=10_000_000)
    p.add_argument("--bench", action="store_true")
    p.add_argument("--bench-sweep", action="store_true", help="env-count sweep 1024..16384")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    check_devices()
    if args.bench_sweep:
        for n in [1024, 2048, 4096, 8192, 16384]:
            try:
                bench_env(args.env, n)
            except Exception as e:  # OOM at large n is expected on 8 GB
                print(f"bench {args.env} {n}: failed ({type(e).__name__}: {str(e)[:120]})", flush=True)
    elif args.bench:
        bench_env(args.env, args.envs)
    else:
        train(args.env, args.envs, int(args.steps), args.seed)
