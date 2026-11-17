"""Measure ViZDoom env throughput at different parallel env counts (random policy)."""
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from doom_env import DoomEnv


def factory(rank):
    return lambda: DoomEnv(scenario="defend_the_center", frame_skip=4, seed=rank)


def bench(n_envs: int, steps_per_env: int = 600, subproc: bool = True) -> float:
    fns = [factory(i) for i in range(n_envs)]
    venv = SubprocVecEnv(fns) if (subproc and n_envs > 1) else DummyVecEnv(fns)
    venv.reset()
    acts = [venv.action_space.sample() for _ in range(n_envs)]
    for _ in range(100):  # warmup: WAD/render caches, thread pools
        venv.step(acts)
    t0 = time.time()
    for _ in range(steps_per_env):
        venv.step(acts)
    dt = time.time() - t0
    venv.close()
    return n_envs * steps_per_env / dt


if __name__ == "__main__":
    for n in [1, 2, 3, 4, 6, 8]:
        try:
            sps = bench(n, 600)
            print(f"{n:>3} envs: {sps:8.0f} steps/s", flush=True)
        except Exception as e:
            print(f"{n:>3} envs: FAILED {e}", flush=True)
