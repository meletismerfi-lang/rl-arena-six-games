"""Dump EnvPool's 84x84 observations alongside the raw 210x160 RGB frames they came from.

Two EnvPool instances with the same seed and the same action sequence stay in lockstep (measured
1490 == 1490 over 300 steps), so the RGB instance can act as ground truth for what the observation
instance actually saw. The dump lets a Windows-side pipeline be validated by exact byte comparison
instead of by score, which conflates preprocessing error with policy quality.
"""
from __future__ import annotations

import numpy as np
import envpool

STEPS = 48
SEED = 7
OUT = "/mnt/c/Users/melet/AI/rl-games/diag/envpool_obs_dump.npz"

obs_env = envpool.make_gymnasium("MsPacman-v5", num_envs=1, seed=SEED, episodic_life=True,
                                 stack_num=4, img_height=84, img_width=84)
rgb_env = envpool.make_gymnasium("MsPacman-v5", num_envs=1, seed=SEED, episodic_life=True,
                                 stack_num=1, img_height=210, img_width=160, gray_scale=False)

def unwrap(x):
    return x[0] if isinstance(x, tuple) else x

o = unwrap(obs_env.reset())
r = unwrap(rgb_env.reset())

frames, stacks, rewards = [], [], []
rng = np.random.default_rng(0)
for t in range(STEPS):
    a = np.array([int(rng.integers(0, 9))])          # any legal action; both envs get the same one
    out_o = obs_env.step(a)
    out_r = rgb_env.step(a)
    o, ro, do = out_o[0], out_o[1], out_o[2]
    r = out_r[0]
    rewards.append(float(ro[0]))
    frames.append(np.asarray(r)[0].transpose(1, 2, 0).copy())   # (210,160,3)
    stacks.append(np.asarray(o)[0].copy())                     # (4,84,84)
    if do[0]:
        break

np.savez_compressed(OUT, frames=np.stack(frames), stacks=np.stack(stacks),
                    rewards=np.asarray(rewards))
print(f"saved {len(frames)} steps -> {OUT}")
print(f"  frames {np.stack(frames).shape} uint8   stacks {np.stack(stacks).shape} uint8")
print(f"  total reward {sum(rewards):.0f}")
print(f"  first stack frame uniques: {np.unique(np.stack(stacks)[0][0])[:10]}")
