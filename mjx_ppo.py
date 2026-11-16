"""GPU-resident PPO: MJX physics + training entirely in VRAM, no CPU env loop.

Brax supplies the MJX env definitions (ant, humanoid, halfcheetah, ...); the PPO
trainer here is self-contained JAX so it works with current JAX instead of brax's
stale training agents.

  python mjx_ppo.py --env ant --envs 4096 --steps 20000000
  python mjx_ppo.py --env ant --envs 4096 --steps 2000000 --bench
"""
import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import numpy as np

OUT = Path("/mnt/c/Users/melet/AI/rl-games/models/mjx")
OUT_WIN = Path("models/mjx")


# ----------------------------------------------------------------------------- model
def init_mlp(key, sizes, scale=1.0):
    params = []
    for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
        key, sub = jax.random.split(key)
        w = jax.random.normal(sub, (n_in, n_out)) * jnp.sqrt(2.0 / n_in)
        params.append({"w": w * (0.01 if i == len(sizes) - 2 else 1.0), "b": jnp.zeros(n_out)})
    return params


def mlp(params, x):
    for i, layer in enumerate(params):
        x = x @ layer["w"] + layer["b"]
        if i < len(params) - 1:
            x = jnp.tanh(x)
    return x


def init_params(key, obs_dim, act_dim, hidden=(256, 256)):
    k1, k2 = jax.random.split(key)
    pi = init_mlp(k1, (obs_dim, *hidden, act_dim))
    vf = init_mlp(k2, (obs_dim, *hidden, 1))
    return {"pi": pi, "vf": vf, "log_std": jnp.zeros(act_dim)}


def policy_logits(params, obs):
    return mlp(params["pi"], obs)


def value(params, obs):
    return mlp(params["vf"], obs)[..., 0]


def sample_action(params, obs, key):
    mean = policy_logits(params, obs)
    std = jnp.exp(jnp.clip(params["log_std"], -5.0, 2.0))
    action = mean + std * jax.random.normal(key, mean.shape)
    logp = gaussian_logprob(action, mean, std)
    return jnp.tanh(action), logp


def gaussian_logprob(action, mean, std):
    z = (action - mean) / std
    return jnp.sum(-0.5 * z**2 - jnp.log(std) - 0.5 * jnp.log(2 * jnp.pi), axis=-1)


# ----------------------------------------------------------------------------- rollout
def make_rollout(env, unroll: int, num_envs: int):
    step_v = jax.vmap(env.step)
    reset_v = jax.vmap(env.reset)

    def rollout(params, states, key, stats):
        def body(carry, _):
            states, key = carry
            key, k = jax.random.split(key)
            obs = normalize(stats, states.obs)
            keys = jax.random.split(k, num_envs)
            mean = policy_logits(params, obs)
            std = jnp.exp(jnp.clip(params["log_std"], -5.0, 2.0))
            z = jax.vmap(lambda kk: jax.random.normal(kk, mean.shape[1:]))(keys)
            act = mean + std * z
            logp = gaussian_logprob(act, mean, std)  # std broadcasts over envs
            val = value(params, obs)
            new_states = step_v(states, jnp.tanh(act))
            trans = {
                "obs": obs,
                "obs_raw": states.obs,
                "action": act,
                "logp": logp,
                "value": val,
                "reward": new_states.reward,
                "done": new_states.done.astype(jnp.float32),
            }
            return (new_states, key), trans

        (states, _), traj = jax.lax.scan(body, (states, key), None, length=unroll)
        return states, traj

    return rollout, reset_v


# ----------------------------------------------------------------------------- GAE + PPO
def compute_gae(traj, last_value, gamma=0.99, lam=0.95):
    rewards = traj["reward"]
    values = traj["value"]
    dones = traj["done"]

    def body(carry, xs):
        adv_next, next_val = carry
        r, v, d = xs
        next_val = next_val * (1.0 - d)
        delta = r + gamma * next_val - v
        adv = delta + gamma * lam * (1.0 - d) * adv_next
        return (adv, v), adv

    _, advs = jax.lax.scan(body, (jnp.zeros_like(last_value), last_value), (rewards, values, dones), reverse=True)
    returns = advs + values
    return advs, returns


def ppo_loss(params, batch, clip=0.2, vf_coef=0.5, ent_coef=0.0):
    obs = batch["obs"]
    mean = policy_logits(params, obs)  # mlp handles the leading batch dim
    std = jnp.exp(jnp.clip(params["log_std"], -5.0, 2.0))
    new_logp = gaussian_logprob(batch["action"], mean, std)  # std broadcasts
    ratio = jnp.exp(new_logp - batch["logp"])
    adv = batch["advantage"]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    pg = -jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - clip, 1 + clip) * adv).mean()
    v = value(params, obs)
    v_loss = jnp.square(v - batch["return"]).mean()
    entropy = 0.5 * jnp.log(2 * jnp.pi * jnp.e) + jnp.sum(jnp.clip(params["log_std"], -5.0, 2.0))
    total = pg + vf_coef * v_loss - ent_coef * entropy
    return total, {"pg": pg, "vf": v_loss, "entropy": entropy, "kl": jnp.mean(batch["logp"] - new_logp)}


def flatten_traj(traj, advs, rets):
    def flat(x):
        return x.reshape((-1,) + x.shape[2:])
    return {
        "obs": flat(traj["obs"]),
        "action": flat(traj["action"]),
        "logp": flat(traj["logp"]),
        "advantage": flat(advs),
        "return": flat(rets),
    }


# --------------------------------------------------------------------------- obs norm
def init_stats(obs_dim: int):
    return {"mean": jnp.zeros(obs_dim), "var": jnp.ones(obs_dim)}


def update_stats(stats, obs, decay=0.99):
    b_mean = obs.mean(axis=0)
    b_var = obs.var(axis=0)
    return {
        "mean": decay * stats["mean"] + (1 - decay) * b_mean,
        "var": decay * stats["var"] + (1 - decay) * b_var,
    }


def normalize(stats, obs):
    return jnp.clip((obs - stats["mean"]) * jax.lax.rsqrt(stats["var"] + 1e-4), -10.0, 10.0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="ant")
    p.add_argument("--envs", type=int, default=4096)
    p.add_argument("--steps", type=float, default=10_000_000)
    p.add_argument("--unroll", type=int, default=16)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatches", type=int, default=8)
    p.add_argument("--minibatch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--reward-scale", type=float, default=1.0)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--kl-stop", type=float, default=0.05, help="skip update if KL exceeds this")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bench", action="store_true", help="only measure throughput")
    args = p.parse_args()

    devs = jax.devices()
    print(f"jax {jax.__version__} devices={devs}", flush=True)
    if not any(d.platform == "gpu" for d in devs):
        raise SystemExit("no GPU backend — refusing to run this on CPU")
    for d in devs:
        if d.platform == "gpu":
            print("gpu:", d.device_kind, flush=True)

    from brax import envs
    env = envs.get_environment(args.env, backend="mjx")
    obs_dim = env.observation_size
    act_dim = env.action_size
    print(f"env={args.env} obs={obs_dim} act={act_dim} envs={args.envs} unroll={args.unroll}", flush=True)

    key = jax.random.PRNGKey(args.seed)
    key, k_init, k_reset = jax.random.split(key, 3)
    params = init_params(k_init, obs_dim, act_dim)
    optimizer = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(args.lr))
    opt_state = optimizer.init(params)

    rollout, reset_v = make_rollout(env, args.unroll, args.envs)
    states = reset_v(jax.random.split(k_reset, args.envs))

    @jax.jit
    def train_step(params, opt_state, states, key, stats):
        states, traj = rollout(params, states, key, stats)
        traj["reward"] = traj["reward"] * args.reward_scale
        last_val = value(params, normalize(stats, states.obs))
        advs, rets = compute_gae(traj, last_val, args.gamma, args.lam)
        batch = flatten_traj(traj, advs, rets)
        new_stats = update_stats(stats, traj["obs_raw"].reshape(-1, obs_dim))
        n_total = traj["reward"].size
        mb = min(args.minibatch_size, n_total)
        n_mb = max(1, n_total // mb)

        def loss_fn(p, b):
            return ppo_loss(p, b, clip=0.2, vf_coef=0.5, ent_coef=args.ent_coef)

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

        def mb_body(carry, idxs):
            p, os_ = carry
            b = jax.tree.map(lambda x: x[idxs], batch)
            (loss, aux), grads = grad_fn(p, b)
            # KL safety valve: skip the update if the policy moved too far
            def do_update(operand):
                p, os_, grads = operand
                updates, os_ = optimizer.update(grads, os_, p)
                return optax.apply_updates(p, updates), os_
            def no_update(operand):
                p, os_, _ = operand
                return p, os_
            p, os_ = jax.lax.cond(aux["kl"] > args.kl_stop, no_update, do_update, (p, os_, grads))
            return (p, os_), (loss, aux)

        def epoch_body(carry, key_e):
            p, os_ = carry
            perm = jax.random.permutation(key_e, n_total)[: n_mb * mb].reshape((n_mb, mb))
            (p, os_), outs = jax.lax.scan(mb_body, (p, os_), perm)
            return (p, os_), outs

        epoch_keys = jax.random.split(key, args.epochs)
        (params, opt_state), (losses, auxs) = jax.lax.scan(epoch_body, (params, opt_state), epoch_keys)
        metrics = {
            "reward": traj["reward"].mean(),
            "loss": losses[-1, -1],
            "kl": auxs["kl"][-1, -1],
            "entropy": auxs["entropy"][-1, -1],
            "updates": n_mb * args.epochs,
        }
        return params, opt_state, states, new_stats, metrics

    batch_steps = args.envs * args.unroll
    n_iters = max(1, int(args.steps // batch_steps))
    print(f"batch = {batch_steps:,} env steps/iter, iters = {n_iters:,}", flush=True)

    # warmup (compilation)
    stats = init_stats(obs_dim)
    t0 = time.time()
    params, opt_state, states, stats, m = train_step(params, opt_state, states, key, stats)
    jax.block_until_ready(m["reward"])
    print(f"compiled in {time.time() - t0:.1f}s", flush=True)
    if args.bench:
        t0 = time.time()
        for _ in range(10):
            key, k = jax.random.split(key)
            params, opt_state, states, stats, m = train_step(params, opt_state, states, k, stats)
        jax.block_until_ready(m["reward"])
        dt = time.time() - t0
        print(f"throughput: {10 * batch_steps / dt:,.0f} env steps/s (PPO included)", flush=True)
        return

    t0 = time.time()
    curve = []
    for it in range(n_iters):
        key, k = jax.random.split(key)
        params, opt_state, states, stats, m = train_step(params, opt_state, states, k, stats)
        if it % max(1, n_iters // 20) == 0 or it == n_iters - 1:
            m = jax.tree.map(lambda x: float(x), m)
            el = time.time() - t0
            done = (it + 1) * batch_steps
            curve.append({"steps": done, "reward": m["reward"], "kl": m["kl"]})
            print(f"[{done:>10,}] reward={m['reward']:8.2f} kl={m['kl']:+.4f} "
                  f"{done / max(el, 1e-9):,.0f} steps/s", flush=True)

    dt = time.time() - t0
    result = {"env": args.env, "envs": args.envs, "steps": int(n_iters * batch_steps),
              "seconds": round(dt, 1), "steps_per_s": round(n_iters * batch_steps / dt, 1),
              "final_reward": curve[-1]["reward"] if curve else None, "curve": curve}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.env}_{args.envs}env.json").write_text(json.dumps(result, indent=2))
    # keep the weights so the policy can be replayed/rendered later
    import pickle
    weights = jax.tree.map(lambda x: np.asarray(x), params)
    with open(OUT / f"{args.env}_{args.envs}env_params.pkl", "wb") as fh:
        pickle.dump({"params": weights, "env": args.env, "obs_dim": obs_dim, "act_dim": act_dim,
                     "stats": jax.tree.map(lambda x: np.asarray(x), stats)}, fh)
    print("TRAINED", json.dumps({k: v for k, v in result.items() if k != "curve"}), flush=True)


if __name__ == "__main__":
    main()
