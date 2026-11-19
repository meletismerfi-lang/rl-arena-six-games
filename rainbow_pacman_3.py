"""Rainbow-lite for Ms. Pac-Man: Dueling + C51 + Prioritised replay + N-step + Double-Q.

Not full Rainbow (no noisy nets) and not ApeX (no distributed actors). A single learner on one
laptop gains little from the actor/learner split, and noisy nets overlap heavily with
epsilon-greedy at this budget.

Reward handling: raw Ms. Pac-Man rewards run 10..1600 and C51 needs bounded support. `--reward-
transform squash` applies sign(x)(sqrt(|x|+1)-1) (R2D2-style), which turns 10 -> 2.32 and 1600 ->
39.0. Note what that costs: the pellet-to-ghost-chain ratio drops from 160x (linear) to 16.8x, so
squashing preserves ordering but still damps the incentive to hunt ghosts. `scale` keeps 160x and
is available for a like-for-like comparison.

  python rainbow_pacman.py --steps 4000000 --envs 32
"""
from __future__ import annotations

import argparse
import os
import time
from collections import deque
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ network
class NoisyLinear(nn.Module):
    """Factorized Gaussian noise layer (Fortunato et al.). Replaces epsilon-greedy: the noise is
    in the weights, so exploration is state-dependent instead of uniform action jitter."""

    def __init__(self, in_features: int, out_features: int, sigma0: float = 0.5):
        super().__init__()
        self.in_f, self.out_f = in_features, out_features
        self.sigma0 = sigma0
        self.weight_mu = nn.Parameter(th.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(th.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(th.empty(out_features))
        self.bias_sigma = nn.Parameter(th.empty(out_features))
        self.register_buffer("weight_eps", th.zeros(out_features, in_features))
        self.register_buffer("bias_eps", th.zeros(out_features))
        bound = 1.0 / in_features ** 0.5
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.weight_sigma, sigma0 * bound)
        nn.init.constant_(self.bias_sigma, sigma0 * bound)

    def reset_noise(self) -> None:
        """Factorised Gaussian noise: f(x) = sign(x) * sqrt(|x|), weight noise = f(eps_out) x f(eps_in).

        Using sign(x)*|x| (i.e. the raw Gaussian) instead of sign(x)*sqrt(|x|) inflates every
        weight's noise to ~N(0,1) regardless of fan-in, which swamps sigma/sqrt(fan_in) and stops
        the network learning -- that is what the first NoisyLinear run did (best eval 210 at 5M).
        """
        eps_in = th.randn(self.in_f, device=self.weight_mu.device)
        eps_out = th.randn(self.out_f, device=self.weight_mu.device)
        f_in = eps_in.sign() * eps_in.abs().sqrt()
        f_out = eps_out.sign() * eps_out.abs().sqrt()
        self.weight_eps.copy_(f_out[:, None] * f_in[None, :])
        self.bias_eps.copy_(f_out)

    def forward(self, x: th.Tensor) -> th.Tensor:
        w = self.weight_mu + self.weight_sigma * self.weight_eps
        b = self.bias_mu + self.bias_sigma * self.bias_eps
        return F.linear(x, w, b)


class ImpalaBlock(nn.Module):
    """Residual block from IMPALA: two 3x3 convs plus a skip, then max-pool."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        return F.max_pool2d(h + self.skip(x), 3, stride=2, padding=1)


class DuelingC51(nn.Module):
    def __init__(self, in_ch: int = 4, n_actions: int = 9, atoms: int = 51, width: int = 64,
                 trunk: str = "nature", noisy: bool = False):
        super().__init__()
        self.atoms = atoms
        self.n_actions = n_actions
        self.noisy = noisy
        if trunk == "impala":
            self.trunk = nn.Sequential(
                ImpalaBlock(in_ch, 16), ImpalaBlock(16, 32), ImpalaBlock(32, 32), nn.Flatten())
            feat = 32 * 11 * 11          # 84 -> 42 -> 21 -> 11
        else:
            self.trunk = nn.Sequential(
                nn.Conv2d(in_ch, 32, 8, 4), nn.ReLU(),
                nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
                nn.Conv2d(64, width, 3, 1), nn.ReLU(),
                nn.Flatten(),
            )
            feat = width * 7 * 7
        head = (lambda i, o: NoisyLinear(i, o)) if noisy else (lambda i, o: nn.Linear(i, o))
        self.value = nn.Sequential(head(feat, 512), nn.ReLU(), head(512, atoms))
        self.adv = nn.Sequential(head(feat, 512), nn.ReLU(), head(512, n_actions * atoms))

    def reset_noise(self) -> None:
        if not self.noisy:
            return
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

    def noise_off(self) -> None:
        """Zero the noise buffers: evaluation should use the mean weights, not a noisy sample."""
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.weight_eps.zero_()
                m.bias_eps.zero_()

    def forward(self, x: th.Tensor) -> th.Tensor:
        h = self.trunk(x.float() / 255.0)
        v = self.value(h).view(-1, 1, self.atoms)
        a = self.adv(h).view(-1, self.n_actions, self.atoms)
        return v + a - a.mean(dim=1, keepdim=True)          # (B, A, atoms) logits

    def q(self, x: th.Tensor, support: th.Tensor) -> th.Tensor:
        return (self(x).softmax(-1) * support).sum(-1)      # expected Q per action


def project_distribution(next_dist, rewards, dones, support, gamma_n, vmin, vmax):
    """C51 projection of r + gamma^n * z onto the fixed support.

    The eq_mask term matters: when the shifted value lands exactly on an atom, lo == hi and both
    interpolation weights are zero, so the mass would silently vanish and the network would be
    trained to predict an empty distribution.
    """
    n_atoms = next_dist.shape[1]
    delta = (vmax - vmin) / (n_atoms - 1)
    tz = (rewards[:, None] + gamma_n * (1.0 - dones[:, None]) * support[None, :]).clamp(vmin, vmax)
    pos = (tz - vmin) / delta
    lo = pos.floor().long().clamp(0, n_atoms - 1)
    hi = pos.ceil().long().clamp(0, n_atoms - 1)
    eq = (hi == lo).float()
    out = th.zeros_like(next_dist)
    out.scatter_add_(1, lo, next_dist * (hi.float() - pos + eq))
    out.scatter_add_(1, hi, next_dist * (pos - lo.float()))
    return out


# ------------------------------------------------------------------ prioritised replay
class PrioritisedReplay:
    """Proportional prioritised replay.

    The cumulative array is rebuilt lazily every `rebuild_every` samples instead of on every add:
    rebuilding over 200k rows per step would dominate the loop, and a stale window of a couple of
    thousand samples changes sampling only marginally. The priority sum and max are cached so
    neither `add` nor `sample` walks the whole array.
    """

    def __init__(self, capacity: int, obs_shape, device: str,
                 alpha: float = 0.6, beta0: float = 0.4, beta_steps: int = 2_000_000,
                 rebuild_every: int = 2_000, shared_frames: bool = False):
        self.capacity = capacity
        # A stack of k frames (k, H, W) and its n-step successor share k-1 frames, so store one
        # (k+1)-frame window instead of two k-frame stacks: obs = w[:k], next_obs = w[1:k+1].
        # 4-frame stacks cost 5/8 of the naive layout -- 500k transitions fell from 26.3 GB to 17.6 GB.
        self.shared = shared_frames
        if shared_frames:
            self.win = np.zeros((capacity, obs_shape[0] + 1, *obs_shape[1:]), dtype=np.uint8)
            self.obs = self.win
            self.next_obs = self.win
        else:
            self.obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
            self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.prio = np.ones(capacity, dtype=np.float32)
        self.pos = 0
        self.written = 0
        self.alpha, self.beta0, self.beta_steps = alpha, beta0, beta_steps
        self.device = device
        self._cum = None
        self._psum = 1.0
        self._max = 1.0
        self._counter = 10**9          # force a rebuild on first sample
        self.rebuild_every = rebuild_every

    def __len__(self):
        return min(self.written, self.capacity)

    def add(self, obs, next_obs, action, reward, done):
        k = obs.shape[1]
        for i in range(len(obs)):
            j = (self.pos + i) % self.capacity
            if self.shared:
                self.win[j, :k] = obs[i]
                self.win[j, k] = next_obs[i][-1]
            else:
                self.obs[j] = obs[i]
                self.next_obs[j] = next_obs[i]
            self.actions[j] = action[i]
            self.rewards[j] = reward[i]
            self.dones[j] = done[i]
            self.prio[j] = self._max
        self.pos = (self.pos + len(obs)) % self.capacity
        self.written += len(obs)

    def _rebuild(self):
        n = len(self)
        p = self.prio[:n] ** self.alpha
        self._psum = float(p.sum())
        self._cum = np.cumsum(p) / max(self._psum, 1e-9)
        self._counter = 0

    def sample(self, batch: int, step: int):
        n = len(self)
        if self._cum is None or self._counter >= self.rebuild_every:
            self._rebuild()
        self._counter += 1
        idx = np.searchsorted(self._cum, np.random.random(batch)).clip(0, n - 1)
        beta = min(1.0, self.beta0 + (1.0 - self.beta0) * step / self.beta_steps)
        p = (self.prio[idx] ** self.alpha) / max(self._psum, 1e-9)
        w = (n * p) ** (-beta)
        return idx, (w / w.max()).astype(np.float32)

    def batch(self, idx):
        d = "cuda" if th.cuda.is_available() else "cpu"
        if self.shared:
            k = self.win.shape[1] - 1
            w = self.win[idx]
            return dict(
                obs=th.as_tensor(w[:, :k], device=d),
                next_obs=th.as_tensor(w[:, 1:], device=d),
                actions=th.as_tensor(self.actions[idx], device=d),
                rewards=th.as_tensor(self.rewards[idx], device=d),
                dones=th.as_tensor(self.dones[idx], device=d),
            )
        return dict(
            obs=th.as_tensor(self.obs[idx], device=d),
            next_obs=th.as_tensor(self.next_obs[idx], device=d),
            actions=th.as_tensor(self.actions[idx], device=d),
            rewards=th.as_tensor(self.rewards[idx], device=d),
            dones=th.as_tensor(self.dones[idx], device=d),
        )

    def update_priorities(self, idx, td: np.ndarray):
        p = np.abs(td) + 1e-6
        self.prio[idx] = p
        self._max = max(self._max, float(p.max()))


# ------------------------------------------------------------------ n-step collector
class NStepCollector:
    """Emits n-step transitions per env.

    `done` for the horizon-length transition is taken from the CURRENT step, not hard-coded to 0:
    when the horizon fills on the step where the episode also ends, that transition ends at the
    terminal and must not bootstrap from the next episode's first frame.
    """

    def __init__(self, n_envs: int, n: int, gamma: float = 0.99):
        self.n_envs, self.n, self.gamma = n_envs, n, gamma
        self.buf = [deque() for _ in range(n_envs)]

    def push(self, obs, actions, rewards, next_obs, dones):
        out = []
        for i in range(self.n_envs):
            self.buf[i].append((obs[i], actions[i], float(rewards[i])))
            if len(self.buf[i]) == self.n or dones[i]:
                items = list(self.buf[i])
                r = sum((self.gamma ** k) * items[k][2] for k in range(len(items)))
                out.append((items[0][0], next_obs[i], items[0][1], r, float(dones[i])))
                self.buf[i].popleft()
            if dones[i]:
                while self.buf[i]:
                    items = list(self.buf[i])
                    r = sum((self.gamma ** k) * items[k][2] for k in range(len(items)))
                    out.append((items[0][0], next_obs[i], items[0][1], r, 1.0))
                    self.buf[i].popleft()
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="MsPacman")
    ap.add_argument("--envs", type=int, default=32)
    ap.add_argument("--steps", type=int, default=4_000_000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--n-step", type=int, default=3)
    ap.add_argument("--atoms", type=int, default=51)
    ap.add_argument("--vmin", type=float, default=-1.0)
    ap.add_argument("--vmax", type=float, default=45.0)
    ap.add_argument("--reward-scale", type=float, default=1 / 100)
    ap.add_argument("--reward-transform", default="squash", choices=["scale", "squash"])
    ap.add_argument("--buffer", type=int, default=200_000)
    ap.add_argument("--naive-frames", action="store_true",
                    help="store obs and next_obs as separate 4-frame stacks (2x the memory). "
                         "Default shares the overlapping frames: 500k costs 16.4 GB, not 26.3 GB.")
    ap.add_argument("--learning-starts", type=int, default=20_000)
    ap.add_argument("--train-freq", type=int, default=4)
    ap.add_argument("--gradient-steps", type=int, default=2)
    ap.add_argument("--target-update", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=500_000)
    ap.add_argument("--eval-episodes", type=int, default=6)
    ap.add_argument("--trunk", default="nature", choices=["nature", "impala"])
    ap.add_argument("--width", type=int, default=64,
                    help="trunk width (last conv channels). 64 -> 3.55M params, 112 -> ~6.0M")
    ap.add_argument("--noisy", action="store_true",
                    help="NoisyLinear heads; disables epsilon-greedy entirely")
    ap.add_argument("--eps-steps", type=int, default=0,
                    help="steps over which epsilon decays 1.0->0.01. 0 means 10%% of --steps, which "
                         "makes the exploration schedule depend on run length and breaks "
                         "comparability between a 4M and a 10M run at equal step counts")
    ap.add_argument("--out", default="models/rainbow_mspacman")
    args = ap.parse_args()

    from envpool_vec import EnvPoolAtari
    dev = "cuda" if th.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    env = EnvPoolAtari(f"{args.game}-v5", num_envs=args.envs, seed=0, reward_clip=False)
    n_act = env.action_space.n
    net = DuelingC51(env.observation_space.shape[0], n_act, args.atoms,
                     width=args.width, trunk=args.trunk, noisy=args.noisy).to(dev)
    tgt = DuelingC51(env.observation_space.shape[0], n_act, args.atoms,
                     width=args.width, trunk=args.trunk, noisy=args.noisy).to(dev)
    tgt.load_state_dict(net.state_dict())
    opt = th.optim.Adam(net.parameters(), lr=args.lr, eps=1e-2 / args.batch_size)
    support = th.linspace(args.vmin, args.vmax, args.atoms, device=dev)
    buf = PrioritisedReplay(args.buffer, env.observation_space.shape, dev,
                            shared_frames=not args.naive_frames)
    coll = NStepCollector(args.envs, args.n_step, args.gamma)

    eps_frac = args.eps_steps if args.eps_steps > 0 else int(0.10 * args.steps)
    print(f"Rainbow-lite | {args.envs} envs | actions {n_act} | atoms {args.atoms} "
          f"[{args.vmin},{args.vmax}] | n-step {args.n_step} | {args.reward_transform} | "
          f"params {sum(p.numel() for p in net.parameters()):,} | eps over {eps_frac:,} steps | "
          f"{args.eval_episodes} eval episodes", flush=True)

    obs = env.reset()
    steps, updates = 0, 0
    t0 = time.time()
    next_eval = args.eval_every
    best = -1e18

    while steps < args.steps:
        if args.noisy:
            # Noisy nets replace epsilon-greedy entirely: exploration lives in the weights, so a
            # fresh noise sample is drawn every step and no uniform random action is taken.
            net.reset_noise()
            with th.no_grad():
                actions = net.q(th.as_tensor(obs, device=dev), support).argmax(1).cpu().numpy()
            if steps < args.learning_starts:
                actions = np.random.randint(0, n_act, size=args.envs)
        else:
            eps = 1.0 + (0.01 - 1.0) * min(1.0, steps / eps_frac)
            if steps < args.learning_starts or np.random.rand() < eps:
                actions = np.random.randint(0, n_act, size=args.envs)
            else:
                with th.no_grad():
                    actions = net.q(th.as_tensor(obs, device=dev), support).argmax(1).cpu().numpy()

        nxt, rew, done, _ = env.step(actions)
        r = np.asarray(rew, dtype=np.float32)
        if args.reward_transform == "squash":
            r = np.sign(r) * (np.sqrt(np.abs(r) + 1.0) - 1.0)
        else:
            r = r * args.reward_scale
        for s in coll.push(obs, actions, r, nxt, done.astype(np.float32)):
            buf.add(np.array([s[0]]), np.array([s[1]]), np.array([s[2]]),
                    np.array([s[3]]), np.array([s[4]]))
        obs = nxt
        steps += args.envs

        if steps >= args.learning_starts and steps % (args.train_freq * args.envs) == 0:
            for _ in range(args.gradient_steps):
                idx, w = buf.sample(args.batch_size, steps)
                b = buf.batch(idx)
                with th.no_grad():
                    next_q = net.q(b["next_obs"], support)                  # double-Q selection
                    best_a = next_q.argmax(1)
                    next_dist = tgt(b["next_obs"])[th.arange(len(idx)), best_a].softmax(-1)
                    target = project_distribution(next_dist, b["rewards"], b["dones"], support,
                                                  args.gamma ** args.n_step, args.vmin, args.vmax)
                logp = F.log_softmax(net(b["obs"])[th.arange(len(idx)), b["actions"]], dim=-1)
                loss = -(target * logp).sum(-1)
                # Rainbow ranks by KL(target || online); with a categorical head that is the
                # cross-entropy minus the (constant) target entropy, so the ranking is preserved
                # but the magnitude is comparable across samples.
                kl = loss + (target * (target + 1e-8).log()).sum(-1)
                buf.update_priorities(idx, kl.detach().abs().cpu().numpy())
                loss = (loss * th.as_tensor(w, device=dev)).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 10.0)
                opt.step()
                updates += 1
                if updates % args.target_update == 0:
                    tgt.load_state_dict(net.state_dict())

        if steps >= next_eval:
            next_eval += args.eval_every
            net.eval()
            net.noise_off()          # evaluate the mean policy
            scores = []
            for _ in range(args.eval_episodes):
                e = EnvPoolAtari(f"{args.game}-v5", num_envs=1, seed=1234, reward_clip=False)
                o = e.reset()
                total = 0.0
                for _ in range(20000):
                    with th.no_grad():
                        a = net.q(th.as_tensor(o, device=dev), support).argmax(1).cpu().numpy()
                    o, r2, d2, _ = e.step(a)
                    total += float(r2[0])
                    if d2[0]:
                        break
                scores.append(total)
                e.close()
            net.train()
            mean = float(np.mean(scores))
            print(f"[{steps:>10,}] eval mean {mean:8.1f}  best {best:8.1f} | "
                  f"{steps/(time.time()-t0):6.0f} steps/s | {(time.time()-t0)/60:5.1f} min",
                  flush=True)
            if mean > best:
                best = mean
                th.save(net.state_dict(), out / "best.pt")
            th.save(net.state_dict(), out / "latest.pt")

    th.save(net.state_dict(), out / "final.pt")
    print(f"done: {steps:,} steps, best eval {best:.1f} -> {out}", flush=True)


if __name__ == "__main__":
    main()
