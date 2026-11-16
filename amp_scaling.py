"""Autocast + loss scaling for Stable-Baselines3 policies.

SB3 computes `loss.backward()` and `optimizer.step()` inside its own train loop, so there is
no seam to insert `torch.amp.GradScaler`. This module provides the two pieces that reproduce
exactly what GradScaler does:

  1. loss scaling: intercept `Tensor.backward()` so the loss is multiplied by `scale`
     before it is differentiated (prevents fp16 underflow in the backward pass);
  2. un-scaling + overflow guard: an optimizer wrapper that divides gradients by `scale`,
     skips the step and shrinks the scale if any gradient is inf/nan, and grows the scale
     again slowly after successful steps.

bf16 does not need scaling (8-bit exponent == fp32 range) — pass scale=0 to run plain
autocast. fp16 needs it; scale defaults to 2**16.

    from amp_scaling import enable_amp
    enable_amp(model, dtype="bf16")                 # autocast only
    enable_amp(model, dtype="fp16", scale=65536)    # autocast + loss scaling
"""
from __future__ import annotations

import contextlib

import torch as th

_DTYPES = {"bf16": th.bfloat16, "fp16": th.float16}


class _ScaledOptimizer:
    """Un-scales grads, skips overflow steps, adapts the scale. API-compatible with torch.optim."""

    def __init__(self, optimizer, state: dict):
        self._opt = optimizer
        self._state = state  # shared mutable: {"scale": float, "growth": int, "skipped": int}

    def step(self, closure=None):
        scale = float(self._state["scale"])
        grads = [p.grad for p in self._opt.param_groups[0]["params"] if p.grad is not None]
        if not grads:
            return None

        # unscale + overflow check in ONE fused kernel (same op torch.amp.GradScaler uses),
        # instead of a per-parameter python loop with a sync each
        if scale != 1.0:
            found_inf = th.zeros(1, device=grads[0].device)
            inv = th.full((1,), 1.0 / scale, device=grads[0].device, dtype=th.float32)
            th._amp_foreach_non_finite_check_and_unscale_(grads, found_inf, inv)
            if bool(found_inf.item() > 0):
                self._state["skipped"] += 1
                self._state["scale"] = max(scale / 2.0, 1.0)
                self._opt.zero_grad(set_to_none=True)
                return None

        out = self._opt.step(closure)
        self._state["growth"] += 1
        if self._state["growth"] >= 200:          # grow slowly after sustained success
            self._state["growth"] = 0
            self._state["scale"] = min(self._state["scale"] * 2.0, 2.0**24)
        return out

    def zero_grad(self, *a, **kw):
        return self._opt.zero_grad(*a, **kw)

    @property
    def param_groups(self):
        return self._opt.param_groups

    def state_dict(self):
        return self._opt.state_dict()

    def __getattr__(self, name):
        return getattr(self._opt, name)


class _ScaledBackward:
    """Global `Tensor.backward` shim: scales the loss that is being differentiated."""

    def __init__(self, state: dict):
        self._state = state
        self._orig = None

    def install(self):
        if self._orig is not None:
            return
        self._orig = th.Tensor.backward
        orig = self._orig
        state = self._state

        def backward(self, *args, **kwargs):
            scale = float(state.get("scale", 1.0))
            if scale != 1.0 and self.requires_grad and self.is_floating_point():
                scaled = self * scale
                return orig(scaled, *args, **kwargs)
            return orig(self, *args, **kwargs)

        th.Tensor.backward = backward

    def remove(self):
        if self._orig is not None:
            th.Tensor.backward = self._orig
            self._orig = None


def enable_amp(model, dtype: str = "bf16", scale: float = 0.0) -> dict:
    """Attach autocast (+ optional loss scaling) to an SB3 model. Returns the state dict."""
    if dtype not in _DTYPES:
        raise ValueError(f"dtype must be one of {list(_DTYPES)}")
    if not th.cuda.is_available():
        raise RuntimeError("autocast requested but CUDA is not available")
    th_dtype = _DTYPES[dtype]
    if dtype == "bf16" and scale:
        print("note: bf16 has fp32 exponent range, loss scaling is a no-op; using scale=1", flush=True)
        scale = 0.0
    if dtype == "fp16" and not scale:
        scale = 65536.0

    state = {"scale": float(scale) if scale else 1.0, "growth": 0, "skipped": 0, "dtype": dtype}

    # 1. autocast the policy forward (SB3 routes every forward through policy.forward)
    orig_forward = model.policy.forward

    def amp_forward(*args, **kwargs):
        with th.autocast("cuda", dtype=th_dtype):
            out = orig_forward(*args, **kwargs)
        # SB3 round-trips values/log_probs through numpy; numpy has no bf16/fp16 dtype
        if isinstance(out, tuple):
            return tuple(o.float() if th.is_tensor(o) and o.dtype in (th.bfloat16, th.float16) else o
                         for o in out)
        return out

    model.policy.forward = amp_forward

    # 2. loss scaling + overflow guard
    if scale:
        model.policy.optimizer = _ScaledOptimizer(model.policy.optimizer, state)
        shim = _ScaledBackward(state)
        shim.install()
        state["_shim"] = shim

    print(f"amp: {dtype} autocast on, loss scale {state['scale']:.0f}"
          f"{' (dynamic)' if scale else ' (off)'}", flush=True)
    return state


def disable_amp(state: dict) -> None:
    shim = state.get("_shim")
    if shim is not None:
        shim.remove()


@contextlib.contextmanager
def amp_scope(model, dtype: str = "bf16", scale: float = 0.0):
    state = enable_amp(model, dtype=dtype, scale=scale)
    try:
        yield state
    finally:
        disable_amp(state)
