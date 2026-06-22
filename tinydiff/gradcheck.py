"""Numerical gradient checker — central differences vs autodiff."""

from __future__ import annotations

import numpy as np

from .tensor import Tensor


def grad_check(fn, *inputs, eps: float = 1e-6, tol: float = 1e-4) -> bool:
    """Return True if numerical grads match analytical grads within `tol`.

    `fn(*tensors) -> scalar Tensor`. We perturb each scalar entry of each
    input by ±eps and compare central-difference grad to autodiff grad.
    """
    inputs = [Tensor(x.data.copy(), requires_grad=True) if isinstance(x, Tensor)
              else Tensor(np.asarray(x).copy(), requires_grad=True)
              for x in inputs]
    out = fn(*inputs)
    out.backward()

    ok = True
    for t in inputs:
        if t.grad is None:
            continue
        flat = t.data.reshape(-1)
        grad_flat = t.grad.reshape(-1)
        for i in range(flat.size):
            orig = flat[i]
            flat[i] = orig + eps
            f_plus = fn(*[Tensor(x.data.copy(), requires_grad=False)
                          for x in inputs]).data.item()
            flat[i] = orig - eps
            f_minus = fn(*[Tensor(x.data.copy(), requires_grad=False)
                           for x in inputs]).data.item()
            flat[i] = orig
            numerical = (f_plus - f_minus) / (2 * eps)
            if abs(numerical - grad_flat[i]) > tol * max(1.0, abs(numerical)):
                ok = False
                print(f"grad mismatch idx={i}: num={numerical:.6f} ad={grad_flat[i]:.6f}")
    return ok
