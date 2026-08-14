"""Numerical gradient checker — central differences vs autodiff."""

from __future__ import annotations

import numpy as np

from .tensor import Tensor


def grad_check(fn, *inputs, eps: float = 1e-6, tol: float = 1e-4) -> bool:
    """Return True if numerical grads match analytical grads within `tol`.

    `fn(*tensors) -> scalar Tensor`. We perturb each scalar entry of each
    input by ±eps and compare central-difference grad to autodiff grad.

    >>> import numpy as np, tinydiff as td
    >>> rng = np.random.default_rng(0)
    >>> def f(a, b):
    ...     return td.mean(td.sigmoid(a @ b))
    >>> td.grad_check(f, rng.standard_normal((4, 3)), rng.standard_normal((3, 2)))
    True

    An input that receives no gradient is a failure, not a skip:

    >>> td.grad_check(lambda a, b: (a * a).sum(), np.ones(2), np.ones(2))
    grad missing: input 1 (shape (2,)) has requires_grad but received no gradient
    False

    `tol` is a relative tolerance. Do not push it below about ``1e-9``: that is
    where the central-difference estimator itself bottoms out, so a tighter
    number would be measuring round-off rather than the engine.
    """
    inputs = [Tensor(x.data.copy(), requires_grad=True) if isinstance(x, Tensor)
              else Tensor(np.asarray(x).copy(), requires_grad=True)
              for x in inputs]
    out = fn(*inputs)
    out.backward()

    ok = True
    for k, t in enumerate(inputs):
        if t.grad is None:
            # An input that never received a gradient is a failure, not a
            # skip: a broken op that forgets to write its grad would
            # otherwise sail through the checker unnoticed.
            print(f"grad missing: input {k} (shape {t.data.shape}) has "
                  f"requires_grad but received no gradient")
            ok = False
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
