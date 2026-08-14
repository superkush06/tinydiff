"""Optimizers."""

from __future__ import annotations

import numpy as np


class Optimizer:
    def __init__(self, params) -> None:
        self.params = list(params)

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def step(self) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    """Vanilla SGD with optional momentum.

    Two steps of ``lr=0.1`` on ``f(x) = x**2`` from x = 1: gradient 2.0 takes
    it to 0.8, then gradient 1.6 takes it to 0.64.

    >>> import numpy as np, tinydiff as td
    >>> x = td.Tensor(np.array([1.0]), requires_grad=True)
    >>> opt = td.SGD([x], lr=0.1)
    >>> for _ in range(2):
    ...     opt.zero_grad()
    ...     (x * x).sum().backward()
    ...     opt.step()
    >>> x.data.tolist()
    [0.64]
    """

    def __init__(self, params, lr: float = 0.01, momentum: float = 0.0) -> None:
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self._v = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            self._v[i] = self.momentum * self._v[i] + p.grad
            p.data = p.data - self.lr * self._v[i]


class Adam(Optimizer):
    """Adam (Kingma & Ba 2015).

    The first step is `lr` in the downhill direction whatever the gradient's
    magnitude — the scale-free property that makes Adam's default lr portable:

    >>> import numpy as np, tinydiff as td
    >>> x = td.Tensor(np.array([5.0]), requires_grad=True)
    >>> opt = td.Adam([x], lr=0.1)
    >>> (x * x).sum().backward()          # gradient is 10, step is still 0.1
    >>> opt.step()
    >>> x.data.round(6).tolist()
    [4.9]
    """

    def __init__(self, params, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8) -> None:
        super().__init__(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self._m = [np.zeros_like(p.data) for p in self.params]
        self._v = [np.zeros_like(p.data) for p in self.params]
        self._t = 0

    def step(self) -> None:
        self._t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            self._m[i] = self.b1 * self._m[i] + (1 - self.b1) * p.grad
            self._v[i] = self.b2 * self._v[i] + (1 - self.b2) * (p.grad ** 2)
            m_hat = self._m[i] / (1 - self.b1 ** self._t)
            v_hat = self._v[i] / (1 - self.b2 ** self._t)
            p.data = p.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
