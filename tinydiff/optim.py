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
    """Vanilla SGD with optional momentum."""

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
    """Adam (Kingma & Ba 2015)."""

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
