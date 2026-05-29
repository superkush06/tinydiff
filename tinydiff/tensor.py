"""Tensor — the autograd node.

A Tensor wraps a numpy array and remembers how it was produced so that we
can walk the computation graph backwards and accumulate gradients.

Key invariants:
    - `data` is always a numpy.ndarray.
    - `grad` is None until backward() is called; after, it's a numpy.ndarray
      with the same shape as `data`.
    - `_children` are the input Tensors of the op that produced this tensor.
    - `_backward()` is a closure that knows how to push grad into _children.
"""

from __future__ import annotations

import numpy as np


class Tensor:
    __slots__ = ("data", "grad", "requires_grad", "_children", "_backward", "_op")

    def __init__(self, data, requires_grad: bool = False,
                 _children: tuple = (), _op: str = "") -> None:
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = bool(requires_grad) or any(
            c.requires_grad for c in _children
        )
        self.grad: np.ndarray | None = None
        self._children = _children
        self._backward = lambda: None
        self._op = _op

    # --- shape & dunder boilerplate ----------------------------------
    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    def __repr__(self) -> str:
        g = "" if self.grad is None else f", grad_shape={self.grad.shape}"
        return f"Tensor(shape={self.shape}, op={self._op!r}{g})"

    # --- topological backward ----------------------------------------
    def backward(self, grad: np.ndarray | float | None = None) -> None:
        """Accumulate gradient through the graph rooted at `self`.

        If `grad` is omitted, `self` must be scalar and we seed with 1.0.
        """
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("backward() on non-scalar requires explicit `grad`")
            grad = np.ones_like(self.data)
        grad = np.asarray(grad, dtype=np.float64)

        # Reverse topo order
        topo: list[Tensor] = []
        seen: set[int] = set()

        def visit(t: Tensor) -> None:
            if id(t) in seen:
                return
            seen.add(id(t))
            for c in t._children:
                visit(c)
            topo.append(t)

        visit(self)
        # Seed
        self.grad = grad if self.grad is None else self.grad + grad
        # Walk reverse
        for t in reversed(topo):
            t._backward()

    def zero_grad(self) -> None:
        """Reset gradients on this tensor and all reachable ancestors."""
        self.grad = None
        seen: set[int] = set()

        def walk(t: Tensor) -> None:
            if id(t) in seen:
                return
            seen.add(id(t))
            t.grad = None
            for c in t._children:
                walk(c)

        walk(self)


def _unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Sum grad axes so it matches `shape` (inverse of numpy broadcasting)."""
    # Remove leading singleton dims
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    # Sum collapsed axes
    for i, (g_dim, t_dim) in enumerate(zip(grad.shape, shape, strict=False)):
        if t_dim == 1 and g_dim != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad
