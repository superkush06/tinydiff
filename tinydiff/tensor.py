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
    __slots__ = ("data", "grad", "requires_grad", "_children", "_backward",
                 "_op", "_backward_ran")

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
        self._backward_ran = False

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

    # --- graph traversal ----------------------------------------------
    def _toposort(self) -> list[Tensor]:
        """Reverse-topological order via an explicit stack.

        Iterative on purpose: a recursive DFS dies with RecursionError
        around Python's ~1000-frame limit, which an unrolled RNN or any
        long op chain hits immediately.
        """
        topo: list[Tensor] = []
        seen: set[int] = set()
        stack: list[tuple[Tensor, bool]] = [(self, False)]
        while stack:
            t, processed = stack.pop()
            if processed:
                topo.append(t)
                continue
            if id(t) in seen:
                continue
            seen.add(id(t))
            stack.append((t, True))
            for c in t._children:
                if id(c) not in seen:
                    stack.append((c, False))
        return topo

    # --- topological backward ----------------------------------------
    def backward(self, grad: np.ndarray | float | None = None,
                 retain_graph: bool = False) -> None:
        """Accumulate gradient through the graph rooted at `self`.

        If `grad` is omitted, `self` must be scalar and we seed with 1.0.
        An explicit `grad` must have exactly `self.data`'s shape — a
        mis-shaped seed would silently broadcast into wrong gradients.

        Calling backward() a second time through the same graph requires
        `retain_graph=True`; intermediate grads are cleared before each
        pass so leaves accumulate correctly instead of compounding stale
        intermediate values.
        """
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("backward() on non-scalar requires explicit `grad`")
            grad = np.ones_like(self.data)
        grad = np.asarray(grad, dtype=np.float64)
        if grad.shape != self.data.shape:
            raise ValueError(
                f"backward() seed shape {grad.shape} does not match "
                f"output shape {self.data.shape}"
            )

        topo = self._toposort()

        # Re-entrancy safety: a second pass through an already-used graph
        # compounds retained intermediate grads into silently wrong numbers
        # unless the caller opts in and we reset the intermediates.
        interior = [t for t in topo if t._children]
        if any(t._backward_ran for t in interior) and not retain_graph:
            raise RuntimeError(
                "backward() was already called through this graph; pass "
                "retain_graph=True to backpropagate again (leaf grads will "
                "accumulate)"
            )
        for t in interior:
            t._backward_ran = True
            t.grad = None  # stale interior grads must not leak into this pass

        # Seed
        self.grad = grad if self.grad is None else self.grad + grad
        # Walk reverse
        for t in reversed(topo):
            t._backward()

    def zero_grad(self) -> None:
        """Reset gradients on this tensor and all reachable ancestors."""
        for t in self._toposort():
            t.grad = None


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
