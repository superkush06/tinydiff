"""Tiny neural-net layers."""

from __future__ import annotations

import math

import numpy as np

from . import ops
from .tensor import Tensor


class Module:
    """Base class — recursively collects Tensors with requires_grad."""

    def parameters(self) -> list[Tensor]:
        """Every requires_grad Tensor held by this Module, its attributes, or lists.

        >>> import numpy as np, tinydiff as td
        >>> net = td.Sequential(td.Linear(3, 4), td.relu, td.Linear(4, 2))
        >>> [p.shape for p in net.parameters()]
        [(3, 4), (4,), (4, 2), (2,)]
        """
        out: list[Tensor] = []
        for _name, val in self.__dict__.items():
            if isinstance(val, Tensor) and val.requires_grad:
                out.append(val)
            elif isinstance(val, Module):
                out.extend(val.parameters())
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, Module):
                        out.extend(item.parameters())
                    elif isinstance(item, Tensor) and item.requires_grad:
                        out.append(item)
        return out

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = None

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


# Shared fallback generator: consecutive Linear layers built without an
# explicit rng must NOT re-seed from scratch, or every same-shaped layer
# in a network gets byte-identical weights (perfectly correlated features).
_default_rng = np.random.default_rng()


class Linear(Module):
    """Fully connected layer: y = x @ W + b.

    Kaiming-He uniform init for ReLU networks (the standard default).
    Pass an explicit `rng` for reproducible init; by default each layer
    draws from a shared process-wide generator.

    >>> import numpy as np, tinydiff as td
    >>> layer = td.Linear(3, 2, rng=np.random.default_rng(0))
    >>> x = td.Tensor(np.ones((5, 3)))
    >>> y = layer(x)
    >>> y.shape
    (5, 2)
    >>> y.sum().backward()
    >>> layer.W.grad.shape, layer.b.grad.tolist()   # one bias hit per row
    ((3, 2), [5.0, 5.0])
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 rng: np.random.Generator | None = None) -> None:
        rng = _default_rng if rng is None else rng
        bound = math.sqrt(6.0 / in_features)
        self.W = Tensor(rng.uniform(-bound, bound, size=(in_features, out_features)),
                        requires_grad=True)
        self.b = (Tensor(np.zeros(out_features), requires_grad=True)
                  if bias else None)

    def forward(self, x: Tensor) -> Tensor:
        y = ops.matmul(x, self.W)
        if self.b is not None:
            y = ops.add(y, self.b)
        return y


class Sequential(Module):
    """Apply layers in sequence; activations can be plain callables.

    >>> import numpy as np, tinydiff as td
    >>> net = td.Sequential(td.Linear(2, 4), td.relu, td.Linear(4, 1))
    >>> net(td.Tensor(np.zeros((6, 2)))).shape
    (6, 1)
    """

    def __init__(self, *layers) -> None:
        self.layers = list(layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> list[Tensor]:
        out: list[Tensor] = []
        for layer in self.layers:
            if isinstance(layer, Module):
                out.extend(layer.parameters())
        return out
