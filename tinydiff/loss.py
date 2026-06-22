"""Loss functions."""

from __future__ import annotations

import numpy as np

from . import ops
from .tensor import Tensor


def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Mean squared error."""
    if not isinstance(target, Tensor):
        target = Tensor(target)
    diff = ops.sub(pred, target)
    return ops.mean(ops.pow_(diff, 2.0))


def softmax_crossentropy(logits: Tensor, target_idx) -> Tensor:
    """Numerically-stable softmax cross-entropy.

    `logits` shape (N, C); `target_idx` shape (N,) with int class indices.
    """
    target_idx = np.asarray(target_idx).astype(np.int64)
    # Shift logits for stability
    z = logits.data - logits.data.max(axis=1, keepdims=True)
    exp_z = np.exp(z)
    sum_exp = exp_z.sum(axis=1, keepdims=True)
    log_softmax = z - np.log(sum_exp)
    nll = -log_softmax[np.arange(z.shape[0]), target_idx]
    loss_val = nll.mean()

    out = Tensor(loss_val, _children=(logits,), _op="softmax_xent")

    def _backward() -> None:
        if logits.requires_grad:
            probs = exp_z / sum_exp
            grad = probs.copy()
            grad[np.arange(z.shape[0]), target_idx] -= 1.0
            grad /= z.shape[0]
            logits.grad = (np.zeros_like(logits.data) if logits.grad is None
                           else logits.grad) + grad * out.grad

    out._backward = _backward
    return out
