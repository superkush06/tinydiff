"""Loss functions."""

from __future__ import annotations

import numpy as np

from . import ops
from .tensor import Tensor


def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Mean squared error. `target` may be a Tensor or anything array-like.

    >>> import numpy as np, tinydiff as td
    >>> pred = td.Tensor(np.array([1.0, 2.0]), requires_grad=True)
    >>> loss = td.mse_loss(pred, np.array([0.0, 0.0]))
    >>> float(loss.data)
    2.5
    >>> loss.backward()
    >>> pred.grad.tolist()                # 2 * (pred - target) / n
    [1.0, 2.0]
    """
    if not isinstance(target, Tensor):
        target = Tensor(target)
    diff = ops.sub(pred, target)
    return ops.mean(ops.pow_(diff, 2.0))


def softmax_crossentropy(logits: Tensor, target_idx) -> Tensor:
    """Numerically-stable softmax cross-entropy.

    `logits` shape (N, C); `target_idx` shape (N,) with int class indices.

    The softmax and the log are fused, and the logits are max-shifted before
    exponentiating: the naive form returns a plausible wrong number — not an
    inf — once the logit gap passes ~745. See ``docs/validation.md`` §4.

    >>> import numpy as np, tinydiff as td
    >>> logits = td.Tensor(np.zeros((1, 4)), requires_grad=True)
    >>> loss = td.softmax_crossentropy(logits, [0])
    >>> round(float(loss.data), 6)        # -log(1/4)
    1.386294
    >>> loss.backward()
    >>> logits.grad.tolist()              # p - onehot, averaged over the batch
    [[-0.75, 0.25, 0.25, 0.25]]
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
