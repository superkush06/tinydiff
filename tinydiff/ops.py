"""Tensor operations with autograd."""

from __future__ import annotations

import numpy as np

from .tensor import Tensor, _unbroadcast


def _ensure(t):
    return t if isinstance(t, Tensor) else Tensor(t)


# --- binary -----------------------------------------------------------
def add(a, b) -> Tensor:
    """Elementwise sum, with NumPy broadcasting.

    The backward pass sums a broadcast operand's gradient back down to that
    operand's own shape, so a bias row added to a batch of 3 collects three
    contributions per column:

    >>> import numpy as np, tinydiff as td
    >>> x = td.Tensor(np.zeros((3, 2)), requires_grad=True)
    >>> bias = td.Tensor(np.zeros(2), requires_grad=True)
    >>> td.add(x, bias).sum().backward()
    >>> x.grad.shape, bias.grad.tolist()
    ((3, 2), [3.0, 3.0])
    """
    a, b = _ensure(a), _ensure(b)
    out = Tensor(a.data + b.data, _children=(a, b), _op="add")

    def _backward() -> None:
        if a.requires_grad:
            a.grad = (np.zeros_like(a.data) if a.grad is None else a.grad) \
                     + _unbroadcast(out.grad, a.data.shape)
        if b.requires_grad:
            b.grad = (np.zeros_like(b.data) if b.grad is None else b.grad) \
                     + _unbroadcast(out.grad, b.data.shape)

    out._backward = _backward
    return out


def sub(a, b) -> Tensor:
    """Elementwise difference, built as ``a + (-b)`` so it needs no VJP of its own.

    >>> import tinydiff as td
    >>> a = td.Tensor(5.0, requires_grad=True)
    >>> b = td.Tensor(3.0, requires_grad=True)
    >>> td.sub(a, b).backward()
    >>> float(a.grad), float(b.grad)
    (1.0, -1.0)
    """
    return add(a, neg(_ensure(b)))


def mul(a, b) -> Tensor:
    """Elementwise product. Each operand's gradient is the *other* operand.

    >>> import tinydiff as td
    >>> a = td.Tensor(4.0, requires_grad=True)
    >>> b = td.Tensor(5.0, requires_grad=True)
    >>> td.mul(a, b).backward()
    >>> float(a.grad), float(b.grad)
    (5.0, 4.0)
    """
    a, b = _ensure(a), _ensure(b)
    out = Tensor(a.data * b.data, _children=(a, b), _op="mul")

    def _backward() -> None:
        if a.requires_grad:
            a.grad = (np.zeros_like(a.data) if a.grad is None else a.grad) \
                     + _unbroadcast(out.grad * b.data, a.data.shape)
        if b.requires_grad:
            b.grad = (np.zeros_like(b.data) if b.grad is None else b.grad) \
                     + _unbroadcast(out.grad * a.data, b.data.shape)

    out._backward = _backward
    return out


def div(a, b) -> Tensor:
    """Elementwise quotient: gradients are ``1/b`` and ``-a / b**2``.

    >>> import tinydiff as td
    >>> a = td.Tensor(1.0, requires_grad=True)
    >>> b = td.Tensor(4.0, requires_grad=True)
    >>> td.div(a, b).backward()
    >>> float(a.grad), float(b.grad)
    (0.25, -0.0625)
    """
    a, b = _ensure(a), _ensure(b)
    out = Tensor(a.data / b.data, _children=(a, b), _op="div")

    def _backward() -> None:
        if a.requires_grad:
            a.grad = (np.zeros_like(a.data) if a.grad is None else a.grad) \
                     + _unbroadcast(out.grad / b.data, a.data.shape)
        if b.requires_grad:
            b.grad = (np.zeros_like(b.data) if b.grad is None else b.grad) \
                     + _unbroadcast(-out.grad * a.data / (b.data ** 2), b.data.shape)

    out._backward = _backward
    return out


def matmul(a, b) -> Tensor:
    """Matrix product with full np.matmul gufunc semantics: (n?,k),(k,m?).

    Backward handles the three things a naive `a.data.T @ out.grad` gets
    wrong: 1-D operands (promoted to a row/column, then squeezed back),
    stacked/batched operands (transpose only the last two axes via
    swapaxes, never `.T` which reverses ALL axes), and broadcast batch
    dimensions (summed out with _unbroadcast).

    >>> import numpy as np, tinydiff as td
    >>> a = td.Tensor(np.ones((4, 3, 2)), requires_grad=True)   # batch of 4
    >>> b = td.Tensor(np.ones((2, 5)), requires_grad=True)      # shared weights
    >>> td.matmul(a, b).sum().backward()
    >>> a.grad.shape, b.grad.shape        # b's batch axis is summed out
    ((4, 3, 2), (2, 5))
    >>> float(b.grad[0, 0])               # one contribution per (batch, row)
    12.0
    """
    a, b = _ensure(a), _ensure(b)
    out = Tensor(a.data @ b.data, _children=(a, b), _op="matmul")

    def _backward() -> None:
        A, B, G = a.data, b.data, out.grad
        # Promote per the gufunc convention: 1-D a is a row (1, k);
        # 1-D b is a column (k, 1). np.matmul then squeezes those axes
        # out of the result, so re-insert them into G to match.
        A2 = A[None, :] if A.ndim == 1 else A
        B2 = B[:, None] if B.ndim == 1 else B
        G2 = G
        if A.ndim == 1 and B.ndim == 1:
            G2 = G.reshape(1, 1)
        elif A.ndim == 1:
            G2 = np.expand_dims(G, axis=-2)
        elif B.ndim == 1:
            G2 = np.expand_dims(G, axis=-1)
        if a.requires_grad:
            dA = _unbroadcast(G2 @ B2.swapaxes(-1, -2), A2.shape).reshape(A.shape)
            a.grad = (np.zeros_like(A) if a.grad is None else a.grad) + dA
        if b.requires_grad:
            dB = _unbroadcast(A2.swapaxes(-1, -2) @ G2, B2.shape).reshape(B.shape)
            b.grad = (np.zeros_like(B) if b.grad is None else b.grad) + dB

    out._backward = _backward
    return out


def pow_(a, p: float) -> Tensor:
    """Elementwise power by a constant exponent, ``a ** p``.

    `p` is a plain float, not a Tensor: there is no gradient with respect to
    the exponent.

    >>> import tinydiff as td
    >>> x = td.Tensor(3.0, requires_grad=True)
    >>> td.pow_(x, 3.0).backward()
    >>> float(x.grad)                     # 3 * x**2
    27.0
    """
    a = _ensure(a)
    out = Tensor(a.data ** p, _children=(a,), _op=f"pow({p})")

    def _backward() -> None:
        if a.requires_grad:
            a.grad = (np.zeros_like(a.data) if a.grad is None else a.grad) \
                     + out.grad * p * (a.data ** (p - 1))

    out._backward = _backward
    return out


# --- unary ------------------------------------------------------------
def neg(a) -> Tensor:
    """Elementwise negation.

    >>> import tinydiff as td
    >>> x = td.Tensor(2.0, requires_grad=True)
    >>> td.neg(x).backward()
    >>> float(x.grad)
    -1.0
    """
    a = _ensure(a)
    out = Tensor(-a.data, _children=(a,), _op="neg")

    def _backward() -> None:
        if a.requires_grad:
            a.grad = (np.zeros_like(a.data) if a.grad is None else a.grad) - out.grad

    out._backward = _backward
    return out


def exp(a) -> Tensor:
    """Elementwise ``e ** a``. Its own derivative, so backward reuses the output.

    >>> import tinydiff as td
    >>> x = td.Tensor(1.0, requires_grad=True)
    >>> y = td.exp(x)
    >>> y.backward()
    >>> float(y.data) == float(x.grad)
    True
    """
    a = _ensure(a)
    out = Tensor(np.exp(a.data), _children=(a,), _op="exp")

    def _backward() -> None:
        if a.requires_grad:
            a.grad = (np.zeros_like(a.data) if a.grad is None else a.grad) \
                     + out.grad * out.data

    out._backward = _backward
    return out


def log(a) -> Tensor:
    """Elementwise natural logarithm.

    >>> import tinydiff as td
    >>> x = td.Tensor(4.0, requires_grad=True)
    >>> td.log(x).backward()
    >>> float(x.grad)                     # 1/x
    0.25
    """
    a = _ensure(a)
    out = Tensor(np.log(a.data), _children=(a,), _op="log")

    def _backward() -> None:
        if a.requires_grad:
            a.grad = (np.zeros_like(a.data) if a.grad is None else a.grad) \
                     + out.grad / a.data

    out._backward = _backward
    return out


# --- activations ------------------------------------------------------
def relu(a) -> Tensor:
    """Elementwise ``max(0, a)``.

    At exactly zero the subgradient 0 is used — the same choice PyTorch makes,
    and one of the deliberate disagreements listed in ``docs/validation.md``:

    >>> import numpy as np, tinydiff as td
    >>> x = td.Tensor(np.array([-2.0, 0.0, 3.0]), requires_grad=True)
    >>> td.relu(x).sum().backward()
    >>> x.grad.tolist()
    [0.0, 0.0, 1.0]
    """
    a = _ensure(a)
    out = Tensor(np.maximum(0.0, a.data), _children=(a,), _op="relu")

    def _backward() -> None:
        if a.requires_grad:
            mask = (a.data > 0).astype(a.data.dtype)
            a.grad = (np.zeros_like(a.data) if a.grad is None else a.grad) \
                     + out.grad * mask

    out._backward = _backward
    return out


def sigmoid(a) -> Tensor:
    """Elementwise logistic function.

    >>> import tinydiff as td
    >>> x = td.Tensor(0.0, requires_grad=True)
    >>> y = td.sigmoid(x)
    >>> y.backward()
    >>> float(y.data), float(x.grad)      # s(0) = 1/2, s'(0) = 1/4
    (0.5, 0.25)
    """
    a = _ensure(a)
    s = 1.0 / (1.0 + np.exp(-a.data))
    out = Tensor(s, _children=(a,), _op="sigmoid")

    def _backward() -> None:
        if a.requires_grad:
            a.grad = (np.zeros_like(a.data) if a.grad is None else a.grad) \
                     + out.grad * s * (1.0 - s)

    out._backward = _backward
    return out


def tanh(a) -> Tensor:
    """Elementwise hyperbolic tangent.

    >>> import tinydiff as td
    >>> x = td.Tensor(0.0, requires_grad=True)
    >>> y = td.tanh(x)
    >>> y.backward()
    >>> float(y.data), float(x.grad)      # tanh'(0) = 1
    (0.0, 1.0)
    """
    a = _ensure(a)
    t = np.tanh(a.data)
    out = Tensor(t, _children=(a,), _op="tanh")

    def _backward() -> None:
        if a.requires_grad:
            a.grad = (np.zeros_like(a.data) if a.grad is None else a.grad) \
                     + out.grad * (1.0 - t * t)

    out._backward = _backward
    return out


# --- reductions -------------------------------------------------------
def sum_(a, axis=None, keepdims: bool = False) -> Tensor:
    """Sum over `axis` (or over everything). The gradient broadcasts straight back.

    >>> import numpy as np, tinydiff as td
    >>> x = td.Tensor(np.arange(6.0).reshape(2, 3), requires_grad=True)
    >>> s = td.sum_(x, axis=0)
    >>> s.data.tolist()
    [3.0, 5.0, 7.0]
    >>> s.backward(np.ones(3))            # a non-scalar root needs an explicit seed
    >>> x.grad.tolist()
    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    """
    a = _ensure(a)
    out = Tensor(a.data.sum(axis=axis, keepdims=keepdims),
                 _children=(a,), _op=f"sum(axis={axis})")

    def _backward() -> None:
        if a.requires_grad:
            grad = out.grad
            if not keepdims and axis is not None:
                grad = np.expand_dims(grad, axis=axis)
            a.grad = (np.zeros_like(a.data) if a.grad is None else a.grad) \
                     + np.broadcast_to(grad, a.data.shape).copy()

    out._backward = _backward
    return out


def mean(a, axis=None, keepdims: bool = False) -> Tensor:
    """Mean over `axis` (or over everything), composed as ``sum_ / n``.

    >>> import numpy as np, tinydiff as td
    >>> x = td.Tensor(np.arange(4.0), requires_grad=True)
    >>> m = td.mean(x)
    >>> m.backward()
    >>> float(m.data), x.grad.tolist()    # every entry gets 1/n
    (1.5, [0.25, 0.25, 0.25, 0.25])
    """
    a = _ensure(a)
    if axis is None:
        n = a.data.size
    else:
        axes = axis if isinstance(axis, tuple) else (axis,)
        n = 1
        for ax in axes:
            n *= a.data.shape[ax]
    return div(sum_(a, axis=axis, keepdims=keepdims), Tensor(float(n)))


# --- operator overloads on Tensor -------------------------------------
Tensor.__add__ = lambda self, o: add(self, o)
Tensor.__radd__ = lambda self, o: add(o, self)
Tensor.__sub__ = lambda self, o: sub(self, o)
Tensor.__rsub__ = lambda self, o: sub(o, self)
Tensor.__mul__ = lambda self, o: mul(self, o)
Tensor.__rmul__ = lambda self, o: mul(o, self)
Tensor.__truediv__ = lambda self, o: div(self, o)
Tensor.__rtruediv__ = lambda self, o: div(o, self)
Tensor.__neg__ = lambda self: neg(self)
Tensor.__matmul__ = lambda self, o: matmul(self, o)
Tensor.__pow__ = lambda self, p: pow_(self, p)
Tensor.sum = lambda self, axis=None, keepdims=False: sum_(self, axis, keepdims)
Tensor.mean = lambda self, axis=None, keepdims=False: mean(self, axis, keepdims)
Tensor.relu = lambda self: relu(self)
