"""Tensor operations with autograd."""

from __future__ import annotations

import numpy as np

from .tensor import Tensor, _unbroadcast


def _ensure(t):
    return t if isinstance(t, Tensor) else Tensor(t)


# --- binary -----------------------------------------------------------
def add(a, b) -> Tensor:
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
    return add(a, neg(_ensure(b)))


def mul(a, b) -> Tensor:
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
    a, b = _ensure(a), _ensure(b)
    out = Tensor(a.data @ b.data, _children=(a, b), _op="matmul")

    def _backward() -> None:
        if a.requires_grad:
            a.grad = (np.zeros_like(a.data) if a.grad is None else a.grad) \
                     + out.grad @ b.data.T
        if b.requires_grad:
            b.grad = (np.zeros_like(b.data) if b.grad is None else b.grad) \
                     + a.data.T @ out.grad

    out._backward = _backward
    return out


def pow_(a, p: float) -> Tensor:
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
    a = _ensure(a)
    out = Tensor(-a.data, _children=(a,), _op="neg")

    def _backward() -> None:
        if a.requires_grad:
            a.grad = (np.zeros_like(a.data) if a.grad is None else a.grad) - out.grad

    out._backward = _backward
    return out


def exp(a) -> Tensor:
    a = _ensure(a)
    out = Tensor(np.exp(a.data), _children=(a,), _op="exp")

    def _backward() -> None:
        if a.requires_grad:
            a.grad = (np.zeros_like(a.data) if a.grad is None else a.grad) \
                     + out.grad * out.data

    out._backward = _backward
    return out


def log(a) -> Tensor:
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
    a = _ensure(a)
    n = a.data.size if axis is None else a.data.shape[axis]
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
