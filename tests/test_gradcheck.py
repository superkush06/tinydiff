"""Numerical vs analytical gradient check on small graphs."""

import numpy as np

import tinydiff as td


def test_gradcheck_quadratic():
    def f(a):
        return (a * a).sum()
    a = td.Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
    assert td.grad_check(f, a)


def test_gradcheck_linear_layer():
    rng = np.random.default_rng(0)
    W = td.Tensor(rng.standard_normal((3, 2)), requires_grad=True)
    b = td.Tensor(rng.standard_normal(2), requires_grad=True)
    x = td.Tensor(rng.standard_normal((4, 3)))
    target = td.Tensor(rng.standard_normal((4, 2)))

    def f(W, b):
        y = x @ W + b
        return td.mse_loss(y, target)

    assert td.grad_check(f, W, b)
