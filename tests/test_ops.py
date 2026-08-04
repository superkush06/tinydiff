"""Op-level tests."""

import numpy as np

import tinydiff as td


def test_matmul_shape_and_grad():
    a = td.Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
    b = td.Tensor(np.array([[5.0, 6.0], [7.0, 8.0]]), requires_grad=True)
    c = a @ b
    loss = c.sum()
    loss.backward()
    assert a.grad.shape == a.shape
    assert b.grad.shape == b.shape


def test_relu_grad_zero_below_zero():
    a = td.Tensor(np.array([-1.0, 0.5, 2.0]), requires_grad=True)
    out = td.relu(a).sum()
    out.backward()
    assert a.grad[0] == 0.0
    assert a.grad[1] == 1.0
    assert a.grad[2] == 1.0


def test_exp_log_inverse():
    a = td.Tensor(2.0, requires_grad=True)
    out = td.log(td.exp(a))
    out.backward()
    assert abs(out.data - 2.0) < 1e-12
    assert abs(a.grad - 1.0) < 1e-9


def test_pow_grad():
    a = td.Tensor(3.0, requires_grad=True)
    out = a ** 2
    out.backward()
    assert a.grad == 6.0  # d/da a^2 = 2a


def test_broadcast_grad():
    """Adding (N,) to (M,N) should unbroadcast grad back to (N,)."""
    a = td.Tensor(np.ones((3, 4)), requires_grad=True)
    b = td.Tensor(np.ones(4), requires_grad=True)
    c = (a + b).sum()
    c.backward()
    assert a.grad.shape == (3, 4)
    assert b.grad.shape == (4,)
    # Each entry of b is broadcast across 3 rows -> grad of 3 per entry
    np.testing.assert_allclose(b.grad, np.full(4, 3.0))


def test_mean_reduces_correctly():
    a = td.Tensor(np.array([1.0, 2.0, 3.0, 4.0]), requires_grad=True)
    out = a.mean()
    out.backward()
    np.testing.assert_allclose(a.grad, np.full(4, 0.25))


def test_mean_tuple_axis():
    a = td.Tensor(np.arange(24.0).reshape(2, 3, 4), requires_grad=True)
    out = td.mean(a, axis=(0, 1))
    np.testing.assert_allclose(out.data, a.data.mean(axis=(0, 1)))
    out.sum().backward()
    np.testing.assert_allclose(a.grad, np.full((2, 3, 4), 1.0 / 6.0))


def test_mean_negative_axis():
    a = td.Tensor(np.arange(12.0).reshape(3, 4), requires_grad=True)
    out = td.mean(a, axis=-1)
    np.testing.assert_allclose(out.data, a.data.mean(axis=-1))
    out.sum().backward()
    np.testing.assert_allclose(a.grad, np.full((3, 4), 0.25))
