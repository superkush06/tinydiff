"""Graph-engine tests: deep graphs, repeated backward, seed validation."""

import numpy as np
import pytest

import tinydiff as td


def test_deep_chain_backward_no_recursion_error():
    x = td.Tensor(1.0, requires_grad=True)
    y = x
    for _ in range(3000):
        y = y + 1.0
    y.backward()
    assert x.grad == 1.0


def test_deep_chain_zero_grad_no_recursion_error():
    x = td.Tensor(1.0, requires_grad=True)
    y = x
    for _ in range(3000):
        y = y * 1.0
    y.backward()
    y.zero_grad()
    assert x.grad is None


def test_deep_diamond_graph_backward():
    """Fan-out/fan-in chains must also traverse iteratively."""
    x = td.Tensor(1.0, requires_grad=True)
    y = x
    for _ in range(1500):
        y = (y + y) * 0.5
    y.backward()
    assert x.grad == 1.0


def test_second_backward_raises_without_retain_graph():
    x = td.Tensor(3.0, requires_grad=True)
    z = 2.0 * x * x
    z.backward()
    with pytest.raises(RuntimeError, match="retain_graph"):
        z.backward()


def test_second_backward_with_retain_graph_accumulates_leaf_grads():
    # dz/dx = 4x = 12 at x = 3; two passes must accumulate to exactly 24,
    # never compound stale intermediate grads into garbage.
    x = td.Tensor(3.0, requires_grad=True)
    z = 2.0 * x * x
    z.backward()
    assert x.grad == 12.0
    z.backward(retain_graph=True)
    assert x.grad == 24.0


def test_fresh_graph_each_step_never_needs_retain_graph():
    """The usual training pattern — rebuild the graph every iteration."""
    x = td.Tensor(3.0, requires_grad=True)
    for _ in range(3):
        x.grad = None
        z = 2.0 * x * x
        z.backward()
        assert x.grad == 12.0


def test_nonscalar_backward_seed_shape_must_match():
    t = td.Tensor(np.ones((2, 2)), requires_grad=True)
    u = t * 2.0
    with pytest.raises(ValueError, match="seed shape"):
        u.backward(1.0)


def test_matching_seed_is_accepted():
    t = td.Tensor(np.ones((2, 2)), requires_grad=True)
    u = t * 2.0
    u.backward(np.ones((2, 2)))
    np.testing.assert_allclose(t.grad, np.full((2, 2), 2.0))
