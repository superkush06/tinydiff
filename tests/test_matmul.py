"""matmul backward across the full np.matmul gufunc shape space.

The forward pass delegates to numpy, so it always worked; every test here
targets the backward pass, which used to assume both operands are exactly
2-D and crashed (or would have silently mis-shaped grads) otherwise.
"""

import numpy as np
import pytest

import tinydiff as td


def _gradcheck_matmul(shape_a, shape_b, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(shape_a)
    b = rng.standard_normal(shape_b)
    c = np.matmul(a, b)
    # Fixed random weights give a non-trivial upstream gradient — a plain
    # .sum() (all-ones grad) can mask transposition mistakes.
    w = td.Tensor(rng.standard_normal(c.shape))

    def f(a_t, b_t):
        return (td.matmul(a_t, b_t) * w).sum()

    return td.grad_check(f, a, b)


EXPLICIT_CASES = [
    ((3,), (3,)),                # vec . vec -> scalar
    ((3,), (3, 2)),              # vec @ mat
    ((2, 3), (3,)),              # mat @ vec
    ((2, 3), (3, 4)),            # plain 2-D
    ((4, 3, 2), (2, 5)),         # stacked lhs @ plain rhs
    ((3, 2), (4, 2, 5)),         # plain lhs @ stacked rhs
    ((4, 3, 2), (4, 2, 5)),      # fully batched
    ((1, 3, 2), (4, 2, 5)),      # broadcast batch dim on lhs
    ((4, 3, 2), (1, 2, 5)),      # broadcast batch dim on rhs
    ((2, 1, 3, 2), (4, 2, 5)),   # 4-D lhs with outer broadcast
    ((3,), (4, 3, 2)),           # vec @ stacked
    ((4, 2, 3), (3,)),           # stacked @ vec
]


@pytest.mark.parametrize("shape_a,shape_b", EXPLICIT_CASES)
def test_matmul_gradcheck_explicit(shape_a, shape_b):
    assert _gradcheck_matmul(shape_a, shape_b)


def test_matmul_1d_dot_gradient_values():
    # d(a.b)/da = b and d(a.b)/db = a
    a = td.Tensor([1.0, 2.0, 3.0], requires_grad=True)
    b = td.Tensor([4.0, 5.0, 6.0], requires_grad=True)
    (a @ b).backward()
    np.testing.assert_allclose(a.grad, [4.0, 5.0, 6.0])
    np.testing.assert_allclose(b.grad, [1.0, 2.0, 3.0])


def test_matmul_grad_shapes_match_operands():
    for shape_a, shape_b in EXPLICIT_CASES:
        rng = np.random.default_rng(0)
        a = td.Tensor(rng.standard_normal(shape_a), requires_grad=True)
        b = td.Tensor(rng.standard_normal(shape_b), requires_grad=True)
        (a @ b).sum().backward()
        assert a.grad.shape == a.shape, (shape_a, shape_b)
        assert b.grad.shape == b.shape, (shape_a, shape_b)


def test_matmul_batched_matches_per_slice_2d():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((4, 3, 2))
    B = rng.standard_normal((4, 2, 5))
    a = td.Tensor(A, requires_grad=True)
    b = td.Tensor(B, requires_grad=True)
    (a @ b).sum().backward()
    for i in range(4):
        ai = td.Tensor(A[i], requires_grad=True)
        bi = td.Tensor(B[i], requires_grad=True)
        (ai @ bi).sum().backward()
        np.testing.assert_allclose(a.grad[i], ai.grad)
        np.testing.assert_allclose(b.grad[i], bi.grad)


def _random_case(rng):
    """Random (n?,k),(k,m?) pair, optionally stacked with broadcast batches."""
    k = int(rng.integers(1, 4))
    n = int(rng.integers(1, 4))
    m = int(rng.integers(1, 4))
    kind = int(rng.integers(0, 4))
    if kind == 0:
        sa, sb = (k,), (k,)
    elif kind == 1:
        sa, sb = (k,), (k, m)
    elif kind == 2:
        sa, sb = (n, k), (k,)
    else:
        sa, sb = (n, k), (k, m)
    n_batch = int(rng.integers(0, 3))
    batch = tuple(int(rng.integers(1, 4)) for _ in range(n_batch))
    if batch:
        def stacked(core):
            # Each batch axis independently kept or broadcast down to 1.
            return tuple(d if rng.random() < 0.7 else 1 for d in batch) + core
        which = int(rng.integers(0, 3))
        if which in (0, 2) and len(sa) == 2:
            sa = stacked(sa)
        if which in (1, 2) and len(sb) == 2:
            sb = stacked(sb)
    return sa, sb


@pytest.mark.parametrize("seed", range(30))
def test_matmul_gradcheck_random_shape_sweep(seed):
    rng = np.random.default_rng(seed)
    shape_a, shape_b = _random_case(rng)
    assert _gradcheck_matmul(shape_a, shape_b, seed=seed + 1000), (shape_a, shape_b)
