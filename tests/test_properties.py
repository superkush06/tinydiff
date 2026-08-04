"""Randomised invariant tests.

The rest of the suite checks fixtures: this input, that gradient. These
check statements that have to hold for *every* input, on inputs nobody
chose. Each test draws from a seeded `default_rng`, so a failure is
reproducible from the test name alone; the seeds are arbitrary and were
not searched over.

The invariants fall into three groups:

  calculus     linearity of the VJP, the sum rule, Euler homogeneity,
               the directional derivative against a central difference
  graph        fan-out accumulation, broadcast conservation, traversal
               order, re-entrancy accumulation
  domain       softmax stays on the simplex, cross-entropy stays above
               zero, convex losses have monotone gradients, Adam's first
               step is scale-free
"""

from __future__ import annotations

import numpy as np
import pytest

import tinydiff as td

DRAWS = 200


def grad_of(fn, *arrays):
    ts = [td.Tensor(np.array(a, dtype=float), requires_grad=True) for a in arrays]
    fn(*ts).backward()
    return [t.grad for t in ts]


def value_of(fn, *arrays):
    return float(fn(*[td.Tensor(np.array(a, dtype=float)) for a in arrays]).data)


# A pool of scalar-valued functions with mixed op coverage, used by the
# tests that must hold for "any differentiable f" rather than one f.
def _pool():
    return [
        ("tanh-mlp", lambda x, W: td.sum_(td.tanh(td.matmul(x, W)) ** 2.0)),
        ("relu-mlp", lambda x, W: td.sum_(td.relu(td.matmul(x, W) + 0.1))),
        ("logexp", lambda x, W: td.mean(td.log(td.exp(td.matmul(x, W)) + 2.0))),
        ("sigmoid", lambda x, W: td.sum_(td.sigmoid(td.matmul(x, W)) * 3.0)),
        ("div", lambda x, W: td.sum_(td.matmul(x, W) / (td.matmul(x, W) ** 2.0 + 4.0))),
    ]


# ----------------------------------------------------------------------
# calculus
# ----------------------------------------------------------------------
@pytest.mark.parametrize("name,f", _pool())
def test_directional_derivative_matches_central_difference(name, f):
    """<grad f(x), v> equals the directional derivative of f along v.

    This is the definition of the gradient, and it is the one invariant
    that cannot be satisfied by an engine that is wrong anywhere on the
    path: a single mis-derived VJP shows up as a mismatch for almost every
    random direction v. Central differences over a random direction cost
    two evaluations regardless of dimension, so this is affordable to
    repeat 200 times where a full per-coordinate check would not be.
    """
    rng = np.random.default_rng(0xD1)
    h = 1e-6
    for _ in range(DRAWS):
        x = rng.standard_normal((4, 3))
        W = rng.standard_normal((3, 5))
        gx, gW = grad_of(f, x, W)
        vx, vW = rng.standard_normal(x.shape), rng.standard_normal(W.shape)
        predicted = float((gx * vx).sum() + (gW * vW).sum())
        plus = value_of(f, x + h * vx, W + h * vW)
        minus = value_of(f, x - h * vx, W - h * vW)
        numeric = (plus - minus) / (2 * h)
        # The central-difference floor is ~1e-9 relative (see docs/theory.md
        # section 5); 1e-6 sits comfortably above it.
        assert abs(predicted - numeric) <= 1e-6 * max(1.0, abs(numeric)), name


def test_vjp_is_linear_in_the_seed():
    """backward(c * g) gives c * backward(g), for every c and every g.

    The backward pass is a linear map (a vector-Jacobian product), so
    scaling the incoming gradient must scale the outgoing one exactly.
    An op that ignores `out.grad` — a surprisingly common bug, and one
    that a sum()-only test never catches, because there the seed is all
    ones — breaks this immediately.
    """
    rng = np.random.default_rng(0xD2)
    for _ in range(DRAWS):
        A = rng.standard_normal((3, 4))
        B = rng.standard_normal((4, 2))
        G = rng.standard_normal((3, 2))
        c = float(rng.standard_normal() * 5.0)

        def run(seed, A=A, B=B):
            at = td.Tensor(A, requires_grad=True)
            bt = td.Tensor(B, requires_grad=True)
            td.tanh(td.matmul(at, bt)).backward(seed)
            return at.grad, bt.grad

        g1a, g1b = run(G)
        gca, gcb = run(c * G)
        assert np.allclose(gca, c * g1a, rtol=1e-12, atol=1e-14)
        assert np.allclose(gcb, c * g1b, rtol=1e-12, atol=1e-14)


def test_sum_rule():
    """grad(f + g) = grad(f) + grad(g), for independently drawn f and g."""
    rng = np.random.default_rng(0xD3)
    for _ in range(DRAWS):
        x = rng.standard_normal(6)

        def f(t):
            return td.sum_(td.tanh(t) * 2.0)

        def g(t):
            return td.sum_(td.sigmoid(t * 3.0))

        (gf,) = grad_of(f, x)
        (gg,) = grad_of(g, x)
        (gs,) = grad_of(lambda t: f(t) + g(t), x)
        assert np.allclose(gs, gf + gg, rtol=1e-12, atol=1e-14)


def test_euler_homogeneous_function_theorem():
    """<x, grad f(x)> = k f(x) whenever f is homogeneous of degree k.

    Euler's theorem is a statement about f, not about the engine, so it
    is a genuine external constraint: it holds only if every VJP on the
    path is correct *and* the fan-out accumulation is correct, and it
    involves no reference implementation at all.
    """
    rng = np.random.default_rng(0xD4)
    M = rng.standard_normal((5, 5))
    cases = [
        ("relu, k=1", 1.0, (5,), lambda t: td.sum_(td.relu(t))),
        ("square, k=2", 2.0, (5,), lambda t: td.sum_(t * t)),
        ("cube, k=3", 3.0, (5,), lambda t: td.sum_(td.pow_(t, 3.0))),
        ("quadratic form, k=2", 2.0, (4, 5),
         lambda t: td.sum_(td.matmul(t, td.Tensor(M)) * t)),
    ]
    for name, k, shape, f in cases:
        for _ in range(DRAWS // 4):
            x = rng.standard_normal(shape)
            (g,) = grad_of(f, x)
            lhs = float((x * g).sum())
            rhs = k * value_of(f, x)
            assert abs(lhs - rhs) <= 1e-10 * max(1.0, abs(rhs)), name


def test_relu_is_positively_homogeneous():
    """relu(cx) = c relu(x) for c > 0, and the gradient scales with it.

    ReLU's only structure is positive homogeneity of degree 1, and the
    backward mask has to be scale-free for that to reach the gradient. A
    mask taken at a fixed non-zero threshold — `a.data > 0.5` in place of
    `a.data > 0` — leaves the value identity alone and breaks this one,
    because scaling the input walks entries across the threshold; that
    mutation fails here on the first draw.

    Two mask errors this does *not* catch, because neither is a different
    function: `out.data > 0` (relu's output is positive exactly where its
    input is) and `a.data >= 0` (differs only at exactly zero, which
    `standard_normal` does not draw). `>= 0` is caught, but by
    tests/test_validation.py's relu-at-zero row, not by this test.
    """
    rng = np.random.default_rng(0xD5)
    for _ in range(DRAWS):
        x = rng.standard_normal((3, 4))
        c = float(rng.uniform(0.05, 20.0))
        (g1,) = grad_of(lambda t: td.sum_(td.relu(t) ** 2.0), x)
        (gc,) = grad_of(lambda t, c=c: td.sum_(td.relu(t * c) ** 2.0), x)
        assert np.allclose(gc, c * c * g1, rtol=1e-11, atol=1e-13)


# ----------------------------------------------------------------------
# graph
# ----------------------------------------------------------------------
def test_fanout_contributions_are_summed():
    """A tensor consumed k times receives all k contributions.

    Overwriting instead of accumulating is the classic first bug in a
    hand-rolled engine and it is invisible in any graph that happens to
    be a tree. Here x feeds k separate multiplications, so d/dx sum(k
    copies of x*a_i) must be sum(a_i) — off by a term if any contribution
    is dropped.
    """
    rng = np.random.default_rng(0xD6)
    for _ in range(DRAWS):
        k = int(rng.integers(2, 9))
        x = rng.standard_normal(5)
        coeffs = rng.standard_normal((k, 5))

        def f(t, coeffs=coeffs):
            acc = t * td.Tensor(coeffs[0])
            for row in coeffs[1:]:
                acc = acc + t * td.Tensor(row)
            return td.sum_(acc)

        (g,) = grad_of(f, x)
        assert np.allclose(g, coeffs.sum(axis=0), rtol=1e-12, atol=1e-14)


def test_broadcasting_conserves_gradient_mass():
    """The adjoint of a copy is a sum: no gradient is created or lost.

    Broadcasting b of shape (C,) against (N, C) replicates each entry N
    times. Its adjoint therefore sums over the replicated axis, and the
    total gradient mass reaching b must equal the total mass in the
    unbroadcast gradient. A conservation law, and the cheapest way to
    catch an `_unbroadcast` that sums the wrong axis.
    """
    rng = np.random.default_rng(0xD7)
    for _ in range(DRAWS):
        N, C = int(rng.integers(2, 7)), int(rng.integers(2, 7))
        a = rng.standard_normal((N, C))
        b = rng.standard_normal(C)
        W = rng.standard_normal((N, C))

        def f(at, bt, W=W):
            return td.sum_((at + bt) * td.Tensor(W))

        ga, gb = grad_of(f, a, b)
        assert ga.shape == (N, C) and gb.shape == (C,)
        assert np.allclose(gb, ga.sum(axis=0), rtol=1e-12, atol=1e-14)
        assert abs(gb.sum() - ga.sum()) <= 1e-11 * max(1.0, abs(ga.sum()))


def test_matmul_backward_is_the_adjoint_of_matmul_forward():
    """<G, A @ V> = <A^T G, V> for every G, V and every operand rank.

    The defining property of a reverse-mode rule: the backward pass is
    the adjoint of the forward pass under the Frobenius inner product.
    Stating it this way is rank-agnostic, so the same assertion covers
    vectors, matrices, stacked operands and broadcast batch dimensions —
    exactly the shape space where a hand-written `.T` goes wrong.
    """
    rng = np.random.default_rng(0xD8)
    shapes = [((3,), (3,)), ((4, 3), (3, 5)), ((3,), (3, 5)), ((4, 3), (3,)),
              ((2, 4, 3), (2, 3, 5)), ((2, 4, 3), (3, 5)), ((4, 3), (2, 3, 5))]
    for sa, sb in shapes:
        for _ in range(DRAWS // len(shapes) + 1):
            A, B = rng.standard_normal(sa), rng.standard_normal(sb)
            V = rng.standard_normal(sb)
            G = rng.standard_normal(np.matmul(A, B).shape)
            bt = td.Tensor(B, requires_grad=True)
            td.matmul(td.Tensor(A), bt).backward(G)
            lhs = float((G * np.matmul(A, V)).sum())
            rhs = float((bt.grad * V).sum())
            assert abs(lhs - rhs) <= 1e-10 * max(1.0, abs(lhs)), (sa, sb)


def test_toposort_visits_children_before_parents():
    """Every node appears after all of its inputs in the traversal order.

    The backward pass reverses this list, so the order guarantee is what
    makes "a node's gradient is complete when its closure runs" true. On
    random DAGs with heavy sharing, not chains.
    """
    rng = np.random.default_rng(0xD9)
    for _ in range(DRAWS):
        nodes = [td.Tensor(rng.standard_normal(3), requires_grad=True)
                 for _ in range(4)]
        for _ in range(int(rng.integers(6, 25))):
            i, j = rng.integers(0, len(nodes), size=2)
            op = rng.integers(0, 3)
            if op == 0:
                nodes.append(nodes[i] + nodes[j])
            elif op == 1:
                nodes.append(nodes[i] * nodes[j])
            else:
                nodes.append(td.tanh(nodes[i]))
        root = td.sum_(nodes[-1])
        order = root._toposort()
        pos = {id(t): k for k, t in enumerate(order)}
        for t in order:
            for c in t._children:
                assert pos[id(c)] < pos[id(t)]


def test_repeated_backward_accumulates_exactly_linearly():
    """k passes with retain_graph=True give exactly k times one pass.

    The interior of the graph is reset before each pass, so only the
    leaves accumulate. If interior gradients survived, the k-th pass
    would compound them and the ratio would blow up super-linearly —
    which is the failure this API exists to make impossible by accident.
    """
    rng = np.random.default_rng(0xDA)
    for _ in range(DRAWS):
        A = rng.standard_normal((3, 4))
        at = td.Tensor(A, requires_grad=True)
        out = td.sum_(td.tanh(td.matmul(at, td.Tensor(np.eye(4)))) ** 2.0)
        out.backward()
        one = at.grad.copy()
        k = int(rng.integers(2, 6))
        for _ in range(k - 1):
            out.backward(retain_graph=True)
        assert np.allclose(at.grad, k * one, rtol=1e-12, atol=1e-14)


# ----------------------------------------------------------------------
# domain
# ----------------------------------------------------------------------
def test_softmax_crossentropy_gradient_rows_sum_to_zero():
    """Each row of dL/dz sums to zero, for every logit matrix.

    softmax is invariant to adding a constant to a row of logits, so the
    derivative along the all-ones direction is identically zero. The
    gradient (p - onehot)/N inherits it because p sums to one. This is
    the simplex constraint showing up as a gradient identity, and it
    holds no matter how extreme the logits are.
    """
    rng = np.random.default_rng(0xDB)
    for _ in range(DRAWS):
        N, C = int(rng.integers(1, 8)), int(rng.integers(2, 9))
        scale = float(rng.uniform(0.1, 300.0))
        Z = rng.standard_normal((N, C)) * scale
        lab = rng.integers(0, C, size=N)
        zt = td.Tensor(Z, requires_grad=True)
        td.softmax_crossentropy(zt, lab).backward()
        assert np.abs(zt.grad.sum(axis=1)).max() < 1e-14


def test_softmax_crossentropy_is_translation_invariant():
    """Adding a per-row constant to the logits changes neither loss nor grad.

    The property the max-shift trick relies on. Tested with shifts large
    enough that a naive exp(z) would be +inf, which is the whole point:
    the invariance must survive the regime where the naive formulation
    stops producing a number at all.
    """
    rng = np.random.default_rng(0xDC)
    for _ in range(DRAWS):
        N, C = int(rng.integers(1, 6)), int(rng.integers(2, 7))
        Z = rng.standard_normal((N, C))
        lab = rng.integers(0, C, size=N)
        shift = rng.uniform(-800.0, 800.0, size=(N, 1))

        def run(logits, lab=lab):
            t = td.Tensor(logits, requires_grad=True)
            out = td.softmax_crossentropy(t, lab)
            out.backward()
            return float(out.data), t.grad

        v0, g0 = run(Z)
        v1, g1 = run(Z + shift)
        budget = 8.0 * np.finfo(float).eps * max(1.0, float(np.abs(shift).max()))
        assert abs(v1 - v0) <= budget * max(1.0, abs(v0))
        assert np.abs(g1 - g0).max() <= budget


def test_crossentropy_is_bounded_below_by_zero_and_by_log_c_when_correct():
    """0 <= L <= log C on any row whose argmax logit is the label.

    Cross-entropy is -log of a probability, so it is non-negative; and a
    row that already ranks the true class first has p_y >= 1/C, hence
    -log p_y <= log C. Both bounds are properties of the softmax simplex
    and must hold for arbitrary logits.

    In exact arithmetic the lower bound is strict. In float64 it is not:
    once the winning logit leads by more than about log(1/eps) = 36 nats,
    p_y rounds to exactly 1.0 and the loss is exactly 0. That is a
    representable-number fact, not an engine bug, so the test asserts the
    weak bound and then pins the margin at which equality is allowed.
    """
    rng = np.random.default_rng(0xDD)
    saturated = 0
    for _ in range(DRAWS):
        C = int(rng.integers(2, 12))
        Z = rng.standard_normal((1, C)) * float(rng.uniform(0.1, 50.0))
        lab = np.array([int(Z.argmax())])
        loss = float(td.softmax_crossentropy(td.Tensor(Z), lab).data)
        assert loss >= 0.0
        assert loss <= np.log(C) + 1e-12
        if loss == 0.0:
            saturated += 1
            top2 = np.sort(Z[0])[-2:]
            assert top2[1] - top2[0] > 35.0
    assert saturated < DRAWS  # the interesting cases must not be all of them


def test_convex_loss_has_a_monotone_gradient():
    """<grad f(a) - grad f(b), a - b> >= 0 for the logistic NLL.

    Equivalent to a positive semi-definite Hessian everywhere, which is
    the convexity of logistic regression — a fact about the loss that
    tinydiff knows nothing about. tinydiff cannot form a Hessian at all
    (no higher-order grads), so this is how PSD-ness gets checked here:
    through first derivatives only.
    """
    rng = np.random.default_rng(0xDE)
    X = rng.standard_normal((48, 4))
    t = (rng.random(48) < 0.5).astype(float)

    def nll(w):
        p = td.sigmoid(td.matmul(td.Tensor(X), w))
        tt = td.Tensor(t)
        return td.neg(td.mean(tt * td.log(p) + (1.0 - tt) * td.log(1.0 - p)))

    for _ in range(DRAWS):
        a, b = rng.standard_normal(4) * 1.5, rng.standard_normal(4) * 1.5
        (ga,) = grad_of(nll, a)
        (gb,) = grad_of(nll, b)
        assert float(np.dot(ga - gb, a - b)) >= -1e-12


def test_negative_gradient_is_a_descent_direction():
    """f(x - t grad f(x)) < f(x) for small enough t, whenever grad != 0.

    The reason a gradient is worth computing. It follows from the
    directional-derivative property, but it is worth asserting on its own
    because a sign error in a single VJP survives every symmetric check
    and fails this one on the first draw.
    """
    rng = np.random.default_rng(0xDF)
    for _ in range(DRAWS):
        x = rng.standard_normal(6)
        W = rng.standard_normal((6, 3))

        def f(t, W=W):
            return td.sum_(td.tanh(td.matmul(t, td.Tensor(W))) ** 2.0) + td.sum_(t * t)

        (g,) = grad_of(f, x)
        if np.abs(g).max() < 1e-8:
            continue
        step = 1e-4 / max(1.0, float(np.abs(g).max()))
        assert value_of(f, x - step * g) < value_of(f, x)


def test_adam_first_step_is_scale_free():
    """Adam's opening step is lr * g / (|g| + eps), whatever |g| is.

    The bias correction makes m_hat = g and v_hat = g^2 at t = 1 exactly,
    so the update collapses to the learning rate times the sign of the
    gradient, damped only by eps. That scale-invariance is Adam's whole
    selling point and it is a closed form worth pinning: a rescaling of
    the loss must not rescale the first step.
    """
    rng = np.random.default_rng(0xE0)
    for _ in range(DRAWS):
        g = float(rng.standard_normal() * 10.0 ** rng.uniform(-6, 6))
        lr = float(rng.uniform(1e-4, 0.5))
        p = td.Tensor(np.array([0.0]), requires_grad=True)
        p.grad = np.array([g])
        td.Adam([p], lr=lr).step()
        expected = -lr * g / (abs(g) + 1e-8)
        assert abs(float(p.data[0]) - expected) <= 1e-12 * max(1.0, abs(expected))


def test_mse_is_nonnegative_and_zero_exactly_when_equal():
    """MSE >= 0, is 0 iff pred == target, and its gradient vanishes there."""
    rng = np.random.default_rng(0xE1)
    for _ in range(DRAWS):
        shape = (int(rng.integers(1, 5)), int(rng.integers(1, 5)))
        target = rng.standard_normal(shape)
        pred = rng.standard_normal(shape)
        assert value_of(lambda p, tg=target: td.mse_loss(p, td.Tensor(tg)), pred) >= 0.0
        same = td.Tensor(target.copy(), requires_grad=True)
        out = td.mse_loss(same, td.Tensor(target))
        out.backward()
        assert float(out.data) == 0.0
        assert np.abs(same.grad).max() == 0.0
