"""Pin the numbers in docs/validation.md to the code that produces them.

`examples/validate.py` prints the ledger; this module makes CI fail if the
ledger would print anything different. Two kinds of assertion here:

  * the script's own verdict per section, so a regression anywhere in the
    ledger turns the build red rather than quietly changing a table, and
  * a handful of the individual numbers, restated independently, so the
    table in the doc can be checked against this file without running
    anything.
"""

from __future__ import annotations

import numpy as np

import tinydiff as td
from examples.validate import (
    canonical,
    closed_forms,
    finite_differences,
    naive_crossentropy,
    naive_logsumexp,
    rosenbrock,
    stability,
)


def _grad(fn, *arrays):
    ts = [td.Tensor(np.array(a, dtype=float), requires_grad=True) for a in arrays]
    fn(*ts).backward()
    return [t.grad for t in ts]


# --- the four ledger sections -----------------------------------------
def test_closed_form_section_agrees():
    assert closed_forms()


def test_canonical_section_agrees():
    assert canonical()


def test_finite_difference_section_agrees():
    assert finite_differences()


def test_stability_section_agrees():
    assert stability()


# --- individual rows, restated -----------------------------------------
def test_rosenbrock_gradient_at_the_standard_start():
    """(-215.6, -88.0) at (-1.2, 1), hand-derived; f = 24.2 there."""
    gx, gy = _grad(rosenbrock, -1.2, 1.0)
    assert abs(float(gx) - (-215.6)) < 1e-12
    assert abs(float(gy) - (-88.0)) < 1e-12
    assert abs(float(rosenbrock(td.Tensor(-1.2), td.Tensor(1.0)).data) - 24.2) < 1e-12


def test_rosenbrock_gradient_vanishes_at_the_minimum():
    gx, gy = _grad(rosenbrock, 1.0, 1.0)
    assert float(gx) == 0.0 and float(gy) == 0.0


def test_cross_entropy_on_uniform_logits_is_log_c():
    for C in (2, 7, 64):
        val = td.softmax_crossentropy(td.Tensor(np.zeros((3, C))), np.arange(3) % C)
        assert abs(float(val.data) - np.log(C)) < 1e-15


def test_naive_logsumexp_overflows_where_the_doc_says_it_does():
    """709 is the last integer shift the naive form survives; 710 is inf."""
    base = np.array([0.0, -1.0, -2.0])
    with np.errstate(over="ignore"):
        assert np.isfinite(naive_logsumexp(base + 709.0))
        assert not np.isfinite(naive_logsumexp(base + 710.0))


def test_fused_cross_entropy_survives_where_the_naive_form_does_not():
    """Shifting a whole row of logits cannot change loss or gradient."""
    rng = np.random.default_rng(11)
    Z = rng.standard_normal((4, 3))
    lab = np.array([0, 2, 1, 1])
    zt = td.Tensor(Z, requires_grad=True)
    td.softmax_crossentropy(zt, lab).backward()
    base_loss = float(td.softmax_crossentropy(td.Tensor(Z), lab).data)
    base_grad = zt.grad.copy()

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        assert not np.isfinite(naive_crossentropy(Z + 710.0, lab))
    shifted = td.Tensor(Z + 710.0, requires_grad=True)
    out = td.softmax_crossentropy(shifted, lab)
    out.backward()
    assert abs(float(out.data) - base_loss) < 1e-12
    assert np.abs(shifted.grad - base_grad).max() < 1e-12


def test_confident_mistake_is_the_logit_gap():
    """logits [0, -d], true class 1: the loss is d + log(1 + e^-d)."""
    for d in (50.0, 400.0, 800.0):
        val = float(td.softmax_crossentropy(td.Tensor([[0.0, -d]]), np.array([1])).data)
        assert abs(val - (d + float(np.log1p(np.exp(-d))))) < 1e-12


def test_relu_at_zero_disagrees_with_the_central_difference_on_purpose():
    """The one documented disagreement: 0 (subgradient) vs 0.5 (symmetric)."""
    (g,) = _grad(lambda t: td.sum_(td.relu(t)), 0.0)
    assert float(g) == 0.0
    eps = 1e-6
    assert (max(0.0, eps) - max(0.0, -eps)) / (2 * eps) == 0.5


def test_central_difference_floor_is_near_eps_two_thirds():
    """The reference method's own error floor, measured, not quoted.

    docs/validation.md justifies tol=1e-6 by claiming the central-difference
    floor sits near eps_mach^(2/3) ~ 4e-11. This measures it on x tanh x,
    whose derivative is written out by hand, and takes the median across the
    trough because round-off makes the bottom of the U a sawtooth.
    """
    def f(x):
        return x * np.tanh(x)

    x0 = 0.7
    exact = float(np.tanh(x0) + x0 * (1.0 - np.tanh(x0) ** 2))
    hs = np.logspace(-6.5, -4.5, 121)
    err = np.array([abs((f(x0 + h) - f(x0 - h)) / (2 * h) - exact) / abs(exact)
                    for h in hs])
    floor = float(np.median(err))
    assert 1e-12 < floor < 1e-9
    assert np.finfo(float).eps ** (2 / 3) / 100 < floor < np.finfo(float).eps ** (2 / 3) * 100
