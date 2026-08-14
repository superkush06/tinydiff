"""The branches the happy path never reaches.

Everything here was an uncovered line before this file existed. Two of them
are load-bearing rather than tidy: `grad_check` had never been shown to
return False for a *wrong* gradient (only for a missing one), and the
optimizers' "this parameter has no gradient" branch is what stops a frozen
or unused parameter from crashing a training loop. A checker whose failure
path is untested is a checker you are trusting on faith.
"""

import numpy as np
import pytest

import tinydiff as td
from tinydiff.nn import Module
from tinydiff.optim import Optimizer
from tinydiff.tensor import Tensor


# --- grad_check's failure path ----------------------------------------
def _wrong_sum_of_squares(a):
    """sum(a**2) with a deliberately wrong VJP: ones instead of 2a."""
    out = Tensor(float((a.data ** 2).sum()), _children=(a,), _op="wrong")

    def _backward():
        if a.requires_grad:
            g = np.ones_like(a.data) * out.grad
            a.grad = g if a.grad is None else a.grad + g

    out._backward = _backward
    return out


def test_gradcheck_rejects_a_wrong_vjp(capsys):
    """The whole point of the checker: a plausible-but-wrong VJP must fail."""
    assert not td.grad_check(_wrong_sum_of_squares, np.array([1.0, 2.0, 3.0]))
    assert "grad mismatch" in capsys.readouterr().out


def test_gradcheck_accepts_the_same_function_written_correctly():
    """Control for the test above — the shape of the call is not what fails."""
    assert td.grad_check(lambda a: (a * a).sum(), np.array([1.0, 2.0, 3.0]))


# --- Module plumbing ---------------------------------------------------
class _Nested(Module):
    """Parameters reachable three ways: attribute, submodule, list."""

    def __init__(self):
        self.w = Tensor(np.ones((2, 2)), requires_grad=True)
        self.frozen = Tensor(np.ones(2))  # requires_grad=False -> not a parameter
        self.child = td.Linear(2, 3, rng=np.random.default_rng(0))
        self.stack = [td.Linear(3, 1, rng=np.random.default_rng(1)),
                      Tensor(np.ones(4), requires_grad=True),
                      "not a parameter"]

    def forward(self, x):
        return self.child(x)


def test_parameters_finds_attributes_submodules_and_lists():
    shapes = sorted(p.shape for p in _Nested().parameters())
    #  w        child.W  child.b  stack[0].W  stack[0].b  stack[1]
    assert shapes == [(1,), (2, 2), (2, 3), (3,), (3, 1), (4,)]


def test_parameters_skips_tensors_that_do_not_require_grad():
    net = _Nested()
    assert all(p.requires_grad for p in net.parameters())
    assert not any(p is net.frozen for p in net.parameters())


def test_module_zero_grad_clears_every_parameter():
    net = _Nested()
    net(td.Tensor(np.ones((5, 2)))).sum().backward()
    assert any(p.grad is not None for p in net.parameters())
    net.zero_grad()
    assert all(p.grad is None for p in net.parameters())


def test_module_forward_is_abstract():
    with pytest.raises(NotImplementedError):
        Module()(td.Tensor(1.0))


def test_optimizer_step_is_abstract():
    with pytest.raises(NotImplementedError):
        Optimizer([]).step()


# --- optimizers on parameters that got no gradient ---------------------
@pytest.mark.parametrize("make_opt", [
    lambda ps: td.SGD(ps, lr=0.1, momentum=0.9),
    lambda ps: td.Adam(ps, lr=0.1),
])
def test_optimizers_leave_gradientless_parameters_alone(make_opt):
    """A parameter off the forward path must not move, and must not crash."""
    used = td.Tensor(np.array([1.0]), requires_grad=True)
    unused = td.Tensor(np.array([7.0]), requires_grad=True)
    opt = make_opt([used, unused])

    (used * used).sum().backward()
    opt.step()

    assert used.data[0] != 1.0
    assert unused.data[0] == 7.0
    assert unused.grad is None


# --- traversal and seeding --------------------------------------------
def test_shared_node_is_visited_once():
    """A fan-out node appears once in the topological order, not twice."""
    x = td.Tensor(2.0, requires_grad=True)
    shared = x * 3.0
    root = shared + shared
    order = root._toposort()
    assert len(order) == len({id(t) for t in order})
    assert sum(t is shared for t in order) == 1
    root.backward()
    assert float(x.grad) == 6.0  # both paths accumulate


def test_nonscalar_backward_without_a_seed_raises():
    """Seeding a vector output with an implicit 1.0 would answer a different question."""
    t = td.Tensor(np.ones(3), requires_grad=True)
    with pytest.raises(RuntimeError, match="requires explicit"):
        (t * 2.0).backward()
