"""Optimizer tests."""


import tinydiff as td


def test_sgd_descends_quadratic():
    """f(x) = x^2 minimized at 0; SGD from 5.0 should converge to 0."""
    x = td.Tensor(5.0, requires_grad=True)
    opt = td.SGD([x], lr=0.1)
    for _ in range(200):
        opt.zero_grad()
        y = x * x
        y.backward()
        opt.step()
    assert abs(x.data) < 1e-3


def test_adam_descends_quadratic():
    x = td.Tensor(5.0, requires_grad=True)
    opt = td.Adam([x], lr=0.1)
    for _ in range(200):
        opt.zero_grad()
        y = x * x
        y.backward()
        opt.step()
    assert abs(x.data) < 1e-2


def test_sgd_momentum_speeds_descent():
    """Momentum should reduce iters-to-converge on a quadratic.

    Uses a small lr so momentum accelerates without overshooting; at large
    lr, momentum can oscillate (which is itself a teaching moment).
    """
    def steps_to_eps(opt_fn, lr):
        x = td.Tensor(5.0, requires_grad=True)
        opt = opt_fn([x], lr=lr)
        for k in range(1, 1000):
            opt.zero_grad()
            (x * x).backward()
            opt.step()
            if abs(x.data) < 0.05:
                return k
        return 1000
    n_plain = steps_to_eps(lambda p, lr: td.SGD(p, lr=lr, momentum=0.0), lr=0.02)
    n_mom   = steps_to_eps(lambda p, lr: td.SGD(p, lr=lr, momentum=0.85), lr=0.02)
    assert n_mom < n_plain
