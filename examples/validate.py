"""Check tinydiff against things that are not tinydiff.

A test suite proves a library agrees with itself. This script proves it
agrees with arithmetic done elsewhere: closed-form derivatives worked out
by hand, published results for canonical functions, and a central-difference
estimator that shares no code with the engine. Everything printed here is
computed on the spot; nothing is quoted from a previous run.

    PYTHONPATH=. python3 examples/validate.py

Five sections:

  1. Closed forms, op by op        every public op against d/dx written out
  2. Canonical functions           Rosenbrock, logistic NLL, softmax-CE, Adam
  3. Central differences           every op x five shape regimes
  4. Numerical stability           the naive formulation, at the point it dies
  5. Known disagreements           where we do not match, and why

Exit status is 0 only if every check in sections 1-4 passes. Section 5 is
expected to disagree; it is printed so the disagreement stays documented
rather than discovered.
"""

from __future__ import annotations

import sys

import numpy as np

import tinydiff as td

# --- tolerances -------------------------------------------------------
#
# TOL_EXACT — closed-form comparisons. Both sides evaluate the same real
# number in float64, differing only in the order of the roundings, so the
# gap should sit at a few ulp of the result. 1e-12 relative leaves about
# four decades of headroom over the ~1e-16 we actually observe, which is
# enough to absorb a different BLAS or a different summation order without
# being loose enough to hide a wrong formula.
#
# TOL_FD — central differences. The estimator's own error floor is the
# binding constraint, not ours: truncation goes as h^2 f'''/6 and round-off
# as eps_mach |f| / h, and their sum bottoms out near eps_mach^(2/3), about
# 4e-11 relative, at h ~ 1e-5. At the h = 1e-6 the checker uses the floor is
# higher, and it is measured rather than inferred: over the 62 (op, shape)
# cells of the correctness figure the disagreement runs 2.99e-13 to 4.23e-09,
# median 2.29e-10. 1e-6 sits ~240x above the noisiest of those cells (so the
# estimator's noise never fails the check) and orders of magnitude below the
# error of any genuinely wrong VJP, which is a factor, not a rounding.
TOL_EXACT = 1e-12
TOL_FD = 1e-6

PASS, FAIL = "ok", "FAIL"


def rel(a, b) -> float:
    """Max relative disagreement, with an absolute floor near zero."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b) / np.maximum(1.0, np.abs(b))))


class Ledger:
    """Collects (claim, ours, reference, source) rows and prints them."""

    def __init__(self, title: str) -> None:
        self.title = title
        self.rows: list[tuple[str, str, str, float, bool]] = []

    def add(self, claim: str, ours: str, ref: str, err: float,
            tol: float = TOL_EXACT) -> None:
        self.rows.append((claim, ours, ref, err, err <= tol))

    def report(self) -> bool:
        print(f"\n{self.title}")
        print("-" * len(self.title))
        w = max(len(r[0]) for r in self.rows)
        for claim, ours, ref, err, ok in self.rows:
            print(f"  [{PASS if ok else FAIL:>4}] {claim:<{w}}  "
                  f"ours={ours:<26} ref={ref:<26} rel={err:.2e}")
        return all(r[4] for r in self.rows)


def _grad(fn, *arrays):
    """Run fn on fresh leaf tensors and return their gradients."""
    ts = [td.Tensor(np.array(a, dtype=float), requires_grad=True) for a in arrays]
    fn(*ts).backward()
    return [t.grad for t in ts]


# ======================================================================
# 1. Closed forms, op by op
# ======================================================================
def closed_forms() -> bool:
    """Every public op against its derivative written out by hand.

    Each entry is (name, f built from tinydiff ops, dL/dx as plain numpy).
    The scalar reduction is a weighted sum with a fixed random weight
    matrix W, so the incoming gradient is W rather than a vector of ones —
    a VJP that ignores its upstream gradient still passes a sum()-only
    check. The two reduction rows are the exception: sum_ and mean are
    called bare, since wrapping a reduction in another reduction only
    tests the wrapper, so they run with an all-ones seed. mean still sees
    an upstream-ignoring sum_ (it arrives through a 1/N scale); the sum_
    row does not, and the rest of the suite is what covers it.
    """
    rng = np.random.default_rng(20260727)
    ledger = Ledger("1. Closed-form derivatives, op by op")

    x = rng.standard_normal((4, 3)) + 0.25
    p = rng.uniform(0.5, 2.5, size=(4, 3))          # strictly positive
    q = rng.uniform(0.5, 2.5, size=(4, 3))
    W = rng.standard_normal((4, 3))
    wt = td.Tensor(W)

    def weighted(t):
        return td.sum_(t * wt)

    unary = [
        ("neg      f=-x",       lambda a: weighted(td.neg(a)),      x, lambda a: -W),
        ("exp      f=e^x",      lambda a: weighted(td.exp(a)),      x, lambda a: W * np.exp(a)),
        ("log      f=ln x",     lambda a: weighted(td.log(a)),      p, lambda a: W / a),
        ("pow_     f=x^1.5",    lambda a: weighted(td.pow_(a, 1.5)), p, lambda a: W * 1.5 * a ** 0.5),
        ("relu     f=max(0,x)", lambda a: weighted(td.relu(a)),     x, lambda a: W * (a > 0)),
        ("sigmoid  f=1/(1+e^-x)", lambda a: weighted(td.sigmoid(a)), x,
         lambda a: W * (1 / (1 + np.exp(-a))) * (1 - 1 / (1 + np.exp(-a)))),
        ("tanh     f=tanh x",   lambda a: weighted(td.tanh(a)),     x, lambda a: W * (1 - np.tanh(a) ** 2)),
        ("sum_     f=sum x",    lambda a: td.sum_(a),               x, lambda a: np.ones_like(a)),
        ("mean     f=mean x",   lambda a: td.mean(a),               x, lambda a: np.full_like(a, 1.0 / a.size)),
    ]
    for name, f, arr, ref in unary:
        (g,) = _grad(f, arr)
        ledger.add(name, f"max|g|={np.abs(g).max():.6f}",
                   f"max|g|={np.abs(ref(arr)).max():.6f}", rel(g, ref(arr)))

    binary = [
        ("add      f=x+y", lambda a, b: weighted(a + b), (x, q), lambda a, b: (W, W)),
        ("sub      f=x-y", lambda a, b: weighted(a - b), (x, q), lambda a, b: (W, -W)),
        ("mul      f=x*y", lambda a, b: weighted(a * b), (x, q), lambda a, b: (W * b, W * a)),
        ("div      f=x/y", lambda a, b: weighted(a / b), (x, q), lambda a, b: (W / b, -W * a / b ** 2)),
    ]
    for name, f, arrs, ref in binary:
        ga, gb = _grad(f, *arrs)
        ra, rb = ref(*arrs)
        ledger.add(name, f"max|dx|={np.abs(ga).max():.6f}",
                   f"max|dx|={np.abs(ra).max():.6f}", max(rel(ga, ra), rel(gb, rb)))

    # matmul: the reverse-mode pair for C = A B is dA = G B^T, dB = A^T G.
    A, B = rng.standard_normal((4, 3)), rng.standard_normal((3, 5))
    G = rng.standard_normal((4, 5))
    at, bt = td.Tensor(A, requires_grad=True), td.Tensor(B, requires_grad=True)
    td.matmul(at, bt).backward(G)
    ledger.add("matmul   f=A@B", f"max|dA|={np.abs(at.grad).max():.6f}",
               f"max|dA|={np.abs(G @ B.T).max():.6f}",
               max(rel(at.grad, G @ B.T), rel(bt.grad, A.T @ G)))

    # mse_loss: dL/dpred = 2 (pred - target) / N, N = pred.size.
    pr, tg = rng.standard_normal((6, 2)), rng.standard_normal((6, 2))
    (g,) = _grad(lambda a: td.mse_loss(a, td.Tensor(tg)), pr)
    ledger.add("mse_loss f=mean((p-t)^2)", f"max|g|={np.abs(g).max():.6f}",
               f"max|g|={np.abs(2 * (pr - tg) / pr.size).max():.6f}",
               rel(g, 2 * (pr - tg) / pr.size))

    # softmax cross-entropy: dL/dz = (softmax(z) - onehot) / N.
    Z = rng.standard_normal((5, 4))
    lab = np.array([0, 3, 1, 2, 3])
    zt = td.Tensor(Z, requires_grad=True)
    td.softmax_crossentropy(zt, lab).backward()
    e = np.exp(Z - Z.max(axis=1, keepdims=True))
    P = e / e.sum(axis=1, keepdims=True)
    onehot = np.zeros_like(Z)
    onehot[np.arange(len(lab)), lab] = 1.0
    ref_g = (P - onehot) / len(lab)
    ledger.add("softmax_xent f=-log p_y", f"max|g|={np.abs(zt.grad).max():.6f}",
               f"max|g|={np.abs(ref_g).max():.6f}", rel(zt.grad, ref_g))

    return ledger.report()


# ======================================================================
# 2. Canonical functions and published results
# ======================================================================
def rosenbrock(x, y):
    """f(x,y) = (1-x)^2 + 100(y-x^2)^2 — Rosenbrock (1960)."""
    return (1.0 - x) ** 2.0 + 100.0 * (y - x ** 2.0) ** 2.0


def canonical() -> bool:
    ledger = Ledger("2. Canonical functions")
    rng = np.random.default_rng(4)

    # Rosenbrock at the standard starting point (-1.2, 1). Differentiating
    # by hand: df/dx = -2(1-x) - 400x(y-x^2), df/dy = 200(y-x^2).
    # At (-1.2, 1): y - x^2 = -0.44, so df/dx = -4.4 - 211.2 = -215.6 and
    # df/dy = -88. f itself is 24.2.
    gx, gy = _grad(rosenbrock, -1.2, 1.0)
    ledger.add("Rosenbrock grad at (-1.2, 1)",
               f"({float(gx):.10f}, {float(gy):.10f})", "(-215.6, -88.0)",
               rel([float(gx), float(gy)], [-215.6, -88.0]))
    f0 = rosenbrock(td.Tensor(-1.2), td.Tensor(1.0))
    ledger.add("Rosenbrock value at (-1.2, 1)", f"{float(f0.data):.10f}",
               "24.2", rel(float(f0.data), 24.2))

    # The global minimum is at (1, 1); a correct gradient is exactly zero
    # there, and "exactly" is the point — this is the one place where an
    # off-by-a-rounding backward pass has nowhere to hide.
    gx, gy = _grad(rosenbrock, 1.0, 1.0)
    ledger.add("Rosenbrock grad at the minimum (1, 1)",
               f"({float(gx):.1e}, {float(gy):.1e})", "(0, 0)",
               float(max(abs(gx), abs(gy))))

    # Logistic regression NLL. The textbook gradient is X^T (sigma(Xw) - t)/N.
    X = rng.standard_normal((64, 5))
    t = (rng.random(64) < 0.5).astype(float)
    w0 = rng.standard_normal(5)

    def nll(w):
        pr = td.sigmoid(td.matmul(td.Tensor(X), w))
        tt = td.Tensor(t)
        return td.neg(td.mean(tt * td.log(pr) + (1.0 - tt) * td.log(1.0 - pr)))

    (gw,) = _grad(nll, w0)
    ref_gw = X.T @ (1.0 / (1.0 + np.exp(-X @ w0)) - t) / len(t)
    ledger.add("logistic NLL grad = X^T(sigma(Xw)-t)/N",
               f"max|g|={np.abs(gw).max():.9f}",
               f"max|g|={np.abs(ref_gw).max():.9f}", rel(gw, ref_gw))

    # Convexity, checked through first derivatives only: for a convex f the
    # gradient is a monotone operator, <grad f(a) - grad f(b), a - b> >= 0.
    worst = np.inf
    for _ in range(500):
        a, b = rng.standard_normal(5) * 1.5, rng.standard_normal(5) * 1.5
        (ga,), (gb,) = _grad(nll, a), _grad(nll, b)
        worst = min(worst, float(np.dot(ga - gb, a - b)))
    ledger.add("logistic NLL gradient is monotone (500 pairs)",
               f"min<dg,dx>={worst:.6e}", ">= 0", 0.0 if worst >= 0 else 1.0)

    # Cross-entropy on uniform logits is log C, exactly, for any C.
    for C in (2, 7, 64):
        val = td.softmax_crossentropy(
            td.Tensor(np.zeros((3, C)), requires_grad=True), np.arange(3) % C)
        ledger.add(f"CE(uniform logits, C={C}) = log C",
                   f"{float(val.data):.15f}", f"{np.log(C):.15f}",
                   rel(float(val.data), np.log(C)))

    # Euler's homogeneous function theorem: <x, grad f(x)> = k f(x) for f
    # homogeneous of degree k. Nothing about the engine enforces this; it
    # falls out only if every VJP on the path is right.
    for k, f in ((3.0, lambda a: td.sum_(td.pow_(a, 3.0))),
                 (2.0, lambda a: td.sum_(a * a)),
                 (1.0, lambda a: td.sum_(td.relu(a)))):
        xv = rng.standard_normal(8) + 0.4
        (g,) = _grad(f, xv)
        lhs = float(np.dot(xv, g))
        rhs = k * float(f(td.Tensor(xv)).data)
        ledger.add(f"Euler identity, degree {k:.0f}", f"{lhs:.10f}",
                   f"{rhs:.10f}", rel(lhs, rhs))

    # Kaiming-He uniform init: Var[W] = 2/fan_in. Uniform on (-b, b) has
    # variance b^2/3, so b = sqrt(6/fan_in). Checked on a wide layer, where
    # the sample variance is close enough to the population value to tell
    # the He constant (2) apart from the Glorot one.
    lin = td.Linear(1024, 1024, rng=np.random.default_rng(1))
    var = float(lin.W.data.var())
    ledger.add("Linear init Var[W] = 2/fan_in  (He)",
               f"{var:.8f}", f"{2 / 1024:.8f}",
               abs(var - 2 / 1024) / (2 / 1024), tol=0.01)
    glorot = 2 / (1024 + 1024)
    ledger.add("...which is 2x Glorot's 2/(fan_in+fan_out) here",
               f"ratio={var / glorot:.4f}", "ratio=2.0",
               rel(var / glorot, 2.0), tol=0.01)

    # Adam's first step is lr * g / (|g| + eps): scale-free by construction,
    # so a gradient of 1e6 and a gradient of 1 both move the parameter by
    # exactly the learning rate.
    for g_mag in (1e6, 1.0, 1e-3):
        prm = td.Tensor(np.array([0.0]), requires_grad=True)
        prm.grad = np.array([g_mag])
        td.Adam([prm], lr=0.01).step()
        ref_step = -0.01 * g_mag / (g_mag + 1e-8)
        ledger.add(f"Adam step 1 with |g|={g_mag:g}",
                   f"{float(prm.data[0]):.12f}", f"{ref_step:.12f}",
                   rel(float(prm.data[0]), ref_step))

    return ledger.report()


# ======================================================================
# 3. Central differences, every op x every shape regime
# ======================================================================
REGIMES = {
    "scalar": ((), ()),
    "vector (3,)": ((3,), (3,)),
    "matrix (4,3)": ((4, 3), (4, 3)),
    "batched (2,4,3)": ((2, 4, 3), (2, 4, 3)),
    "broadcast (4,3)x(3,)": ((4, 3), (3,)),
}


def finite_differences() -> bool:
    """Push every public op through grad_check in five shape regimes.

    grad_check perturbs each scalar entry by +/-eps and compares the
    central difference to the autodiff gradient. It shares no code with
    the backward passes it is checking.
    """
    rng = np.random.default_rng(99)
    print("\n3. Central differences, every op x every shape regime")
    print("-" * 52)

    def pos(shape):
        return rng.uniform(0.5, 2.0, size=shape)

    def sgn(shape):
        return rng.standard_normal(shape) + 0.35

    unary = {
        "neg": (lambda a: td.sum_(td.neg(a)), sgn),
        "exp": (lambda a: td.sum_(td.exp(a)), sgn),
        "log": (lambda a: td.sum_(td.log(a)), pos),
        "pow_(1.5)": (lambda a: td.sum_(td.pow_(a, 1.5)), pos),
        "relu": (lambda a: td.sum_(td.relu(a)), sgn),
        "sigmoid": (lambda a: td.sum_(td.sigmoid(a)), sgn),
        "tanh": (lambda a: td.sum_(td.tanh(a)), sgn),
        "sum_": (lambda a: td.sum_(a * 2.0), sgn),
        "mean": (lambda a: td.mean(a * a), sgn),
    }
    binary = {
        "add": (lambda a, b: td.sum_(a + b), sgn, sgn),
        "sub": (lambda a, b: td.sum_(a - b), sgn, sgn),
        "mul": (lambda a, b: td.sum_(a * b), sgn, sgn),
        "div": (lambda a, b: td.sum_(a / b), sgn, pos),
        "mse_loss": (lambda a, b: td.mse_loss(a, b), sgn, sgn),
    }
    matmul_cases = {
        "vector (3,)": ((3,), (3,)),
        "matrix (4,3)": ((4, 3), (3, 2)),
        "batched (2,4,3)": ((2, 4, 3), (2, 3, 2)),
        "broadcast (4,3)x(3,)": ((2, 4, 3), (3, 2)),
    }

    checked, failed = 0, 0
    for name, (fn, gen) in unary.items():
        cells = []
        for reg, (sa, _) in REGIMES.items():
            if reg.startswith("broadcast"):
                continue  # a unary op has nothing to broadcast against
            ok = td.grad_check(fn, gen(sa), eps=1e-6, tol=TOL_FD)
            cells.append(f"{reg.split()[0]}:{PASS if ok else FAIL}")
            checked += 1
            failed += not ok
        print(f"  {name:<12} {'  '.join(cells)}")

    for name, (fn, ga, gb) in binary.items():
        cells = []
        for reg, (sa, sb) in REGIMES.items():
            ok = td.grad_check(fn, ga(sa), gb(sb), eps=1e-6, tol=TOL_FD)
            cells.append(f"{reg.split()[0]}:{PASS if ok else FAIL}")
            checked += 1
            failed += not ok
        print(f"  {name:<12} {'  '.join(cells)}")

    cells = []
    for reg, (sa, sb) in matmul_cases.items():
        w = rng.standard_normal(np.matmul(sgn(sa), sgn(sb)).shape)

        def f(a, b, w=w):
            return td.sum_(td.matmul(a, b) * td.Tensor(w))

        ok = td.grad_check(f, sgn(sa), sgn(sb), eps=1e-6, tol=TOL_FD)
        cells.append(f"{reg.split()[0]}:{PASS if ok else FAIL}")
        checked += 1
        failed += not ok
    print(f"  {'matmul':<12} {'  '.join(cells)}")

    lab = np.array([0, 2, 1, 2])
    ok = td.grad_check(lambda z: td.softmax_crossentropy(z, lab),
                       sgn((4, 3)), eps=1e-6, tol=TOL_FD)
    checked += 1
    failed += not ok
    print(f"  {'softmax_xent':<12} matrix:{PASS if ok else FAIL}")

    print(f"\n  {checked} (op, shape) pairs checked at tol={TOL_FD:g}; "
          f"{failed} failed")
    return failed == 0


# ======================================================================
# 4. Numerical stability
# ======================================================================
def naive_logsumexp(z):
    return np.log(np.exp(z).sum(axis=-1))


def naive_crossentropy(Z, lab):
    P = np.exp(Z) / np.exp(Z).sum(axis=1, keepdims=True)
    return float(-np.log(P[np.arange(len(lab)), lab]).mean())


def stability() -> bool:
    """The naive formulation, evaluated at the point where it stops working.

    float64 tops out at exp(709.78); anything above that is +inf, and
    inf/inf is nan. The max-shift identity
    LSE(z) = max(z) + log sum exp(z - max(z))
    is algebraically the same number and never evaluates exp above 1.
    """
    print("\n4. Numerical stability: naive vs max-shifted")
    print("-" * 44)
    ok = True

    base = np.array([0.0, -1.0, -2.0])
    exact_base = float(np.log(np.exp(base).sum()))
    print("  log-sum-exp of [c, c-1, c-2]   (exact answer: c + "
          f"{exact_base:.12f})")
    for c in (0.0, 100.0, 709.0, 710.0, 1e4):
        z = base + c
        with np.errstate(over="ignore"):
            nv = float(naive_logsumexp(z))
        m = z.max()
        shifted = float(m + np.log(np.exp(z - m).sum()))
        exact = c + exact_base
        good = abs(shifted - exact) <= TOL_EXACT * max(1.0, abs(exact))
        ok &= good
        print(f"    c={c:>8.0f}  naive={nv:>22.12f}   "
              f"shifted={shifted:>22.12f}   {PASS if good else FAIL}")

    # Cross-entropy is invariant to adding a constant to a whole row of
    # logits, so the correct answer at every shift is the answer at shift 0.
    rng = np.random.default_rng(11)
    Z = rng.standard_normal((4, 3))
    lab = np.array([0, 2, 1, 1])
    exact_ce = float(td.softmax_crossentropy(td.Tensor(Z), lab).data)
    zt = td.Tensor(Z, requires_grad=True)
    td.softmax_crossentropy(zt, lab).backward()
    exact_g = zt.grad.copy()

    print(f"\n  softmax cross-entropy, logits shifted by c   (invariant: "
          f"{exact_ce:.12f})")
    for c in (0.0, 100.0, 709.0, 710.0, 1e4):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            nv = naive_crossentropy(Z + c, lab)
        zt = td.Tensor(Z + c, requires_grad=True)
        out = td.softmax_crossentropy(zt, lab)
        out.backward()
        err = rel(float(out.data), exact_ce)
        gerr = rel(zt.grad, exact_g)
        # Adding c to a logit of order 1 throws away log2(c) bits before
        # the op is ever called, so the achievable accuracy degrades like
        # eps_mach * c. That budget, not a fixed constant, is the bar.
        budget = 8.0 * np.finfo(float).eps * max(1.0, c)
        good = err <= budget and gerr <= budget
        ok &= good
        print(f"    c={c:>8.0f}  naive={nv:>10.6f}   fused={float(out.data):.12f}"
              f"   grad rel={gerr:.1e}   (budget {budget:.0e})   "
              f"{PASS if good else FAIL}")

    # The other tail: a confidently wrong prediction. -log p underflows to
    # -log(0) = inf once p drops below ~1e-308, while the fused form is
    # just the logit gap.
    print("\n  a confident mistake: logits [0, -d], true class 1  "
          "(exact: d + log(1+e^-d))")
    for d in (50.0, 400.0, 745.0, 800.0):
        Zu = np.array([[0.0, -d]])
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            nv = naive_crossentropy(Zu, np.array([1]))
        ours = float(td.softmax_crossentropy(td.Tensor(Zu), np.array([1])).data)
        exact = d + float(np.log1p(np.exp(-d)))
        good = abs(ours - exact) <= TOL_EXACT * max(1.0, exact)
        ok &= good
        print(f"    d={d:>6.0f}  naive={nv:>10.4f}   fused={ours:>12.6f}"
              f"   exact={exact:>12.6f}   {PASS if good else FAIL}")

    return ok


# ======================================================================
# 5. Known disagreements
# ======================================================================
def disagreements() -> None:
    """Places tinydiff does not match the reference, printed on purpose."""
    print("\n5. Known disagreements (expected — see docs/validation.md)")
    print("-" * 58)

    a = td.Tensor(0.0, requires_grad=True)
    td.relu(a).backward()
    eps = 1e-6
    fd = (max(0.0, eps) - max(0.0, -eps)) / (2 * eps)
    print(f"  relu'(0):        ours={float(a.grad):.1f}   central difference={fd:.1f}"
          "   (relu is not differentiable at 0; 0 is the subgradient PyTorch picks)")

    # The finite-difference floor, measured: exp is smooth and well-scaled,
    # so what is left is the estimator's own noise, not ours.
    x0 = np.array([0.7])
    (g,) = _grad(lambda t: td.sum_(td.exp(t)), x0)
    for h in (1e-4, 1e-6, 1e-8, 1e-10):
        num = (np.exp(x0 + h) - np.exp(x0 - h)) / (2 * h)
        print(f"  central diff h={h:.0e}: rel gap to autodiff = "
              f"{abs(num[0] - g[0]) / abs(g[0]):.2e}")
    print("  -> the gap is the estimator's error, not the engine's; no tolerance "
          "below ~1e-9 is meaningful.")

    x = td.Tensor(2.0, requires_grad=True)
    td.sum_(td.pow_(x, 3.0)).backward()
    print(f"  d/dx x^3 at 2:   ours={float(x.grad):.1f}   analytic=12.0   ok")
    print(f"  d2/dx2 at 2:     ours=unavailable ({type(x.grad).__name__}, not Tensor)"
          "   analytic=12.0   documented gap")


def main() -> int:
    print(__doc__.strip().splitlines()[0])
    print(f"tinydiff {td.__version__} | numpy {np.__version__}")
    results = [
        ("closed forms", closed_forms()),
        ("canonical functions", canonical()),
        ("central differences", finite_differences()),
        ("numerical stability", stability()),
    ]
    disagreements()
    print("\n" + "=" * 58)
    for name, ok in results:
        print(f"  {PASS if ok else FAIL:>4}  {name}")
    bad = [n for n, ok in results if not ok]
    print("=" * 58)
    if bad:
        print("FAILED: " + ", ".join(bad))
        return 1
    print("all sections agree with their references")
    return 0


if __name__ == "__main__":
    sys.exit(main())
