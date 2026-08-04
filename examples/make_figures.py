"""Regenerate the figures in `docs/` from real runs of the library.

Nothing here is illustrative: every number plotted is measured on the spot
by calling tinydiff. Two figures come out.

`docs/engine_report.png` — the three things an autodiff engine has to get
right, each measured:

  1. Cost.       One backward pass returns all n partials. Finite
                 differences needs 2n forward passes. Timed head to head.
  2. Correctness. Max relative error between autodiff and central
                 differences, for every public op across the shape regimes
                 (scalar / vector / matrix / batched / broadcast).
  3. Depth.      Wall time of a k-op chain, from k = 10 to k = 100,000,
                 with the depth at which a recursive graph walk stops
                 measured rather than assumed.

`docs/spiral_training.png` — the engine driving a real fit: two interleaved
spirals, the decision boundary it learns, and the per-layer gradient
magnitudes that got it there.

Run:  PYTHONPATH=. python3 examples/make_figures.py
Deps: matplotlib (``pip install -e ".[plot]"``).
"""

from __future__ import annotations

import argparse
import gc
import pathlib
import time

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

import tinydiff as td  # noqa: E402
from examples.spiral import spiral  # noqa: E402

DOCS = pathlib.Path(__file__).resolve().parents[1] / "docs"

INK = "#20242b"
MUTED = "#8b929c"
BLUE = "#1a5fb4"
RUST = "#b4460f"
GREEN = "#357a38"
GRID = "#dfe3e8"

DEFAULT_WIDTHS = (4, 8, 16, 32, 64, 128, 256, 512, 1024)
DEFAULT_DEPTHS = (10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000)


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "axes.linewidth": 0.8,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "font.family": "DejaVu Sans",
    })


def tidy(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.55)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------
# Panel 1 — the cost of a full gradient
# ---------------------------------------------------------------------
def _time(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def cost_sweep(widths=DEFAULT_WIDTHS):
    """Time one backward pass against a full central-difference sweep.

    The model is a 1-hidden-layer tanh MLP on fixed data; widening the
    hidden layer is how we grow n, the number of scalar parameters.
    """
    rng = np.random.default_rng(0)
    d, n_obs = 8, 64
    X, Y = rng.standard_normal((n_obs, d)), rng.standard_normal((n_obs, 1))
    xt, yt = td.Tensor(X), td.Tensor(Y)

    rows = []
    for h in widths:
        params = [
            td.Tensor(rng.standard_normal((d, h)) * 0.3, requires_grad=True),
            td.Tensor(np.zeros(h), requires_grad=True),
            td.Tensor(rng.standard_normal((h, 1)) * 0.3, requires_grad=True),
        ]
        n = sum(p.data.size for p in params)

        def loss(ps=params):
            hidden = td.tanh(xt @ ps[0] + ps[1])
            return td.mse_loss(hidden @ ps[2], yt)

        # Best of N: microbenchmarks are one-sided, so the minimum is the
        # honest estimate of the work and the mean is mostly scheduler noise.
        reps = 25
        t_fwd = min(_time(lambda: loss().data) for _ in range(reps))

        def one_backward(ps=params):
            for p in ps:
                p.grad = None
            loss().backward()

        t_ad = min(_time(one_backward) for _ in range(reps))

        # The finite-difference competitor, done properly: two forward
        # passes per scalar parameter, and the answer kept so we can check
        # the two methods actually agree before comparing their costs.
        eps = 1e-6
        t0 = time.perf_counter()
        fd_grads = []
        for p in params:
            flat = p.data.reshape(-1)
            g = np.empty_like(flat)
            for i in range(flat.size):
                orig = flat[i]
                flat[i] = orig + eps
                f_plus = loss().data.item()
                flat[i] = orig - eps
                f_minus = loss().data.item()
                flat[i] = orig
                g[i] = (f_plus - f_minus) / (2 * eps)
            fd_grads.append(g)
        t_fd = time.perf_counter() - t0

        for p, g in zip(params, fd_grads, strict=True):
            assert np.allclose(p.grad.reshape(-1), g, atol=1e-6)

        rows.append((n, t_fwd, t_ad, t_fd))
        print(f"  n={n:>6}  forward={t_fwd * 1e3:8.3f} ms   "
              f"backward={t_ad * 1e3:8.3f} ms   "
              f"central-diff={t_fd:8.3f} s   speedup={t_fd / t_ad:8.0f}x")
    return rows


def plot_cost(ax, rows) -> None:
    n = np.array([r[0] for r in rows], dtype=float)
    fwd = np.array([r[1] for r in rows]) * 1e3
    ad = np.array([r[2] for r in rows]) * 1e3
    fd = np.array([r[3] for r in rows]) * 1e3

    ax.plot(n, fd, "o-", color=RUST, lw=1.8, ms=4.5,
            label="central differences (2n forward passes)")
    ax.plot(n, ad, "o-", color=BLUE, lw=1.8, ms=4.5,
            label="tinydiff  (1 forward + 1 backward)")
    ax.plot(n, fwd, "--", color=MUTED, lw=1.2, label="1 forward pass")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("parameters differentiated,  n")
    ax.set_ylabel("wall time for the full gradient  (ms)")
    ax.set_title("1.  One pass, all n partials")
    ax.legend(loc="upper left")

    ratio = fd[-1] / ad[-1]
    ax.text(0.97, 0.06,
            f"{ratio:,.0f}× faster\nat n = {int(n[-1]):,}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9.5, color=BLUE, fontweight="bold", linespacing=1.35)
    tidy(ax)


# ---------------------------------------------------------------------
# Panel 2 — the correctness ledger
# ---------------------------------------------------------------------
def _max_rel_error(fn, *arrays, eps=1e-6):
    """Worst relative disagreement between autodiff and central differences."""
    inputs = [td.Tensor(np.array(a, dtype=np.float64), requires_grad=True)
              for a in arrays]
    fn(*inputs).backward()
    worst = 0.0
    for t in inputs:
        assert t.grad is not None, "input received no gradient"
        flat, gflat = t.data.reshape(-1), t.grad.reshape(-1)
        for i in range(flat.size):
            orig = flat[i]
            flat[i] = orig + eps
            f_plus = fn(*inputs).data.item()
            flat[i] = orig - eps
            f_minus = fn(*inputs).data.item()
            flat[i] = orig
            num = (f_plus - f_minus) / (2 * eps)
            worst = max(worst, abs(num - gflat[i]) / max(1.0, abs(num)))
    return worst


REGIMES = ["scalar", "vector\n(3,)", "matrix\n(4,3)", "batched\n(2,4,3)",
           "broadcast\n(4,3)+(3,)"]


def _shapes(regime_idx):
    return [(), (3,), (4, 3), (2, 4, 3), (4, 3)][regime_idx]


def _rhs_shape(regime_idx):
    return [(), (3,), (4, 3), (2, 4, 3), (3,)][regime_idx]


def correctness_ledger():
    """Max relative gradient error for each (op, shape regime) pair."""
    rng = np.random.default_rng(7)

    def pos(shape):  # strictly positive input, for log / div / pow
        return rng.uniform(0.5, 2.0, size=shape)

    def any_sign(shape):
        return rng.standard_normal(shape) + 0.35

    unary = {
        "pow(1.5)": lambda a: (a ** 1.5).sum(),
        "exp": lambda a: td.exp(a).sum(),
        "log": lambda a: td.log(a).sum(),
        "relu": lambda a: td.relu(a).sum(),
        "sigmoid": lambda a: td.sigmoid(a).sum(),
        "tanh": lambda a: td.tanh(a).sum(),
        "sum": lambda a: (a * 2.0).sum(),
        "mean": lambda a: td.mean(a * a),
    }
    binary = {
        "add": lambda a, b: (a + b).sum(),
        "sub": lambda a, b: (a - b).sum(),
        "mul": lambda a, b: (a * b).sum(),
        "div": lambda a, b: (a / b).sum(),
        "mse_loss": lambda a, b: td.mse_loss(a, b),
    }
    # Ordered the way the README lists them: binary, matmul, unary,
    # activations, reductions, losses.
    ops = ["add", "sub", "mul", "div", "matmul", "pow(1.5)", "exp", "log",
           "relu", "sigmoid", "tanh", "sum", "mean",
           "mse_loss", "softmax_xent"]

    # matmul walks its own shape space: the gufunc signature (n?,k),(k,m?).
    matmul_cases = {
        1: ((3,), (3,)),
        2: ((4, 3), (3, 2)),
        3: ((2, 4, 3), (2, 3, 2)),
        4: ((2, 4, 3), (3, 2)),
    }

    grid = np.full((len(ops), len(REGIMES)), np.nan)
    for j in range(len(REGIMES)):
        shape = _shapes(j)
        for i, name in enumerate(ops):
            if name in unary:
                if j == 4:
                    continue  # broadcasting is meaningless for a unary op
                arr = pos(shape) if name in ("log", "pow(1.5)") else any_sign(shape)
                grid[i, j] = _max_rel_error(unary[name], arr)
            elif name in binary:
                a = any_sign(shape)
                b = pos(_rhs_shape(j)) if name == "div" else any_sign(_rhs_shape(j))
                grid[i, j] = _max_rel_error(binary[name], a, b)
            elif name == "matmul":
                if j not in matmul_cases:
                    continue
                sa, sb = matmul_cases[j]
                w = rng.standard_normal(np.matmul(any_sign(sa), any_sign(sb)).shape)

                def f(x, y, w=w):
                    return (td.matmul(x, y) * td.Tensor(w)).sum()

                grid[i, j] = _max_rel_error(f, any_sign(sa), any_sign(sb))
            elif name == "softmax_xent" and j == 2:
                labels = np.array([0, 2, 1, 2])

                def f(logits, labels=labels):
                    return td.softmax_crossentropy(logits, labels)

                grid[i, j] = _max_rel_error(f, any_sign((4, 3)))

    covered = int(np.count_nonzero(~np.isnan(grid)))
    print(f"  {covered} (op, shape) pairs checked;  "
          f"worst relative error = {np.nanmax(grid):.1e}")
    return ops, grid


def plot_ledger(ax, ops, grid) -> None:
    logged = np.log10(np.where(np.isnan(grid), np.nan, np.maximum(grid, 1e-17)))
    cmap = LinearSegmentedColormap.from_list(
        "agree", ["#dfeaf6", "#eef2f6", "#f6e6d5", "#e8b07c"])
    cmap.set_bad("#f4f5f7")
    # Span exactly the decades the ledger occupies. A fixed -16..-8 scale
    # left half the colourbar dead and compressed the real spread into a
    # few shades, which is the opposite of what the panel is for.
    lo = float(np.floor(np.nanmin(logged)))
    hi = float(np.ceil(np.nanmax(logged)))
    im = ax.imshow(logged, cmap=cmap, vmin=lo, vmax=hi, aspect="auto")

    ax.set_xticks(range(len(REGIMES)), REGIMES, fontsize=7.6)
    ax.set_yticks(range(len(ops)), ops, fontsize=8, family="monospace")
    ax.set_title("2.  Every op, every shape regime")
    for i in range(len(ops)):
        for j in range(len(REGIMES)):
            v = grid[i, j]
            if np.isnan(v):
                ax.text(j, i, "·", ha="center", va="center",
                        color="#c3c8ce", fontsize=11)
            else:
                exp = int(np.floor(np.log10(max(v, 1e-17))))
                ax.text(j, i, f"1e{exp}", ha="center", va="center",
                        fontsize=7.2, family="monospace", color=INK)
    ax.set_xticks(np.arange(-0.5, len(REGIMES), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ops), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.6)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("log₁₀ max relative error  (autodiff vs central differences)",
                 fontsize=8)
    cb.ax.tick_params(labelsize=7.5)
    cb.outline.set_visible(False)


# ---------------------------------------------------------------------
# Panel 3 — graph depth
# ---------------------------------------------------------------------
def _chain(k):
    """y = x * m applied k times, with m chosen so that dy/dx == 2 exactly."""
    m = 2.0 ** (1.0 / k)
    x = td.Tensor(1.0, requires_grad=True)
    y = x
    for _ in range(k):
        y = y * m
    return x, y


def _recursive_topo(root):
    """The textbook recursive DFS, kept here only to measure where it dies."""
    topo, seen = [], set()

    def visit(t):
        if id(t) in seen:
            return
        seen.add(id(t))
        for c in t._children:
            visit(c)
        topo.append(t)

    visit(root)
    return topo


def depth_sweep(depths=DEFAULT_DEPTHS):
    """Time the backward pass as the graph gets deeper.

    Best of three passes, with the collector paused: at 10^5 live Tensors a
    generational GC sweep costs more than the backward pass it interrupts.
    Passes two and three are exactly what `retain_graph=True` is for.
    """
    rows = []
    for k in depths:
        x, y = _chain(k)
        gc.collect()
        gc.disable()
        try:
            times = [_time(y.backward)]
            times += [_time(lambda root=y: root.backward(retain_graph=True))
                      for _ in range(2)]
        finally:
            gc.enable()
        t_b = min(times)
        # Three passes accumulate into the leaf, so the first pass's value
        # is x.grad / 3 — the chain rule, not an artefact.
        rel = abs(float(x.grad) / 3.0 - 2.0) / 2.0
        rows.append((k, t_b, rel))
        print(f"  depth={k:>7}  backward={t_b * 1e3:8.2f} ms   "
              f"dy/dx={float(x.grad) / 3.0:.12f}  (exact: 2)   "
              f"rel err={rel:.1e}")

    lo, hi = 1, 100000
    while lo < hi:  # measure, don't assume, where recursion gives out
        mid = (lo + hi + 1) // 2
        try:
            _recursive_topo(_chain(mid)[1])
            lo = mid
        except RecursionError:
            hi = mid - 1
    print(f"  deepest chain a recursive graph walk survives: {lo} ops")
    return rows, lo


def plot_depth(ax, rows, cliff) -> None:
    k = np.array([r[0] for r in rows], dtype=float)
    t = np.array([r[1] for r in rows]) * 1e3

    ax.axvspan(cliff, 2e5, color="#fbeeea", zorder=0)
    ax.plot(k, t, "o-", color=GREEN, lw=1.8, ms=4.5, label="backward pass")
    ax.axvline(cliff, color=RUST, lw=1.2, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(7, 2e5)
    ax.set_xlabel("chain length  (ops between leaf and loss)")
    ax.set_ylabel("backward pass  (ms)")
    ax.set_title("3.  Depth is a budget, not a wall")
    ax.annotate(f"a recursive graph walk\nstops at {cliff} ops",
                xy=(cliff, t[-1] * 0.09), xytext=(21, t[-1] * 0.55),
                fontsize=8.5, color=RUST, ha="left", va="center",
                arrowprops={"arrowstyle": "->", "color": RUST, "lw": 1.0,
                            "shrinkA": 2, "shrinkB": 3})
    ax.text(0.97, 0.06,
            f"{int(k[-1]):,} ops in one pass\n"
            f"dy/dx correct to {rows[-1][2]:.0e}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9.5, color=GREEN, fontweight="bold", linespacing=1.35)
    ax.legend(loc="upper left")
    tidy(ax)


# ---------------------------------------------------------------------
# Figure 2 — the engine driving a fit
# ---------------------------------------------------------------------
def train_spiral(epochs=1500):
    rng = np.random.default_rng(0)
    X, y = spiral(seed=0)
    X_test, y_test = spiral(seed=1)
    model = td.Sequential(
        td.Linear(2, 32, rng=rng), td.relu,
        td.Linear(32, 32, rng=rng), td.relu,
        td.Linear(32, 2, rng=rng),
    )
    opt = td.Adam(model.parameters(), lr=0.01)
    linears = [layer for layer in model.layers if isinstance(layer, td.Linear)]

    hist = {"loss": [], "train": [], "test": [], "grad": [[] for _ in linears]}
    xt, xv = td.Tensor(X), td.Tensor(X_test)
    for _ in range(epochs):
        opt.zero_grad()
        logits = model(xt)
        loss = td.softmax_crossentropy(logits, y)
        loss.backward()
        hist["loss"].append(float(loss.data))
        hist["train"].append(float((logits.data.argmax(axis=1) == y).mean()))
        hist["test"].append(
            float((model(xv).data.argmax(axis=1) == y_test).mean()))
        for i, lin in enumerate(linears):
            hist["grad"][i].append(float(np.sqrt(np.mean(lin.W.grad ** 2))))
        opt.step()

    print(f"  {epochs} epochs   final loss={hist['loss'][-1]:.4f}   "
          f"train={hist['train'][-1]:.1%}   held out={hist['test'][-1]:.1%}")
    return X_test, y_test, model, hist


def _ema(series, alpha=0.02):
    out, acc = [], series[0]
    for v in series:
        acc = (1 - alpha) * acc + alpha * v
        out.append(acc)
    return np.array(out)


def plot_spiral(fig, axes, X, y, model, hist) -> None:
    ax = axes[0]
    ep = np.arange(len(hist["loss"]))
    ax.plot(ep, hist["loss"], color=BLUE, lw=1.5, label="cross-entropy")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss")
    ax.set_title("Fitting the spirals")
    tidy(ax)
    ax2 = ax.twinx()
    ax2.plot(ep, hist["train"], color=MUTED, lw=1.2, label="train accuracy")
    ax2.plot(ep, hist["test"], color=RUST, lw=1.4, label="held-out accuracy")
    ax2.set_ylim(0.4, 1.03)
    ax2.set_ylabel("accuracy")
    ax2.spines["top"].set_visible(False)
    ax2.grid(False)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [ln.get_label() for ln in lines], loc="center",
              bbox_to_anchor=(0.68, 0.42))

    ax = axes[1]
    pad = 0.25
    gx = np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, 300)
    gy = np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, 300)
    gxx, gyy = np.meshgrid(gx, gy)
    grid = np.stack([gxx.ravel(), gyy.ravel()], axis=1)
    logits = model(td.Tensor(grid)).data
    logits -= logits.max(axis=1, keepdims=True)
    p1 = (np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True))[:, 1]
    cmap = LinearSegmentedColormap.from_list(
        "spiral", ["#dbe6f4", "#ffffff", "#f6e0d4"])
    ax.contourf(gxx, gyy, p1.reshape(gxx.shape), levels=21, cmap=cmap)
    ax.contour(gxx, gyy, p1.reshape(gxx.shape), levels=[0.5],
               colors=[INK], linewidths=1.1)
    for cls, color in ((0, BLUE), (1, RUST)):
        sel = y == cls
        ax.scatter(X[sel, 0], X[sel, 1], s=7, color=color,
                   edgecolors="white", linewidths=0.25, label=f"class {cls}")
    ax.set_title(f"Decision boundary, held-out points  "
                 f"—  {hist['test'][-1]:.1%}")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.legend(loc="upper right", markerscale=1.6)
    ax.set_aspect("equal")
    tidy(ax)
    ax.grid(False)

    ax = axes[2]
    labels = ["Linear(2, 32)", "Linear(32, 32)", "Linear(32, 2)"]
    for series, label, color in zip(hist["grad"], labels, (BLUE, GREEN, RUST),
                                    strict=True):
        ax.plot(ep, series, lw=0.6, color=color, alpha=0.22)
        ax.plot(ep, _ema(series), lw=1.7, color=color, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"RMS of $\partial L/\partial W$")
    ax.set_title("Gradient reaching each layer")
    ax.legend(loc="upper right", title="faint = per epoch,  bold = EMA",
              title_fontsize=7.5)
    tidy(ax)
    fig.tight_layout()


# ---------------------------------------------------------------------
# Figure 3 — what the max-shift buys, and what the reference costs
# ---------------------------------------------------------------------
BASE = np.array([0.0, -1.0, -2.0])


def _naive_lse(z):
    return np.log(np.exp(z).sum())


def _shifted_lse(z):
    m = z.max()
    return m + np.log(np.exp(z - m).sum())


def stability_sweep():
    """Naive vs max-shifted log-sum-exp as the whole vector slides up.

    The exact answer is c + LSE([0,-1,-2]) for every c, so the error is
    known without a reference implementation.
    """
    exact0 = _naive_lse(BASE)
    cs = np.linspace(0.0, 1000.0, 401)
    naive, shifted = [], []
    with np.errstate(over="ignore"):
        for c in cs:
            z = BASE + c
            exact = c + exact0
            naive.append(abs(_naive_lse(z) - exact) / abs(exact if exact else 1.0))
            shifted.append(abs(_shifted_lse(z) - exact) / abs(exact if exact else 1.0))
    naive = np.array(naive)
    died = float(cs[np.argmax(~np.isfinite(naive))]) if not np.isfinite(naive).all() else np.inf
    print(f"  log-sum-exp: naive stops returning a number at c = {died:.1f}")
    return cs, naive, np.array(shifted), died


def confident_mistake_sweep():
    """Cross-entropy on logits [0, -d] with the true class second.

    The exact loss is d + log(1 + e^-d); the naive route computes a
    probability first and then takes its log, which loses the answer once
    the probability goes subnormal.
    """
    ds = np.linspace(600.0, 780.0, 361)
    naive, fused = [], []
    for d in ds:
        Z = np.array([[0.0, -d]])
        exact = d + float(np.log1p(np.exp(-d)))
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            P = np.exp(Z) / np.exp(Z).sum(axis=1, keepdims=True)
            nv = float(-np.log(P[0, 1]))
        ours = float(td.softmax_crossentropy(td.Tensor(Z), np.array([1])).data)
        naive.append(abs(nv - exact))
        fused.append(abs(ours - exact))
    naive = np.array(naive)
    finite = np.isfinite(naive)
    first_wrong = float(ds[np.argmax(naive > 1e-9)]) if (naive > 1e-9).any() else np.inf
    first_inf = float(ds[np.argmax(~finite)]) if not finite.all() else np.inf
    print(f"  cross-entropy: naive is silently wrong from d = {first_wrong:.1f}, "
          f"returns inf from d = {first_inf:.1f}")
    return ds, naive, np.array(fused), first_wrong, first_inf


def step_size_sweep():
    """Central-difference error against step size, for f(x) = x tanh x.

    The closed form f'(x) = tanh x + x (1 - tanh^2 x) is written out by
    hand, so neither side is computing the other's expression. Autodiff's
    error is one rounding; the estimator's is a U in h whose floor is the
    best any finite-difference check can ever do. The floor is reported as
    a median across the trough because round-off makes the bottom of the U
    a sawtooth, not a curve.
    """
    x0 = 0.7
    exact = float(np.tanh(x0) + x0 * (1.0 - np.tanh(x0) ** 2))

    def f(x):
        return x * np.tanh(x)

    hs = np.logspace(-14, -1, 261)
    err = np.array([abs((f(x0 + h) - f(x0 - h)) / (2 * h) - exact) / abs(exact)
                    for h in hs])
    t = td.Tensor(x0, requires_grad=True)
    (t * td.tanh(t)).backward()
    ad = abs(float(t.grad) - exact) / abs(exact)
    trough = (hs >= 3e-7) & (hs <= 3e-5)
    floor = float(np.median(err[trough]))
    print(f"  finite differences floor at {floor:.1e} (median over h in "
          f"[3e-7, 3e-5]);  autodiff vs the closed form: {ad:.1e}")
    return hs, err, max(ad, 1e-17), floor


def plot_stability(fig, axes, lse, conf, step) -> None:
    FLOOR = 1e-17
    cs, naive, shifted, died = lse
    ax = axes[0]
    ok = np.isfinite(naive)
    ax.plot(cs[ok], np.maximum(naive[ok], FLOOR), color=RUST, lw=3.6,
            alpha=0.85, solid_capstyle="butt", label="naive  log(sum(exp z))")
    ax.plot(cs, np.maximum(shifted, FLOOR), color=BLUE, lw=1.5, ls=(0, (4, 3)),
            label="max-shifted (tinydiff)")
    ax.axvspan(died, cs[-1], color="#fbeeea", zorder=0)
    ax.axvline(died, color=RUST, lw=1.1, ls="--")
    ax.text(died + 20, 4e-11, f"naive returns inf\nfrom c = {died:.0f}",
            fontsize=8.5, color=RUST, va="center", linespacing=1.35)
    ax.text(20, 4e-11, f"both exact to the last bit\nfor every c < {died:.0f}",
            fontsize=8.5, color=INK, va="center", linespacing=1.35)
    ax.set_yscale("log")
    ax.set_ylim(FLOOR * 0.4, 1e-8)
    ax.set_xlabel("logits shifted by c")
    ax.set_ylabel("relative error vs the exact answer")
    ax.set_title("A.  log-sum-exp, slid upwards")
    ax.legend(loc="upper left")
    tidy(ax)

    ds, nv, fused, first_wrong, first_inf = conf
    ax = axes[1]
    ax.plot(ds, np.maximum(fused, FLOOR), color=BLUE, lw=1.9, label="fused (tinydiff)")
    ok = np.isfinite(nv)
    ax.plot(ds[ok], np.maximum(nv[ok], FLOOR), color=RUST, lw=1.9, label="naive  -log softmax")
    ax.axvspan(first_wrong, first_inf, color="#fdf5e8", zorder=0)
    ax.axvspan(first_inf, ds[-1], color="#fbeeea", zorder=0)
    ax.set_yscale("log")
    ax.set_ylim(FLOOR * 0.4, 10)
    ax.set_xlabel("logit gap d,  logits [0, -d],  true class 1")
    ax.set_ylabel("absolute error in the loss (nats)")
    ax.set_title("B.  A confident mistake")
    ax.text((first_wrong + first_inf) / 2, 1e-13, "silently wrong", rotation=90,
            fontsize=8.5, color="#9a6b12", ha="center", va="center")
    ax.text((first_inf + ds[-1]) / 2, 1e-13, "returns inf", rotation=90,
            fontsize=8.5, color=RUST, ha="center", va="center")
    ax.legend(loc="upper left")
    tidy(ax)

    hs, err, ad, floor = step
    ax = axes[2]
    ax.plot(hs, np.maximum(err, FLOOR), color=RUST, lw=1.4,
            label="central difference, step h")
    ax.axhline(ad, color=BLUE, lw=1.9,
               label="tinydiff vs the closed form"
                     + ("  (0: identical bits)" if ad <= 1e-17 else f"  ({ad:.0e})"))
    ax.axhspan(floor / 3, floor * 3, color="#f3ece2", zorder=0)
    ax.axhline(floor, color=INK, lw=1.0, ls=":")
    ax.annotate(f"floor {floor:.0e}\n"
                r"$\approx \epsilon_{\mathrm{mach}}^{2/3}$",
                xy=(1.6e-13, floor * 9.0), fontsize=8.5, color=INK,
                linespacing=1.35)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(FLOOR * 0.4, 1e-1)
    ax.set_xlabel("finite-difference step,  h")
    ax.set_ylabel(r"relative error in $d(x\tanh x)/dx$ at $x=0.7$")
    ax.set_title("C.  Which one is the noisy one")
    ax.legend(loc="lower left")
    tidy(ax)
    fig.tight_layout()


# ---------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=pathlib.Path, default=DOCS,
                    help="directory to write the PNGs into")
    ap.add_argument("--quick", action="store_true",
                    help="tiny sweeps — for the CI smoke test, not for docs")
    args = ap.parse_args()

    style()
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/3] cost of a full gradient")
    cost = cost_sweep(widths=(4, 16) if args.quick else DEFAULT_WIDTHS)
    print("[2/3] correctness ledger")
    ops, grid = correctness_ledger()
    print("[3/3] graph depth")
    depth, cliff = depth_sweep(
        depths=(10, 100, 1000) if args.quick else DEFAULT_DEPTHS)

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.9))
    plot_cost(axes[0], cost)
    plot_ledger(axes[1], ops, grid)
    plot_depth(axes[2], depth, cliff)
    fig.suptitle("tinydiff — what a reverse-mode engine has to get right",
                 fontsize=13.5, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    out = out_dir / "engine_report.png"
    fig.savefig(out, dpi=100, facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")

    print("[stability] naive vs max-shifted")
    lse = stability_sweep()
    conf = confident_mistake_sweep()
    step = step_size_sweep()
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.4))
    plot_stability(fig, axes, lse, conf, step)
    fig.suptitle("tinydiff — the numbers the naive formulation cannot return",
                 fontsize=13.5, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    out = out_dir / "validation.png"
    fig.savefig(out, dpi=100, facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")

    print("[spiral] training")
    X, y, model, hist = train_spiral(epochs=25 if args.quick else 1500)
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.4))
    plot_spiral(fig, axes, X, y, model, hist)
    out = out_dir / "spiral_training.png"
    fig.savefig(out, dpi=100, facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
