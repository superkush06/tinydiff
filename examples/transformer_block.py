"""The oracle a hand-written backward pass gets checked against.

A transformer written from scratch does not use an autodiff engine — it
derives every backward pass on paper and writes it out as NumPy. That is
the point of the exercise, and it is also where the bugs live: the
LayerNorm gradient has three terms that are easy to reduce to one, the
GELU derivative is a product rule inside a chain rule, and a bias
gradient has to be summed over the token axis that broadcasting silently
created on the way forward.

This script builds one transformer feed-forward sublayer twice — once as
hand-derived NumPy, once as tinydiff expressions — on the same parameters
and the same inputs, and compares every gradient. Then it does the same
with two bugs of the kind that actually get shipped, to show what the
comparison is worth: both buggy gradients still point downhill, so a loss
curve does not notice them, and the parameter-by-parameter comparison
does, immediately.

Nothing is imported from anywhere else. The token vectors that would come
out of an embedding table are drawn from a seeded generator instead.

    PYTHONPATH=. python3 examples/transformer_block.py
"""

from __future__ import annotations

import numpy as np

import tinydiff as td

EPS = 1e-5
GELU_C = np.sqrt(2.0 / np.pi)
GELU_A = 0.044715


# ======================================================================
# The block, written out by hand: forward and backward, plain NumPy
# ======================================================================
def gelu(x):
    """GELU, tanh approximation (Hendrycks & Gimpel 2016, eq. 2)."""
    return 0.5 * x * (1.0 + np.tanh(GELU_C * (x + GELU_A * x ** 3)))


def gelu_grad(x):
    """d/dx of the above — product rule around a chain rule.

    With u = c(x + a x^3) and t = tanh u:
        g   = 0.5 x (1 + t)
        g'  = 0.5 (1 + t) + 0.5 x (1 - t^2) c (1 + 3 a x^2)
    """
    u = GELU_C * (x + GELU_A * x ** 3)
    t = np.tanh(u)
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t ** 2) * GELU_C * (1.0 + 3.0 * GELU_A * x ** 2)


def forward_numpy(params, X, targets):
    """LayerNorm -> W1 -> GELU -> W2 -> residual -> unembed -> cross-entropy."""
    gamma, beta, W1, b1, W2, b2, Wu, bu = params
    mu = X.mean(axis=-1, keepdims=True)
    xc = X - mu
    var = (xc ** 2).mean(axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + EPS)
    xhat = xc * inv
    ln = xhat * gamma + beta

    a = ln @ W1 + b1
    g = gelu(a)
    ff = g @ W2 + b2
    h = X + ff                      # residual stream

    logits = h @ Wu + bu
    z = logits - logits.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    loss = float(-logp[np.arange(len(targets)), targets].mean())
    cache = (xhat, inv, ln, a, g, h, np.exp(logp))
    return loss, cache


def backward_numpy(params, X, targets, cache, *, bug=None):
    """The hand-derived gradients. `bug` injects a plausible mistake."""
    gamma, beta, W1, b1, W2, b2, Wu, bu = params
    xhat, inv, ln, a, g, h, probs = cache
    T, d = X.shape

    # cross-entropy + unembedding
    dlogits = probs.copy()
    dlogits[np.arange(T), targets] -= 1.0
    dlogits /= T
    dWu = h.T @ dlogits
    dbu = dlogits.sum(axis=0)
    dh = dlogits @ Wu.T

    # residual: the stream carries gradient to both branches
    dff = dh
    dX = dh.copy()

    # second projection
    dW2 = g.T @ dff
    db2 = dff.sum(axis=0)
    dg = dff @ W2.T

    # GELU
    da = dg * gelu_grad(a)

    # first projection
    dW1 = ln.T @ da
    db1 = da.sum(axis=0) if bug != "bias" else da[0].copy()
    dln = da @ W1.T

    # LayerNorm
    dgamma = (dln * xhat).sum(axis=0)
    dbeta = dln.sum(axis=0)
    dxhat = dln * gamma
    if bug == "layernorm":
        # The mistake: treating 1/sigma as a constant. It is not — mu and
        # var both depend on every entry of x, which is where the two
        # subtracted means come from.
        dX += dxhat * inv
    else:
        dX += inv * (dxhat - dxhat.mean(axis=-1, keepdims=True)
                     - xhat * (dxhat * xhat).mean(axis=-1, keepdims=True))
    return dict(zip(NAMES, [dgamma, dbeta, dW1, db1, dW2, db2, dWu, dbu],
                    strict=True)) | {"X": dX}


# ======================================================================
# The same block, as tinydiff expressions
# ======================================================================
def forward_tinydiff(params, X, targets):
    gamma, beta, W1, b1, W2, b2, Wu, bu = params
    x = td.Tensor(X, requires_grad=True)
    mu = td.mean(x, axis=-1, keepdims=True)
    xc = x - mu
    var = td.mean(xc * xc, axis=-1, keepdims=True)
    xhat = xc / (var + EPS) ** 0.5
    ln = xhat * gamma + beta

    a = td.matmul(ln, W1) + b1
    inner = GELU_C * (a + GELU_A * a ** 3.0)
    g = 0.5 * a * (1.0 + td.tanh(inner))
    h = x + (td.matmul(g, W2) + b2)

    logits = td.matmul(h, Wu) + bu
    return td.softmax_crossentropy(logits, targets), x


NAMES = ["gamma", "beta", "W1", "b1", "W2", "b2", "Wu", "bu"]


def make_params(rng, d=16, d_ff=32, vocab=11):
    raw = [
        np.ones(d),                                     # gamma
        np.zeros(d),                                    # beta
        rng.standard_normal((d, d_ff)) * (2.0 / d) ** 0.5,
        rng.standard_normal(d_ff) * 0.05,
        rng.standard_normal((d_ff, d)) * (2.0 / d_ff) ** 0.5,
        rng.standard_normal(d) * 0.05,
        rng.standard_normal((d, vocab)) * (2.0 / d) ** 0.5,
        np.zeros(vocab),
    ]
    return raw


def compare(hand, auto):
    """Max relative gap and cosine similarity, per parameter."""
    rows = []
    for name in [*NAMES, "X"]:
        a, b = np.asarray(hand[name], dtype=float), np.asarray(auto[name], dtype=float)
        if a.shape != b.shape:
            rows.append((name, f"shape {a.shape} vs {b.shape}", float("nan")))
            continue
        denom = np.maximum(1e-12, np.abs(b))
        err = float(np.max(np.abs(a - b) / np.maximum(denom, np.abs(b).max())))
        cos = float(np.dot(a.ravel(), b.ravel())
                    / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))
        rows.append((name, err, cos))
    return rows


def setup():
    rng = np.random.default_rng(7)
    T, d, vocab = 12, 16, 11
    X = rng.standard_normal((T, d))          # stands in for embedded tokens
    targets = rng.integers(0, vocab, size=T)
    return X, targets, make_params(rng, d=d, vocab=vocab)


def descends(raw, X, targets, hand, step=0.05):
    """Take one step along -hand_gradient; report the loss before and after."""
    before, _ = forward_numpy(raw, X, targets)
    moved = [p - step * np.broadcast_to(hand[n], p.shape)
             for n, p in zip(NAMES, raw, strict=True)]
    after, _ = forward_numpy(moved, X, targets)
    return before, after


def run(bug=None):
    X, targets, raw = setup()

    loss_np, cache = forward_numpy(raw, X, targets)
    hand = backward_numpy(raw, X, targets, cache, bug=bug)

    tensors = [td.Tensor(p, requires_grad=True) for p in raw]
    loss_td, xt = forward_tinydiff(tensors, X, targets)
    loss_td.backward()
    auto = {n: t.grad for n, t in zip(NAMES, tensors, strict=True)}
    auto["X"] = xt.grad
    return loss_np, float(loss_td.data), compare(hand, auto), (raw, X, targets, hand)


def main() -> None:
    print(__doc__.strip().splitlines()[0])
    print()

    loss_np, loss_td, rows, ctx = run()
    print(f"forward loss   hand-written {loss_np:.12f}   tinydiff {loss_td:.12f}   "
          f"gap {abs(loss_np - loss_td):.1e}")
    print()
    print("correct hand-derived backward pass")
    print(f"  {'parameter':<8} {'max rel gap':>12}   {'cosine':>10}")
    worst = 0.0
    for name, err, cos in rows:
        worst = max(worst, err)
        print(f"  {name:<8} {err:>12.2e}   {cos:>10.8f}")
    print(f"  -> worst disagreement {worst:.1e}; the two derivations are the "
          "same function.")
    before, after = descends(*ctx)
    print(f"  -> one step of size 0.05 downhill: loss {before:.6f} -> {after:.6f}")

    for bug, blurb, note in (
        ("bias", "b1 gradient not summed over the token axis",
         "wrong by a factor of T on one parameter, and still downhill"),
        ("layernorm", "LayerNorm backward missing its two mean terms",
         "no parameter of THIS block is affected — only dL/dX, which is what "
         "the block hands to the layer below it"),
    ):
        _, _, rows, ctx = run(bug=bug)
        before, after = descends(*ctx)
        print(f"\ninjected bug: {blurb}")
        print(f"  one step of size 0.05 downhill: loss {before:.6f} -> {after:.6f}"
              f"   ({'still descends' if after < before else 'ascends'})")
        print(f"  {note}")
        print(f"  {'parameter':<8} {'max rel gap':>12}   {'cosine':>10}")
        for name, err, cos in rows:
            flag = "  <-- wrong" if (isinstance(err, str) or err > 1e-8) else ""
            shown = err if isinstance(err, str) else f"{err:12.2e}"
            print(f"  {name:<8} {shown:>12}   {cos:>10.8f}{flag}")

    print("\nNeither bug announces itself. The first keeps a positive cosine with")
    print("the true gradient and still reduces the loss, so the training curve")
    print("looks fine. The second does not touch this block's own parameters at")
    print("all — it corrupts dL/dX, the gradient handed to the layer below, where")
    print("it becomes somebody else's slow convergence. Comparing against an")
    print("engine finds both on the first backward pass, before any training.")


if __name__ == "__main__":
    main()
