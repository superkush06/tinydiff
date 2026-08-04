"""Two interleaved spirals — the smallest problem a linear model cannot touch.

Two Archimedean spirals share a centre and run half a turn out of phase, so
every straight line through the plane cuts both classes in roughly equal
proportion. A 2-32-32-2 ReLU network gets it essentially exactly right, and
nothing in the library is special-cased to help: the same `backward()` that
differentiates a scalar chain differentiates this.

Run:  PYTHONPATH=. python3 examples/spiral.py
"""

import argparse

import numpy as np

import tinydiff as td


def spiral(n: int = 200, turns: float = 2.5, noise: float = 0.10,
           seed: int = 0):
    """`n` points per class on two interleaved spirals.

    Jitter is applied along the arm (angular) and across it (radial), not as
    isotropic Cartesian noise, so the arms stay arms: at radius `r` the
    nearest opposite-class arm sits about `r / (2 * turns)` away, and the
    noise scale stays well inside that gap.
    """
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for cls in (0, 1):
        theta = np.linspace(0.0, 2.0 * np.pi * turns, n) + cls * np.pi
        radius = np.linspace(0.08, 1.0, n)
        theta = theta + rng.normal(scale=noise, size=n)
        radius = radius * (1.0 + rng.normal(scale=noise * 0.35, size=n))
        xs.append(np.stack([radius * np.cos(theta),
                            radius * np.sin(theta)], axis=1))
        ys.append(np.full(n, cls, dtype=np.int64))
    X = np.concatenate(xs)
    y = np.concatenate(ys)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def accuracy(model, X, y) -> float:
    return float((model(td.Tensor(X)).data.argmax(axis=1) == y).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=0.01)
    args = ap.parse_args()

    # Same two spirals, independent noise draw: a held-out set, not a reshuffle.
    X_train, y_train = spiral(seed=0)
    X_test, y_test = spiral(seed=1)

    rng = np.random.default_rng(0)
    model = td.Sequential(
        td.Linear(2, 32, rng=rng),
        td.relu,
        td.Linear(32, 32, rng=rng),
        td.relu,
        td.Linear(32, 2, rng=rng),
    )
    opt = td.Adam(model.parameters(), lr=args.lr)
    xt = td.Tensor(X_train)

    for epoch in range(args.epochs):
        opt.zero_grad()
        logits = model(xt)
        loss = td.softmax_crossentropy(logits, y_train)
        loss.backward()
        if epoch % 250 == 0 or epoch == args.epochs - 1:
            train_acc = float((logits.data.argmax(axis=1) == y_train).mean())
            print(f"epoch {epoch:>4}: loss={loss.data:.4f}  "
                  f"train={train_acc:.1%}  "
                  f"test={accuracy(model, X_test, y_test):.1%}")
        opt.step()


if __name__ == "__main__":
    main()
