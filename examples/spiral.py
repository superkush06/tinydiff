"""Two-spiral classification with a 3-layer MLP.

Run:  PYTHONPATH=. python3 examples/spiral.py
"""

import argparse

import numpy as np

import tinydiff as td


def spiral(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    out_X, out_y = [], []
    for cls in (0, 1):
        theta = np.linspace(0.0, 4.0, n)
        r = np.linspace(0.05, 1.0, n)
        rot = cls * np.pi
        xs = r * np.cos(4.0 * theta + rot) + rng.normal(scale=0.05, size=n)
        ys = r * np.sin(4.0 * theta + rot) + rng.normal(scale=0.05, size=n)
        out_X.append(np.stack([xs, ys], axis=1))
        out_y.append(np.full(n, cls, dtype=np.int64))
    X = np.concatenate(out_X)
    y = np.concatenate(out_y)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=600)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    X, y = spiral(n=200, seed=0)

    model = td.Sequential(
        td.Linear(2, 32, rng=rng),
        td.relu,
        td.Linear(32, 32, rng=rng),
        td.relu,
        td.Linear(32, 2, rng=rng),
    )
    opt = td.Adam(model.parameters(), lr=0.01)

    for epoch in range(args.epochs):
        opt.zero_grad()
        logits = model(td.Tensor(X))
        loss = td.softmax_crossentropy(logits, y)
        loss.backward()
        opt.step()
        if epoch % 100 == 0 or epoch == args.epochs - 1:
            preds = logits.data.argmax(axis=1)
            acc = (preds == y).mean()
            print(f"epoch {epoch:>4}: loss={loss.data:.4f}  acc={acc:.1%}")


if __name__ == "__main__":
    main()
