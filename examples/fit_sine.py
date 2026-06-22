"""Fit y = sin(x) with a 2-hidden-layer MLP.

Run:  PYTHONPATH=. python3 examples/fit_sine.py
"""

import argparse

import numpy as np

import tinydiff as td


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.01)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    X = np.linspace(-3.0, 3.0, 256).reshape(-1, 1)
    y_true = np.sin(X)

    model = td.Sequential(
        td.Linear(1, 32, rng=rng),
        td.relu,
        td.Linear(32, 32, rng=rng),
        td.relu,
        td.Linear(32, 1, rng=rng),
    )
    opt = td.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        opt.zero_grad()
        pred = model(td.Tensor(X))
        loss = td.mse_loss(pred, td.Tensor(y_true))
        loss.backward()
        opt.step()
        if epoch % 20 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:>4}: loss={loss.data:.6f}")

    print(f"\nfinal loss: {loss.data:.6f}")


if __name__ == "__main__":
    main()
