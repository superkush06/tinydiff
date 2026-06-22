"""Train a 2-hidden MLP to learn XOR.

The classic non-linearly-separable problem — single layer can't, two layers can.

Run:  PYTHONPATH=. python3 examples/xor.py
"""

import numpy as np

import tinydiff as td


def main() -> None:
    rng = np.random.default_rng(0)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    y = np.array([0, 1, 1, 0], dtype=np.int64)

    model = td.Sequential(
        td.Linear(2, 4, rng=rng),
        td.tanh,
        td.Linear(4, 2, rng=rng),
    )
    opt = td.Adam(model.parameters(), lr=0.1)

    for epoch in range(1000):
        opt.zero_grad()
        logits = model(td.Tensor(X))
        loss = td.softmax_crossentropy(logits, y)
        loss.backward()
        opt.step()
        if epoch % 100 == 0 or epoch == 999:
            preds = logits.data.argmax(axis=1)
            acc = (preds == y).mean()
            print(f"epoch {epoch:>4}: loss={loss.data:.4f}  acc={acc:.0%}")


if __name__ == "__main__":
    main()
