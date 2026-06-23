# tinydiff

[![ci](https://github.com/superkush06/tinydiff/actions/workflows/ci.yml/badge.svg)](https://github.com/superkush06/tinydiff/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> A reverse-mode autodiff engine in <500 lines of library code. Trains an MLP
> on toy classification + regression tasks. NumPy is the array backend; the
> autograd is built from scratch.

## TL;DR

```python
import numpy as np
import tinydiff as td

model = td.Sequential(
    td.Linear(2, 32), td.relu,
    td.Linear(32, 2),
)
opt = td.Adam(model.parameters(), lr=0.01)

X, y = td.Tensor(X_np), y_np  # X: (N, 2), y: (N,) int classes
for _ in range(epochs):
    opt.zero_grad()
    loss = td.softmax_crossentropy(model(X), y)
    loss.backward()
    opt.step()
```

## What's inside

- `Tensor` — wraps numpy arrays, tracks autograd graph (`tensor.py`, ~80 LOC).
- Operations: `add`, `sub`, `mul`, `div`, `matmul`, `pow`, `exp`, `log`,
  `relu`, `sigmoid`, `tanh`, `sum`, `mean` (`ops.py`, ~180 LOC).
- `nn.Module`, `Linear`, `Sequential` (`nn.py`, ~60 LOC).
- Losses: `mse_loss`, `softmax_crossentropy` (`loss.py`, ~40 LOC).
- Optimizers: `SGD`, `Adam` (`optim.py`, ~50 LOC).
- `grad_check` — numerical vs analytical gradient comparison.

## Examples

```
PYTHONPATH=. python3 examples/fit_sine.py     # 2-layer MLP fits sin(x)
PYTHONPATH=. python3 examples/xor.py           # XOR classification
PYTHONPATH=. python3 examples/spiral.py        # two-spiral classifier
```

Sample output from `xor.py`:
```
epoch    0: loss=0.7203  acc=50%
epoch  100: loss=0.0212  acc=100%
epoch 1000: loss=0.0003  acc=100%
```

## Theory

See [`docs/theory.md`](docs/theory.md) for a 5-minute primer on reverse-mode
autodiff: local Jacobians, broadcasting, topo traversal, and numerical
stability.

## Install

```bash
git clone https://github.com/superkush06/tinydiff.git
cd tinydiff
pip install -e ".[dev]"
pytest
```

## Roadmap

- [ ] Convolutional layers.
- [ ] BatchNorm and Dropout.
- [ ] Higher-order grads (grad of grad).
- [ ] GPU backend (cupy).

## License

MIT — see [LICENSE](LICENSE).
