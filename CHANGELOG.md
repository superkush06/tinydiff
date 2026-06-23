# Changelog

## [0.1.0] - 2026-08-XX

### Added
- `Tensor` with reverse-mode autograd, including reverse-topo traversal
  and gradient accumulation.
- Ops: binary (`add`, `sub`, `mul`, `div`, `matmul`, `pow`), unary
  (`neg`, `exp`, `log`), activations (`relu`, `sigmoid`, `tanh`), and
  reductions (`sum_`, `mean`).
- Broadcasting-aware gradient unbroadcast.
- `nn.Module`, `Linear` (Kaiming-He init), `Sequential`.
- Losses: `mse_loss`, numerically-stable `softmax_crossentropy`.
- Optimizers: `SGD` (with momentum), `Adam`.
- `grad_check` — central-difference numerical gradient verifier.
- Examples: `fit_sine.py`, `xor.py`, `spiral.py`.
- CI on Python 3.11 + 3.12.
