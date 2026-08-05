# Changelog

## [Unreleased]

### Changed
- The README's spiral paragraph called the gradient fall after epoch 800 a
  "three-decade drop" and then quoted two losses 1.5 decades apart, in one
  sentence. Neither half survived re-measurement. Three decades is the *height
  of the panel*: the per-layer EMA curves fall 1.50, 1.72 and 1.82 decades
  from epoch 750 to 1499, the loss falls 1.53 decades over the same span, and
  the largest fall any single curve makes over the whole run is 2.83 decades.
  The quoted `0.064` was also anchored to epoch 750 inside a sentence about
  epoch 800 (`loss[800]` is `0.0500`); the paragraph now anchors everything to
  epoch 750, which `spiral.py` actually prints, and the printed block is no
  longer trimmed.
- Backward-pass wall-clock claims. "About 300 ms of pure Python, roughly 3 µs
  per node" was the best case, not a typical one. Ten standalone best-of-three
  trials plus five in-process sweeps put one 100,000-op backward pass between
  300 ms and 470 ms (3.0 to 4.7 µs per op) on a machine whose load average ran
  4.9 to 6.6 on eight cores. README and the Limitations bullet now quote the
  range and say the machine was loaded.
- "Per node" was also the wrong denominator: `y = y * m` wraps the scalar `m`
  in a `Tensor` of its own, so the 100,000-op chain holds 200,001 live
  `Tensor`s and the walk visits all of them. The Limitations bullet said
  100,000.
- Panel 2's caption claimed "every public op". The ledger in
  `make_figures.correctness_ledger` runs fifteen of the sixteen public ops —
  `neg` is not in it. `examples/validate.py` does cover `neg`, which is why
  its sweep is 66 pairs where the panel's is 62. Caption now says so.
- "About 440 lines of NumPy" held only on the tightest of three counting
  rules. `tinydiff/*.py` is 654 physical lines, 492 non-blank/non-comment, 436
  with docstrings also removed. The headline now gives 654 and 436 and names
  the rule for each.
- "124 tests, about 5 seconds" — measured 3.45 s, 3.83 s and 3.95 s wall on
  three consecutive runs. Now "about 4 seconds". The 124/123 split was
  re-confirmed against a matplotlib-free venv: 123 passed, 1 skipped.
- "The closed forms in section 2 are what pin the last digits" pointed at
  `validate.py`'s "Canonical functions"; the op-by-op closed forms are section
  1. Same off-by-one fixed in `docs/validation.md`, which sourced the 62-cell
  spread to §4 (66 cells) rather than to panel 2 of the engine report.
- Panel B's "half a nat wrong" is `745 - 744.4401 = 0.56` nats; panel C's
  finite-difference floor prints `3.2e-11`, not `3e-11`. `1e-6` sits 236× (not
  "~240×") above the noisiest ledger cell — corrected in `docs/validation.md`
  and `examples/validate.py`.
- `fit_sine.py`'s "three orders of magnitude in 200 epochs" is a factor of 840
  — 2.9 decades — from the two losses printed directly above the claim.
- The `transformer_block.py` output blocks showed 3 of 9 and 1 of 9 parameter
  rows with nothing to mark the cut. Rows elided are now marked `...`, the
  nine-row list is named in the prose, and the elided rows' actual range
  (`1.60e-16` to `4.84e-16`) is stated.
- `tests/test_properties.py` was described by a ten-item list that reads as
  exhaustive; there are seventeen invariants (21 collected cases). The prose
  now gives the count and presents the list as ten of them.
- The recursion cliff is 995 ops when `make_figures.py` runs the probe and 996
  under `python -c` — it moves with the caller's stack depth. Panel 3's
  caption now says that instead of implying a fixed constant.
- `pyproject.toml` advertised "<500 LOC", true only on the
  non-blank/non-comment count. Now states 654 lines.

## [0.4.1] - 2026-08-03

### Changed
- `docs/validation.md` §2 and the matching docstring in `examples/validate.py`
  claimed every closed-form row is seeded with a random weight matrix. Two of
  the sixteen are not: `sum_` and `mean` are called bare, so they run with an
  all-ones seed. Both places now say so, and say which of the two still
  detects a reduction that ignores its upstream gradient (`mean` does, through
  its `1/N` scale; `sum_` does not).
- `docs/validation.md` §1 justified `tol=1e-6` as "about 100x above the noise"
  in the same sentence that put the finite-difference floor at `1e-9` — those
  two are 1000x apart. Replaced the inferred ratio with the measured spread of
  the 62 (op, shape) cells: `2.99e-13` to `4.23e-09`, median `2.29e-10`, so the
  tolerance sits ~240x above the noisiest cell. Same correction in
  `examples/validate.py` and in the README's panel-2 caption.
- `tests/test_properties.py::test_relu_is_positively_homogeneous` named two
  backward-mask errors it cannot catch, because neither is a different
  function from the original: a mask read off the output, and `>=` in place of
  `>`. The docstring now names a mask break the test does catch — a fixed
  non-zero threshold, which is the scale dependence the invariant exists to
  rule out — and says where `>=` is caught instead.
- `test_toposort_visits_children_before_parents` and
  `test_repeated_backward_accumulates_exactly_linearly` drew 50 samples while
  the docs advertised 200 for every invariant. Both now draw 200; the docs
  spell out that tests carrying several cases split the 200 across them.

## [0.4.0] - 2026-07-27

### Added
- `docs/validation.md` — every claim the library makes about its own
  correctness, checked against something outside it: closed-form derivatives
  for all sixteen ops, the Rosenbrock gradient at the conventional starting
  point, the logistic and softmax cross-entropy gradients, Euler's
  homogeneous function theorem, He's initialisation variance, Adam's
  bias-corrected first step, and a naive log-sum-exp evaluated at the shift
  where it stops returning a number. Includes a section on the places
  tinydiff does *not* match its reference, with the reason for each.
- `examples/validate.py` — the script that produces those numbers; exits
  non-zero if any row fails, and runs on every CI build.
- `tests/test_validation.py` — the same rows as assertions, so the doc
  cannot drift from the code without turning the build red.
- `tests/test_properties.py` — 21 randomised invariant tests, 200 draws
  each from a seeded generator: the directional derivative against a central
  difference, linearity of the VJP in its seed, the sum rule, Euler
  homogeneity, ReLU's positive homogeneity, fan-out accumulation, broadcast
  gradient conservation, the matmul adjoint identity at every operand rank,
  traversal order on random DAGs, exact linearity of repeated `backward`,
  softmax rows summing to zero, translation invariance of cross-entropy at
  shifts a naive `exp` cannot survive, the simplex bounds on the loss,
  convexity of the logistic NLL as a monotone gradient, the descent
  property, Adam's scale-free first step, and MSE's zero.
- `examples/transformer_block.py` — one transformer feed-forward sublayer
  built twice, once as hand-derived NumPy and once as tinydiff, with every
  gradient compared; then the same comparison with two injected bugs that a
  loss curve does not notice.
- `docs/validation.png` — naive versus max-shifted log-sum-exp and
  cross-entropy, and the finite-difference step-size sweep that sets the
  gradient-check tolerance.

### Changed
- `docs/theory.md` now uses `$...$` for inline maths. GitHub renders that;
  it does not render `\(...\)`, which was passing through literally on
  thirty-four expressions.
- The 995-op recursion depth in `docs/theory.md` is attributed to the
  interpreter that measured it rather than to a version number that was not
  the one under test.
- Panel 2 of `docs/engine_report.png` sizes its colourbar to the decades the
  ledger actually occupies; the fixed `-16..-8` scale left half of it empty.
- The README no longer restates timings that the figure already annotates,
  so a regenerated figure cannot silently disagree with the prose beside it.

## [0.3.0] - 2026-07-27

### Added
- `docs/engine_report.png` and `docs/spiral_training.png`, both produced by
  `examples/make_figures.py` from live calls into the library: a cost
  comparison against central differences, a per-op x per-shape gradient
  agreement ledger, a graph-depth sweep, and the spiral fit with per-layer
  gradient magnitudes.
- `examples/make_figures.py`, with a `--quick` mode CI runs as a smoke test so
  the figures cannot quietly stop reproducing.
- `tests/test_examples.py` - every example script runs in CI, and the spiral
  dataset's geometry is pinned.
- Held-out evaluation in `examples/spiral.py`: trained on one noise draw,
  scored on an independent one.

### Changed
- `spiral()` now generates two genuinely interleaved Archimedean spirals.
  Jitter is applied along and across the arm instead of as isotropic Cartesian
  noise, which at the old scale swamped the gap between the arms - the
  "two-spiral" dataset was a cloud, and the example topped out near 85%. It
  now reaches 100% train / 98.8% held out in 1500 epochs.
- `docs/theory.md` rewritten: forward-vs-reverse cost bounds, the traversal
  and re-entrancy argument, the `np.matmul` gufunc derivation, the
  finite-difference error floor, and where `sqrt(6 / fan_in)` comes from.
- README leads with what the engine measures about itself, and states its
  limitations rather than a roadmap.
- CI installs the `plot` extra so the figure script is exercised on every run.

## [0.2.0] - 2026-07-09

### Fixed
- `matmul` backward crashed (ValueError) for 1-D operands and for
  stacked/batched operands; it now implements the full `np.matmul` gufunc
  semantics — vector promotion, `swapaxes(-1, -2)` transposes, and
  `_unbroadcast` over broadcast batch dimensions.
- A second `backward()` through the same graph silently compounded stale
  intermediate grads into wrong numbers; it now raises unless
  `retain_graph=True` is passed, and repeated passes accumulate leaf grads
  correctly.
- Recursive graph traversal (`backward`, `zero_grad`) hit Python's
  recursion limit around 1000 ops; both walks are now iterative.
- `Linear` re-seeded `default_rng(0)` per layer, giving every same-shaped
  layer byte-identical weights; layers now share one process-wide generator
  (pass `rng=` explicitly for reproducibility).
- `grad_check` silently skipped inputs that received no gradient; a missing
  gradient on a `requires_grad` input now fails the check.
- `mean` crashed with TypeError for tuple axes; `mean(a, axis=(0, 1))` works.
- `backward(seed)` with a seed shape that does not match the output shape now
  raises instead of broadcasting into wrong gradients.

### Added
- `Tensor.backward(retain_graph=True)` for deliberate repeated backprop.
- Randomized 30-combination matmul shape sweep through `grad_check`
  (`tests/test_matmul.py`), plus deep-graph and re-entrancy regression tests.

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
