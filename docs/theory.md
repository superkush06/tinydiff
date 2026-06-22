# Reverse-mode autodiff in 5 minutes

## The problem

You have a computation graph that evaluates a scalar loss \(L\) from inputs
\(x_1, \ldots, x_n\). You want \(\partial L / \partial x_i\) for each input,
fast.

Naive symbolic differentiation explodes. Numerical (finite differences)
needs O(n) evaluations of \(L\) — too slow for large \(n\).

**Reverse-mode autodiff** computes all \(n\) partials in **one** backward pass,
costing the same as one forward pass (up to a small constant).

## The trick

For each operation \(z = f(x, y)\), define a **local Jacobian** \(\partial f / \partial x\)
and \(\partial f / \partial y\). When the upstream gradient \(\partial L / \partial z\)
arrives, push it into inputs by chain rule:

\[
\frac{\partial L}{\partial x} \mathrel{+}= \frac{\partial L}{\partial z} \cdot \frac{\partial f}{\partial x}.
\]

In code, every op produces a tensor with a `_backward()` closure that knows
how to do this push. After the forward pass we have a DAG of these closures;
walking the DAG in reverse topological order propagates gradients to every
leaf in one pass.

## What `tinydiff` does

1. `Tensor` wraps a numpy array and remembers its `_children` + `_backward`.
2. Each op (e.g. `add`, `matmul`) returns a new Tensor whose `_backward` knows
   the local Jacobian.
3. `Tensor.backward()` does a reverse-topo traversal and accumulates grads.

## Broadcasting

If `a + b` broadcasts `b` from shape `(C,)` to `(N, C)`, the gradient \(\partial L / \partial b\)
has shape `(N, C)` but `b` has shape `(C,)`. We sum out the broadcast axes — see
`_unbroadcast` in `tensor.py`.

## Numerical stability

Some operations need careful implementation to avoid overflow:

- **softmax_crossentropy** shifts logits by their max before exponentiating.
- We avoid the explicit `exp` of `softmax(x)` chained with `log(...)` and
  instead fuse the loss + gradient into one op.

## References

- Baydin, Pearlmutter, Radul, Siskind (2017), *Automatic Differentiation in
  Machine Learning: a Survey*.
- Andrej Karpathy, *micrograd* — the inspiration for this design at a smaller
  scale (scalars only).
- Goodfellow, Bengio, Courville, *Deep Learning*, ch. 6.5 (backpropagation).
