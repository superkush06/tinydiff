# Reverse-mode autodiff, in enough detail to reimplement it

This is the reasoning behind the code in `tinydiff/`. It assumes you can read
a chain rule and would like to know why the library is shaped the way it is.

---

## 1. Why reverse and not forward

Take a program that computes a scalar loss $L$ from $n$ inputs. Every
differentiation strategy is a way of contracting the Jacobians of its
elementary operations; they differ only in the order of contraction, and the
order is everything.

**Finite differences.** Perturb one input, re-run, subtract. Each partial
costs one (central: two) evaluations of $L$, so the full gradient costs
$O(n)$ forward passes — and every entry carries a truncation error you
cannot make arbitrarily small (see §5).

**Forward mode.** Carry a derivative alongside every value. One pass computes
$\partial L / \partial x_i$ for a *single* $i$ — a Jacobian-vector
product, one column at a time. Full gradient: $O(n)$ passes again. Forward
mode is the right tool when $n$ is small and the output is large.

**Reverse mode.** Carry the derivative of the *output* with respect to every
intermediate, backwards. One pass computes a vector-Jacobian product
$\bar{x}^\top = \bar{y}^\top J$, which for a scalar output *is* the whole
gradient. Full gradient: **one** pass, at a constant multiple of the forward
cost — typically 2–3×, and independent of $n$.

Training is exactly the regime where $n$ is enormous (every weight) and the
output is a single number, which is why every deep-learning framework is
reverse-mode. Panel 1 of `docs/engine_report.png` is this paragraph, measured.

## 2. The mechanism

For every operation $z = f(x, y)$ we need the *local* partials
$\partial f/\partial x$ and $\partial f/\partial y$. When the upstream
gradient $\bar z = \partial L / \partial z$ arrives, the chain rule says

$$\bar x \mathrel{+}= \bar z \cdot \frac{\partial f}{\partial x}, \qquad
  \bar y \mathrel{+}= \bar z \cdot \frac{\partial f}{\partial y}.$$

The `+=` matters: a tensor used twice receives two contributions, and the
total derivative is their sum. Overwriting instead of accumulating is the
classic first bug in a hand-rolled engine.

In `tinydiff`, each op returns a `Tensor` carrying

- `_children` — the input tensors, i.e. the edges of the graph, and
- `_backward()` — a closure that performs the two `+=` above.

The forward pass builds the graph as a side effect of computing values. The
backward pass is then just: visit every node once, in an order where a node is
visited only after everything that consumes it.

## 3. Traversal: order, and the stack

That order is a reverse topological sort of the DAG. The textbook version is a
recursive depth-first search — six lines, and correct, and it stops working at
around a thousand operations because each visited node holds a Python stack
frame. `tinydiff` measures the exact depth where that happens rather than
quoting it — 995 ops on the interpreter that produced the figure, at the
default `sys.setrecursionlimit(1000)` — and uses an explicit stack, so graph
depth is bounded by heap memory rather than by `sys.getrecursionlimit()`. An
unrolled sequence model is trivially deeper than a thousand ops, so this is not
a hypothetical.

**Re-entrancy.** Calling `backward()` twice on one graph is not a doubling.
Leaf gradients accumulate, which is what you want; *interior* gradients also
accumulate and are then re-propagated, which is not. For $z = 2x^2$ at
$x = 3$, two naive passes give `x.grad = 48` where correct accumulation is
$24$. `tinydiff` therefore refuses the second pass unless you ask for it with
`retain_graph=True`, and clears interior gradients before each pass so the
leaves accumulate cleanly.

## 4. Shapes

Two shape problems account for most of the code in `ops.py`.

**Broadcasting.** If `a + b` broadcasts `b` from `(C,)` to `(N, C)`, then
$\partial L / \partial b$ arrives with shape `(N, C)` while `b` has shape
`(C,)`. Broadcasting is a linear map that copies; its adjoint is a linear map
that sums. So the gradient is summed over the axes that were expanded — that
is all `_unbroadcast` in `tensor.py` does, and every elementwise op routes
through it.

**`np.matmul`'s gufunc signature.** For 2-D operands the gradients are the
familiar pair

$$\bar A = \bar C B^\top, \qquad \bar B = A^\top \bar C.$$

`np.matmul` accepts a good deal more than 2-D, and each extension changes the
backward pass:

1. **1-D promotion.** A 1-D left operand acts as a `(1, k)` row, a 1-D right
   operand as a `(k, 1)` column, and NumPy then *squeezes those axes out of
   the result*. So the incoming gradient has to have them re-inserted before
   the contraction, and the finished gradient squeezed back down.
2. **Batch transposes.** For stacked operands only the last two axes are the
   matrix. The transpose is `swapaxes(-1, -2)`. Plain `.T` reverses *all* axes
   and misaligns every batch dimension — which raises for some shapes and, less
   kindly, returns correctly-shaped wrong numbers for others.
3. **Broadcast batch dims.** `(4,3,2) @ (2,5)` shares one `B` across four
   batches, so $\bar B$ must be summed over the batch axis — `_unbroadcast`
   again.

## 5. Why the numerical check bottoms out around 1e-9

`grad_check` compares against a central difference, and it is worth knowing
what that comparison can and cannot prove.

For step $h$, the central difference carries truncation error
$\tfrac{h^2}{6} f'''$ and round-off error
$\sim \epsilon_{\text{mach}} |f| / h$. Minimising their sum gives

$$h^\star \sim \left(\frac{3\,\epsilon_{\text{mach}}\,|f|}{|f'''|}\right)^{1/3}
  \approx 10^{-5},\qquad
  \text{error}(h^\star) \sim \epsilon_{\text{mach}}^{2/3}
  \approx 4\times10^{-11}.$$

With `eps = 1e-6` the floor lands around $10^{-10}$–$10^{-9}$, which is
exactly where the darker cells in panel 2 of the engine report sit. Those cells
are not autodiff error; they are the *reference method's* error. The practical
consequence: a checker at this precision detects wrong formulas and misaligned
axes, and cannot detect a relative error below about $10^{-8}$. Tolerances
tighter than that are cargo cult.

The checker's other job is structural rather than numerical: an input that
receives **no** gradient fails the check. Silently skipping such an input is
how a broken op — one that forgets to write into a child — sails past a
verifier that only compares the numbers it was handed.

## 6. Two details that look like typos and are not

**`sqrt(6 / fan_in)` in `Linear`.** Kaiming initialisation wants
$\operatorname{Var}[W] = 2 / \text{fan\_in}$ for a ReLU network — the 2
compensates for ReLU zeroing half the units. A uniform distribution on
$(-b, b)$ has variance $b^2/3$. Setting $b^2/3 = 2/\text{fan\_in}$ gives
$b = \sqrt{6/\text{fan\_in}}$. The 6 is Kaiming's 2 times the uniform's 3; it
is not Glorot's $\sqrt{6/(\text{fan\_in} + \text{fan\_out})}$ with a term
dropped.

**`softmax_crossentropy` is one op, not two.** Written naively as
`log(softmax(z))[target]` it overflows for large logits and loses precision on
confident predictions. Fused and max-shifted, with
$p = \operatorname{softmax}(z)$, the gradient collapses to something you can
write down exactly:

$$\frac{\partial L}{\partial z_{ij}}
  = \frac{p_{ij} - \mathbb{1}[\,y_i = j\,]}{N}.$$

No exponential appears in the backward pass at all, and each row of the
gradient sums to zero — a property the tests pin, because it is the softmax's
translation invariance showing up as a gradient identity.

## 7. What this design cannot do

Every `_backward()` closure computes NumPy arrays. That keeps the backward pass
fast and the code readable, and it makes higher-order differentiation
structurally impossible: to obtain $\nabla^2$, the backward pass itself has
to be assembled from differentiable ops, so that a gradient is a graph node
rather than a number. JAX and PyTorch both do this. It is a change to every op,
not a flag.

## 8. Where the arguments above get checked

Every claim in this file that is a number rather than a derivation is
checked against something outside the library, and the checking is written
down in [`validation.md`](validation.md): closed forms for §2's chain rule,
Giles' matrix results for §4's matmul pair, a measured error floor for §5,
He's variance for §6, and a naive log-sum-exp evaluated at the shift where
it stops returning a number. That page also lists the places tinydiff does
*not* agree with its references, which is the more interesting half.

## References

- Baydin, Pearlmutter, Radul & Siskind (2018), *Automatic Differentiation in
  Machine Learning: a Survey*, JMLR 18(153).
- Griewank & Walther (2008), *Evaluating Derivatives*, 2nd ed. — chapters 3–4
  for the forward/reverse cost bounds.
- He, Zhang, Ren & Sun (2015), *Delving Deep into Rectifiers* — the
  initialisation in §6.
- Goodfellow, Bengio & Courville (2016), *Deep Learning*, §6.5.
- Karpathy, *micrograd* — the same idea at scalar granularity, and the reason
  this file exists.

Full citations, including the ones behind the validation tables, are at the
bottom of [`validation.md`](validation.md).
