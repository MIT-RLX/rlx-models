# rlx-fgd — functional gradient descent with adaptive representations

Rust implementation of **[Functional Gradient Descent with Adaptive
Representations](https://arxiv.org/abs/2606.16926)** (Csillag, Schuller,
Dall'Antonia, Guibas, Velho, Novello — NeurIPS 2026): Algorithm 1, the
convergence theory of Section 3.1, and all four of the paper's experiments —
function fitting, RKHS regression, the wave equation, and radiance fields.

## What the paper is about

Fit a function by descending in *function space* rather than on the parameters
of a network. The catch is that a functional gradient is infinite-dimensional:
you cannot store it, so you approximate it — and a **fixed** approximation
leaves an error floor the iteration can never get below, however long it runs.

The paper's answer is to refine the representation *during* optimisation, using
a test built only from quantities you can compute, and to carry that
approximation through the analysis so the guarantees survive it:

```text
for t = 0, …, T−1:
    while true:
        g_t ← approximate ∇L(f_t) in the current representation
        U_t ← upper bound on ‖g_t − ∇L(f_t)‖_B
        if (1 + ε)·U_t < ε·‖g_t‖_B: break
        refine the representation
    f_{t+1} ← f_t − η·g_t
```

`‖g_t‖_B` is the norm of your own approximation, and `U_t` never needs
`∇L(f_t)` itself. That is what makes the method *implementable* — and, with
Lemma 3.9 guaranteeing the inner loop terminates, what makes Theorem 3.10's
convergence to a global minimiser apply to a program rather than to an idea.

## Layout

| Module | Contents |
|--------|----------|
| [`algorithm`](src/algorithm.rs) | Algorithm 1; the `Representation` and `Functional` traits; `Report` |
| [`theory`](src/theory.rs) | Lemma 3.5, Propositions 3.6/3.8, Theorem 3.10 in closed form |
| [`rep`](src/rep/) | dyadic grids (`L²`), CART trees (`L^∞`), frequency grids (`H²`), voxel grids + spherical harmonics |
| [`tasks`](src/tasks/) | `L²` fitting (Figure 1), RKHS regression (§4.1), the wave equation (§4.2), radiance fields (§4.3) |
| [`render`](src/render.rs) / [`scene`](src/scene.rs) | the volumetric renderer, cameras, NeRF-Synthetic loading |
| [`baseline`](src/baseline.rs) | the Fourier-feature MLP + Adam comparison |
| [`rlx`](src/rlx.rs) | §4.1's gradient kernel compiled to RLX backends |

Setting `Refinement::Fixed` turns the same driver into the paper's
fixed-representation baseline, so the central comparison runs on one code path
rather than two that might differ for uninteresting reasons.

## Running it

```bash
# Figure 1 — adaptive vs fixed grids on L² fitting
just fgd-teaser
cargo run -p rlx-fgd --release --example teaser -- --steps 100 --lr 0.05 --eps 3 --nn-steps 500

# Figure 2 — RKHS regression with adaptive trees
just fgd-regression
cargo run -p rlx-fgd --release --example kernel_regression -- --loss bce --max-depth 16

# Section 4.1's own dataset
curl -L https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/cod-rna -o /tmp/cod-rna
cargo run -p rlx-fgd --release --example kernel_regression -- --libsvm /tmp/cod-rna --loss bce

# Figure 3 — the wave equation in frequency space
just fgd-wave
cargo run -p rlx-fgd --release --example wave_equation -- --steps 1500 --max-level 6 --out /tmp/wave

# Figure 4 — inverse rendering; the synthetic scene needs no download
just fgd-radiance -- --steps 60 --fixed

# Figure 4 on the paper's own scene (~38 MB of a 1.27 GB archive)
just fetch-nerf-synthetic ficus
just fgd-radiance -- --scene /tmp/nerf_synthetic/ficus --views 24 --downsample 5 \
    --steps 200 --eps 0.4 --init-res 16 --max-level 3 --fixed --out /tmp/ficus

# the gradient kernel on a backend
cargo run -p rlx-fgd --release --features metal --example kernel_regression -- --device metal
```

Representative output (`--steps 100 --lr 0.05 --eps 3`; timings are from one
loaded machine and move around, the rest is deterministic):

```text
method                             dofs       L(f_T)   worst RelErr       time
fixed FGD, 16×16 grid               256     1.672e-2         8.6256      2.62s
fixed FGD, 64×64 grid              4096     1.616e-3        12.7808      3.24s
fixed FGD, 256×256 grid           65536     1.212e-4         7.4587      5.41s
adaptive FGD (ours)             1048576     6.358e-6         1.4424     11.67s
```

The adaptive run reaches the finest grid only at step 72, having spent the
first ten steps on a 16×16 one. `worst RelErr` above `ε/(1+ε) = 0.75` is the
level cap binding, which `Report::guarantees_hold` reports — see the notes
below.

Does it actually solve the PDE? `--steps 1500 --max-level 6 --eps 3` takes
`L_PDE` from `1.134e-1` to `1.606e-2`, and the recovered `t = 0` slice
correlates `+0.976` with the initial displacement `h` (at `+0.65` of its
amplitude, the unconverged remainder), while later slices decorrelate from it
as the wave propagates. At the default `--max-level 5` the same check gives
`+0.738` — the run is resolution-limited, not wrong.

## Tests

```bash
cargo test -p rlx-fgd --release
just features=macos-gpu test-fgd-backends     # Metal / MLX / wgpu parity
```

- `tests/theory.rs` — the closed forms against the algebra of the proofs
  (`max_learning_rate` is exactly where `r` vanishes; the `B = H` case collapses
  to the classical descent lemma), then Lemma 3.5 and Proposition 3.8 against an
  actual run.
- `tests/adaptive_vs_fixed.rs` — refinement preserves the encoded function for
  every representation; the `L^∞` error bound really bounds the sup-norm
  error; Proposition A.2's `α` and `β` verified against Assumptions 3.3/3.4;
  `h` and `ĥ` checked to be a Fourier pair; the fixed-vs-adaptive comparison.
- `tests/backend_parity.rs` — §4.1's kernel on CPU / Metal / MLX / wgpu / CUDA /
  ROCm / Vulkan against the `f64` host reference, including that the backends
  make the *same refinement decisions* and not merely close numbers.
- `tests/radiance.rs` — the harmonics checked orthonormal by quadrature and
  against the addition theorem; the renderer against the closed form for a
  uniform medium; `Camera::project` as the exact inverse of `Camera::ray_through`;
  **Proposition A.3's gradient against finite differences of the loss**; and a
  reconstruction from posed images alone, scored on novel views.

## Notes on the implementation

Places where this deviates from, or pins down something left open by, the
authors' [reference code](https://github.com/dccsillag/experiments-adaptive-fgd).
Section 4.3 has four more of its own, in its section below.

- **Two different tolerances.** Algorithm 1's `ε` and the `RelErr` bound in
  Propositions 3.6/3.8 are not the same number — running at tolerance `ε`
  guarantees `RelErr ≤ ε/(1+ε)` (Theorem 3.10(i)). `theory::guaranteed_relative_error`
  converts. Theorem 3.10 is stated for `ε ∈ (0,1)`, which is exactly what keeps
  that bound under the `1/2` Lemma 3.5 needs; the authors' own `ε = 3` teaser
  and `ε = 50` wave runs sit outside it, so those runs are the algorithm
  without the theorem. `Report::guarantees_hold` says which case you are in.

- **The `L²` error estimator.** The fitted constant on a cell is the *sample*
  mean, so it is itself off by `O(σ/√N)`; the unbiased estimate of
  `‖g − ∇L(f)‖²` on that cell is `s²(1 + 1/N)` with `s²` Bessel-corrected. The
  reference divides the squared deviations by `N`, understating the quantity
  Algorithm 1 tests by a factor `(N−1)/(N+1)` — 29% at its `N = 8`. (That choice
  has a compensating virtue, noted in `CellFit::from_samples`.) With the
  corrected estimator, Lemma 3.5's bound holds with margin at every step of the
  runs in `tests/theory.rs`; before the correction it was violated.

- **Losses are reported on a pinned quadrature.** Both `L2Fit` and
  `WaveEquation` take a `loss_quadrature` fixed for the whole run rather than
  read off the iterate's own grid. A quadrature that follows the refinement
  level changes scale exactly when the representation does, turning every
  refinement into a step change in the reported loss that has nothing to do
  with the iterate — which makes the adaptive-versus-fixed comparison, the
  whole point of Figure 1, meaningless. This was a real bug here before it was
  a note: the wave loss appeared to *rise* on the first step at every step size.

- **The `H²` symbol.** Proposition 4.2 divides by `1 + |ω|² + |ω|⁴`, and the
  same weight defines the `H²` norm used for `‖g‖` and `U`. The reference code
  writes the gradient's denominator as `(1 + |ω|²)²` while using the paper's
  form for the norm; the paper's self-consistent version is used throughout
  here. Frequencies follow the `e^{2πiω·x}` convention, in which Proposition
  4.2's `2π` factors are absorbed into the transform.

- **Cross-entropy on logits.** The paper writes `ℓ` in probability form; the
  reference (and this port) use the logit form, which is the only one that
  stays finite for an unconstrained `f`, and is what the `1/4`-smoothness
  refers to.

- **A refinement cap.** Algorithm 1 has no bound on the inner loop; a program
  needs one. `Refinement::Adaptive { max_level }` caps it, and a step that hits
  the cap is recorded as not having met the tolerance, which is what makes
  `Report::guarantees_hold` false. Note that a *large* step size makes the cap
  bind quickly — `‖g‖` falls to the representation's error floor within a few
  iterations, so the budget has to double every few steps to keep up. That is
  why the authors run the teaser at `η = 0.05`.

## Section 4.3 in particular

Inverse rendering is the one task whose gradient is not a couple of lines, so
it is worth saying how it was checked. `tests/radiance.rs` compares
Proposition A.3's analytic gradient against central finite differences of the
photometric loss, for both density and colour coefficients. Both sides are
Monte-Carlo — the render averages jittered rays, the gradient averages points
inside a voxel — so the check is a convergence claim: at 256 rays/pixel and
4096 samples/voxel the two disagree by 10%, at 2048/32768 by 3%, and at
16384/262144 every probe agrees to within **0.4%**. Nothing short of that
distinguishes a correct Jacobian from one that is merely plausible.

### What the comparison actually shows

On the synthetic scene (16 views, 48×48, 80 steps, grids from 8³ to 64³):

```text
                       train loss   novel PSNR           work     time
fixed  8³               4.343e-3     24.84 dB         532_480      3.3s
fixed 16³               1.132e-3     30.13 dB       4_259_840      6.4s
fixed 32³               2.006e-4     36.85 dB      34_078_720     15.8s
fixed 64³               5.491e-5     40.58 dB     272_629_760    158.3s
adaptive, ε = 0.4       6.654e-5     38.53 dB     245_859_328    214.7s
adaptive, ε = 1.5       8.154e-5     37.64 dB     213_058_560    147.1s
adaptive, ε = 5.0       9.078e-5     37.45 dB     182_121_472     99.4s
```

Adaptive beats every fixed grid it can afford: at ε = 5.0 it lands `2.2×` below
fixed 32³, the only fixed grid whose cost fits in its budget, because the next
rung of the ladder costs 50% more than it spent. That is the claim the
`adaptive_sits_ahead_of_the_fixed_representation_frontier` test makes for
Figure 1, and it holds here too.

It does not *dominate* the frontier, though. Interpolating the fixed runs
between 32³ and 64³ in log-work/log-loss puts adaptive about 13–20% above that
line at equal work, consistently across ε. Two caveats before reading much into
that: the interpolated frontier is not something you can actually buy — fixed
grids come in powers of two — and this scene is two smooth blobs, which a 32³
grid nearly resolves, so there is little for a coarse phase to exploit. The
paper's Ficus scene at 160×160 and 128³ has thin structure at scales a coarse
grid cannot see at all.

### On the paper's own scene

`just fetch-nerf-synthetic ficus` then the `--scene` command above runs
Section 4.3's actual setting: NeRF-Synthetic Ficus, 24 training and 25 held-out
views at 160×160, 200 steps, `ε = 0.4`, grids from 16³ to 128³, `η_σ = 20` and
`η_c = 8`. Novel-view PSNR is scored against the test images themselves.

*(Table pending — the 128³ and adaptive rows are still running; fixed 16³/32³/64³
came in at 19.82 / 22.10 / 23.37 dB.)*

Four things this implementation does differently from the reference:

- **The colour gradient is computed in closed form, not fitted.**
  `∇_{c_k}L(x, ·)` is a sum of kernels `K(ω_i, ·)`, and `K(ω, ω') = Σ_j Y_j(ω) Y_j(ω')`
  is the reproducing kernel of the span of the harmonics — so expanding it
  gives the coefficients directly, `[∇_{c_k}L]_j = (1/N) Σ_i a_i [D̃_i]_k T_i σ Y_j(ω_i)`.
  The representation holds the colour gradient *exactly* in the direction
  variable; only the spatial part is approximated. The reference instead fits
  the harmonics by least squares over sampled directions, which recovers the
  same coefficients when that system has full rank and silently drops
  components when it does not. Computing them removes the sampling, the
  pseudo-inverse and that failure mode together.

- **Refinement re-imposes the visual hull.** A coarse voxel straddling the
  silhouette counts as occupied, and its children outside the hull inherit its
  density — which the gradient is identically zero on, so nothing ever removes
  it. On the test scene, leaving it in stalls the run at `1.1e-3` where
  re-imposing the hull reaches `1.1e-4`; the difference is a permanent blocky
  shell. This is what `Functional::after_refine` exists for.

- **Directions are normalised everywhere.** The reference's RKHS kernel
  normalises its arguments but its renderer evaluates the harmonics at the raw,
  unnormalised ray direction, whose length varies across the image plane. The
  harmonics are only orthonormal on the sphere.

- **The density and colour error terms are weighted consistently.** The
  reference's density residual carries a ½ that its colour residual and both
  norms do not, so the density half of `RelErr`'s numerator is understated by
  √2 against its own denominator. (Its two density branches disagree with each
  other on this too.)

Rendering is a DDA walk through voxels, which is pointer-chasing rather than
dense linear algebra, so this task runs natively rather than through an RLX
graph; it is parallel across voxels and views instead. §4.1's kernel is the one
that is genuinely a matmul, and that is the one `src/rlx.rs` compiles.
