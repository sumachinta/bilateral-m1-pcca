# pCCA-FA Model Outputs — Reference Notes

Project: BILATERAL-M1-PCCA
Source: `SmithLabNeuro/pcca_fa` (`pcca_fa_mdl.py`), as called by `fit_session_pcca()` / `extract_session_metrics()`

---

## 1. How the model runs (context for everything below)

pCCA-FA decomposes the trial-to-trial spike count variance of two simultaneously recorded
populations (here: left hemisphere M1 = "area 1", right hemisphere M1 = "area 2") into three
pieces per neuron:

- an **across-hemisphere** component — variance explained by latent variables shared by
  *both* hemispheres (loading matrices `W_1`, `W_2`)
- a **within-hemisphere** component — variance explained by latent variables private to
  *one* hemisphere (loading matrices `L_1` for LH, `L_2` for RH)
- an **independent** component — variance unique to that one neuron, not shared with
  anything (`psi_1`, `psi_2`)

It is fit with an EM algorithm. The catch is that you don't know in advance how many
across-hemisphere latents (`d`) or within-hemisphere latents (`d1` for LH, `d2` for RH) to
use. `fit_session_pcca()` handles this by calling `model.crossvalidate()`, which:

1. Builds every combination of `d`, `d1`, `d2` from 0–6 (343 combos with `d_max=6`).
2. For each combo, runs 10-fold CV: fit on 9 folds, evaluate held-out log-likelihood and
   leave-one-out prediction error on the 10th fold, sum across folds.
3. Picks the `(d, d1, d2)` combo with the highest summed cross-validated log-likelihood.
4. Refits the model one final time on **all** trials using that winning combo — this final
   fit is what `model.params` holds afterward.
5. Automatically also runs `compute_cv_canonical_corrs()`, a second round of 10-fold CV
   that produces `cv_rho` (see §4).

So by the time `fit_session_pcca()` returns, `model` already contains the final fitted
parameters, and `cv_results` is a record of the hyperparameter search that got you there.
`extract_session_metrics()` then calls `model.compute_metrics()` to compute the descriptive
statistics (psv, dshared, etc.) from those final fitted loading matrices, and builds your
`within_ratio` from one piece of that (`psv`).

---

## 2. `model.params` — the fitted model parameters

Accessed via `model.get_params()` or `model.params` directly. This is the actual fit: the
loading matrices and noise variances that define the model.

| Key | Shape | What it is |
|---|---|---|
| `mu_x1`, `mu_x2` | (n1,), (n2,) | Per-neuron mean spike count, LH and RH |
| `W_1`, `W_2` | (n1, d), (n2, d) | **Across-hemisphere loading matrices.** Column *k* is the weight each neuron places on across-hemisphere latent variable *k*. `W_1` and `W_2` share the same `d` latent variables — that's what makes them "across-hemisphere": the same latent drives both hemispheres simultaneously, just with different per-neuron weights. |
| `L_1`, `L_2` | (n1, d1), (n2, d2) | **Within-hemisphere loading matrices.** `L_1`'s latents are private to LH, `L_2`'s are private to RH — they don't interact across hemispheres at all. |
| `psi_1`, `psi_2` | (n1,), (n2,) | Private (independent) variance per neuron — the part of each neuron's variance not explained by any latent variable. |
| `L_total` | (n1+n2, d+d1+d2) | The full stacked, block-structured loading matrix used internally by the EM algorithm (concatenation of `W_1`/`L_1`/zeros for LH rows, `W_2`/zeros/`L_2` for RH rows). You generally won't need this directly — `W_1/W_2/L_1/L_2` are the same data, already split out. |
| `d`, `d1`, `d2` | scalars | The selected dimensionalities — across-hemisphere, within-LH, within-RH respectively. |
| `cv_rho` | (d,) | Added after `crossvalidate()` runs (see §4). Cross-validated canonical correlations. |

**Interpretation note:** `W_1 @ W_1.T` (an n1×n1 matrix) gives you the portion of LH's
neuron-by-neuron covariance attributable to across-hemisphere latents; `L_1 @ L_1.T` gives
the portion attributable to within-LH latents. This is the mathematical basis for every
downstream metric.

---

## 3. `cv_results` — the dimensionality search record

Returned directly by `model.crossvalidate()` (your `fit_session_pcca()` return value).
Documents the hyperparameter search itself, not the final fit.

| Key | Shape | What it is |
|---|---|---|
| `d_list`, `d1_list`, `d2_list` | (343,) each | Every `(d, d1, d2)` combination tested (flattened meshgrid over 0–6 for each) |
| `LLs` | (343,) | Summed cross-validated log-likelihood for each combo, summed across all 10 folds |
| `PEs` | (343,) | Summed leave-one-out prediction error (squared error) for each combo, summed across folds |
| `d`, `d1`, `d2` | scalars | The winning dimensionalities (highest `LLs`) |
| `final_LL` | scalar | The CV log-likelihood of the winning combo (i.e. `LLs[argmax]`) |

**Interpretation:**
- Higher `final_LL` (less negative) = the model explains held-out trials better.
- `final_LL` is **not directly comparable across sessions** with different neuron counts —
  it scales with the number of neurons. Use it as a within-session diagnostic (e.g. "did
  the search converge on a clear winner, or was the likelihood surface flat?"), not as a
  cross-session quality metric.
- Worth occasionally checking: if the selected `d`, `d1`, or `d2` sits at the edge of the
  search grid (0 or 6), the true optimum might be outside the range you searched and you'd
  want to extend `d_max`.

---

## 4. `metrics` — descriptive statistics computed from the fitted loading matrices

Returned by `model.compute_metrics()`, called inside `extract_session_metrics()`. This is
where the loading matrices in `model.params` get turned into interpretable numbers. It's a
dict of five (or six) sub-dicts/arrays:

```python
metrics = {
    'dshared': {...},      # shared dimensionality
    'psv': {...},          # percent shared variance — your primary metric source
    'part_ratio': {...},   # participation ratio
    'load_sim': {...},     # loading similarity
    'rho': array,          # canonical correlations
    'cv_rho': array,       # only present if crossvalidate() was run (it was)
}
```

### 4a. `psv` — percent of variance shared (your primary metric's source)

For each neuron, %sv asks: of this neuron's total variance, what fraction is explained by
across-hemisphere latents vs. within-hemisphere latents vs. independent noise? These three
always sum to 100% per neuron.

| Key | Shape | Meaning |
|---|---|---|
| `psv_W_1`, `psv_W_2` | (n1,), (n2,) | Per-neuron % variance explained by across-hemisphere latents, LH / RH |
| `psv_L_1`, `psv_L_2` | (n1,), (n2,) | Per-neuron % variance explained by within-hemisphere latents, LH / RH |
| `ind_var_x1`, `ind_var_x2` | (n1,), (n2,) | Per-neuron % independent variance, LH / RH |
| `avg_psv_W_1`, `avg_psv_W_2` | scalar | `psv_W_*` averaged over neurons in that hemisphere |
| `avg_psv_L_1`, `avg_psv_L_2` | scalar | `psv_L_*` averaged over neurons in that hemisphere |
| `avg_psv_W_total` | scalar | `avg_psv_W_1` and `avg_psv_W_2` pooled and averaged over **all** neurons in both hemispheres |
| `avg_psv_L_total` | scalar | Same, for within-hemisphere |

**This is what `within_ratio` is built from:**
`within_ratio = avg_psv_L_total / (avg_psv_L_total + avg_psv_W_total)`.
High `within_ratio` → variance is more within-hemisphere-dominated. Low `within_ratio` →
more across-hemisphere-dominated. As your project notes already establish, this number is
only meaningful relative to other sessions, not in isolation.

### 4b. `dshared` — shared dimensionality

For each loading matrix, this asks: how many latent dimensions are actually doing the
work, vs. how many were just allowed by the model? It's the minimum number of components
of `W W^T` (or `L L^T`) needed to explain 95% of that matrix's variance (via eigenvalues /
singular values), so `dshared ≤ d` (or `≤ d1`/`d2`) always.

| Key | Meaning |
|---|---|
| `dshared_W_1`, `dshared_W_2` | Effective across-hemisphere dimensionality, judged from LH / RH loadings separately |
| `dshared_W_total` | Effective across-hemisphere dimensionality judged from `W_1` and `W_2` stacked together |
| `dshared_L_1`, `dshared_L_2` | Effective within-hemisphere dimensionality, LH / RH |

**Interpretation:** a larger `dshared` means more complex, higher-dimensional interactions;
a smaller one means the shared activity is closer to "everything moves together along one
axis." If `d` was selected as 6 by cross-validation but `dshared_W_total` comes out as 2,
that tells you most of the cross-validated benefit is coming from a couple of dominant
latent dimensions, not all 6 equally.

### 4c. `part_ratio` — participation ratio

A second, continuous-valued measure of effective dimensionality (an alternative to the
hard 95%-cutoff used by `dshared`), computed as `(Σs)² / Σ(s²)` where `s` are the
eigenvalues of `W W^T` or `L L^T`.

| Key | Meaning |
|---|---|
| `pr_W_1`, `pr_W_2`, `pr_W_total` | Participation ratio for across-hemisphere loadings (LH / RH / pooled) |
| `pr_L_1`, `pr_L_2` | Participation ratio for within-hemisphere loadings, LH / RH |

**Interpretation:** ranges from 1 (all variance along a single latent dimension) up to the
total number of latents available (variance spread evenly across all of them). Useful as a
sanity check alongside `dshared` — they should generally tell a consistent story about
how concentrated vs. distributed the shared variance is.

### 4d. `load_sim` — loading similarity

For each latent dimension, this measures how *uniformly* neurons weight onto it (a metric
from Umakantha, Morina, Cowley et al. 2021), computed as `1 − n·Var(orthonormalized
loading column)`, where `n` is the neuron count.

| Key | Meaning |
|---|---|
| `ls_W_1`, `ls_W_2` | Loading similarity per across-hemisphere latent dimension, LH / RH |
| `ls_L_1`, `ls_L_2` | Loading similarity per within-hemisphere latent dimension, LH / RH |

**Interpretation:** values close to 1 mean most neurons in that hemisphere have a similar
weight on that latent (a "synchronous"-like pattern — neurons rising and falling together).
Values close to 0 mean weights are highly heterogeneous across neurons (some neurons
strongly positive, others strongly negative or near-zero on that same latent). This is a
different axis of information than %sv — two sessions could have identical `within_ratio`
but very different `load_sim`, meaning the *type* of co-fluctuation differs even though its
overall magnitude doesn't.

### 4e. `rho` — canonical correlations

`rho` is a `(d,)` array, one value per across-hemisphere latent dimension, computed in
closed form from the model's estimated covariances (via `get_canonical_directions()`,
which is essentially CCA performed on the model's own fitted covariance matrices, ordered
by descending correlation strength).

**Interpretation:** each `rho[k]` is how strongly LH and RH co-vary along that one shared
latent direction, on a 0–1 scale (1 = perfectly correlated). Because `rho` is computed from
the same data the model was fit on, it can be optimistically biased (especially with few
trials and many neurons) — it tells you about the *fitted* relationship, not necessarily
how well that relationship would generalize.

### 4f. `cv_rho` — cross-validated canonical correlations

Same concept as `rho`, but honest: in each of the 10 CV folds, the model is refit on the
training trials only, the held-out trials are projected through the across-hemisphere
loadings to get per-trial latent estimates for each hemisphere, and the actual Pearson
correlation between LH's and RH's estimated latent trajectories is computed on those
held-out trials. Repeated for each of the `d` dimensions.

**Interpretation:** `cv_rho` is the metric to trust over `rho` when you want to know "how
reproducible is this cross-hemisphere correlation," since it isn't inflated by the model
having seen the same trials it's being evaluated on. A large gap between `rho` and `cv_rho`
for a given dimension is itself informative — it signals that dimension's correlation may
be partly an artifact of limited trial count (overfitting) rather than a robust effect.

---

## 5. `summary` — the trimmed-down dict your pipeline actually saves/compares

Built by `extract_session_metrics()`, this pulls just the numbers needed for the
group-level analysis, mostly straight out of `psv` (§4a) plus `d`/`d1`/`d2` from
`model.params`.

| Key | Source | Meaning |
|---|---|---|
| `session_id` | argument passed in | Session identifier (e.g. `"U02"`) |
| `avg_psv_W` | `psv['avg_psv_W_total']` | Across-hemisphere %sv, pooled over all neurons |
| `avg_psv_L` | `psv['avg_psv_L_total']` | Within-hemisphere %sv, pooled over all neurons |
| `within_ratio` | computed | **Primary metric.** `avg_psv_L / (avg_psv_L + avg_psv_W)` |
| `avg_psv_W_lh`, `avg_psv_L_lh` | `psv['avg_psv_W_1']`, `psv['avg_psv_L_1']` | Same split, LH only |
| `avg_psv_W_rh`, `avg_psv_L_rh` | `psv['avg_psv_W_2']`, `psv['avg_psv_L_2']` | Same split, RH only |
| `within_ratio_lh`, `within_ratio_rh` | computed | Hemisphere-specific within_ratio — for the (flagged underpowered, n=6/group) hemispheric asymmetry secondary analysis |
| `d`, `d1`, `d2` | `model.params` | Selected dimensionalities for this session, useful to report alongside `within_ratio` since sessions with very different `d`/`d1`/`d2` may not be perfectly comparable |

**Note:** `summary` currently doesn't carry `dshared`, `part_ratio`, `load_sim`, `rho`, or
`cv_rho` forward — those live only in the full `metrics` dict returned alongside it. If the
secondary analyses (e.g. characterizing *how* cross-hemisphere co-fluctuation differs
between groups, not just *how much*) end up needing `load_sim` or `dshared`, those will
need to be pulled from `metrics` and added to what gets saved/cached per session.

---

## 6. Quick-reference: where everything comes from

```
model.params          ← model.train() (called inside crossvalidate())
  ├── W_1, W_2         (across-hemisphere loadings)
  ├── L_1, L_2         (within-hemisphere loadings)
  ├── psi_1, psi_2     (independent variance)
  ├── d, d1, d2        (selected dims)
  └── cv_rho           (added by crossvalidate() → compute_cv_canonical_corrs())

cv_results             ← model.crossvalidate() return value
  ├── d_list, d1_list, d2_list, LLs, PEs   (full grid search)
  └── d, d1, d2, final_LL                  (winning combo)

metrics                ← model.compute_metrics()
  ├── psv          ← compute_psv()          (used for within_ratio)
  ├── dshared      ← compute_dshared()
  ├── part_ratio   ← compute_part_ratio()
  ├── load_sim     ← compute_load_sim()
  ├── rho          ← get_canonical_directions()
  └── cv_rho       ← model.params['cv_rho'] (pulled in if present)

summary                ← extract_session_metrics(), your own code
  └── built from metrics['psv'] + model.params['d'/'d1'/'d2']
```
