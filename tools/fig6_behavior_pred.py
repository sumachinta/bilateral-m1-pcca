"""
fig6_behavior_pred.py
=====================
Figure 6 analog for the bilateral-M1 pCCA-FA project.

The paper (McDonnell et al., Nat Commun 2026, Fig. 6) regresses the pCCA-FA
LATENT time-courses onto trial-by-trial pupil diameter, and finds the
across-area latents predict pupil far better than the within-area latents
(across-area = a global/brain-wide process).

Here we do the behavioral analog Scott asked for: regress the latents onto
trial-by-trial LICK LATENCY instead of pupil, and compare training groups.
Three latent sets per session (exactly the paper's split):
    across      -> z_across_*      (shared by both hemispheres)
    within_LH   -> z_within_LH_*   (private to left hemisphere)
    within_RH   -> z_within_RH_*   (private to right hemisphere)

Predictions to evaluate (Scott, Comment 4):
  (a) RH within-latents predict lick latency better in bilateral than unilateral.
      NOTE laterality caveat: task-relevant right-side input is contralateral to
      the LEFT cortex, so LH may be the biologically correct side. We therefore
      report BOTH hemispheres and leave the interpretation flagged.
  (b) The gap (across r^2 - within r^2) is smaller in unilateral experts.

Method notes
------------
* Latents come from the fitted model's E-step (get_trial_latents), lick latency
  from build_trial_variable_table -- the same infrastructure used elsewhere in
  this repo, aligned to the same trial_indices the model was fit on.
* Lick latency is only defined on trials with a lick (hit / false-alarm); no-lick
  trials are NaN and dropped.
* We report TWO r^2 values per latent set:
    - r2_insample : LinearRegression().score(X, y)  -- matches the paper's default
                    (CROSSVAL=False), but inflates with the number of latent
                    dimensions, which differs across sessions/groups.
    - r2_cv       : 10-fold cross_val_predict r^2  -- the fair metric for a
                    GROUP comparison, since it is not biased by latent-dim count.
  Use r2_cv as primary when comparing groups; r2_insample for the paper analog.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score


COMPONENTS = {
    'across':    'z_across',
    'within_LH': 'z_within_LH',
    'within_RH': 'z_within_RH',
}


# ---------------------------------------------------------------------------
# per-session assembly
# ---------------------------------------------------------------------------
def load_session_latents_behavior(session_id, results_dir='results', data_dir='data'):
    """Assemble one session's latent time-courses + trial behavior.

    Returns a DataFrame with columns z_across_*, z_within_LH_*, z_within_RH_*,
    plus 'lick_latency', 'outcome', 'stimulus'. One row per trial, aligned to
    the trials the pCCA-FA model was fit on.

    Uses this repo's existing helpers so the alignment matches the rest of the
    pipeline. Must be called with the repo root on sys.path.
    """
    from tools.neuron_behavior_analysis import (
        load_saved_session, base_session_id, get_trial_latents,
        build_trial_variable_table,
    )
    from tools.load_session import load_session
    from tools.trial_epoching import compute_derived

    payload = load_saved_session(session_id, results_dir=results_dir)
    pcca_input_data = payload['pcca_input_data']
    metrics = payload['metrics']

    session_data = load_session('{}/{}.mat'.format(data_dir, base_session_id(session_id)))
    derived = compute_derived(session_data)

    latents = get_trial_latents(payload, pcca_input_data)
    behav = build_trial_variable_table(
        session_id, metrics, pcca_input_data, session_data, derived,
        hemisphere='LH', psv_threshold=None,       # LH/RH give identical behavior cols
    )

    for col in ('lick_latency', 'outcome', 'stimulus'):
        if col in behav.columns:
            latents = latents.reset_index(drop=True)
            latents[col] = behav[col].reset_index(drop=True)
    return latents


def component_columns(latents_df, component):
    """Column names for a latent set ('across'/'within_LH'/'within_RH')."""
    prefix = COMPONENTS[component]
    # guard: z_within_LH must not match z_within_LH... only; and z_across is unique
    return [c for c in latents_df.columns if c.startswith(prefix + '_')]


# ---------------------------------------------------------------------------
# regression
# ---------------------------------------------------------------------------
def _fit_r2(X, y, cv, seed):
    """(insample_r2, cv_r2) for design matrix X predicting y."""
    insample = LinearRegression().fit(X, y).score(X, y)
    cv_r2 = np.nan
    if cv and len(y) >= cv and X.shape[1] >= 1:
        yhat = cross_val_predict(LinearRegression(), X, y, cv=cv)
        cv_r2 = r2_score(y, yhat)
    return insample, cv_r2


def predict_behavior(latents_df, target='lick_latency', cv=10, n_null=0, seed=0):
    """Regress `target` on each latent set; return a per-component dict.

    For each component: n_trials (after dropping NaN target), n_dim,
    r2_insample, r2_cv, and (if n_null>0) null_cv_mean / null_cv_p from a
    label-shuffle permutation of the target.
    """
    rng = np.random.default_rng(seed)
    y_all = latents_df[target].values.astype(float)
    mask = ~np.isnan(y_all)
    y = y_all[mask]

    out = {}
    for comp in COMPONENTS:
        cols = component_columns(latents_df, comp)
        rec = dict(component=comp, n_dim=len(cols), n_trials=int(mask.sum()),
                   r2_insample=np.nan, r2_cv=np.nan, null_cv_mean=np.nan, null_cv_p=np.nan)
        if cols and mask.sum() >= max(cv, 5):
            X = latents_df[cols].values[mask]
            rec['r2_insample'], rec['r2_cv'] = _fit_r2(X, y, cv, seed)
            if n_null > 0:
                nulls = np.empty(n_null)
                for b in range(n_null):
                    yb = rng.permutation(y)
                    _, nulls[b] = _fit_r2(X, yb, cv, seed)
                rec['null_cv_mean'] = float(np.nanmean(nulls))
                # one-sided: how often does shuffled r2 meet/exceed observed
                rec['null_cv_p'] = float((np.nansum(nulls >= rec['r2_cv']) + 1) / (n_null + 1))
        out[comp] = rec
    return out


# ---------------------------------------------------------------------------
# group table
# ---------------------------------------------------------------------------
def group_from_session(session_id):
    s = session_id[0]
    return 'bilateral' if s == 'P' else ('unilateral' if s == 'U' else 'unknown')


def build_fig6_table(session_ids, results_dir='results', data_dir='data',
                     target='lick_latency', cv=10, n_null=0, seed=0, verbose=True):
    """Long table over sessions x latent components.

    Columns: session, group, component, n_dim, n_trials,
             r2_insample, r2_cv [, null_cv_mean, null_cv_p].
    `session_ids` should include the window suffix, e.g. 'P1_w0.0-1.0'.
    """
    rows = []
    for sid in session_ids:
        if verbose:
            print('Fig6 regression:', sid)
        latents_df = load_session_latents_behavior(sid, results_dir, data_dir)
        res = predict_behavior(latents_df, target=target, cv=cv, n_null=n_null, seed=seed)
        base = sid.split('_')[0]
        for comp, rec in res.items():
            rows.append(dict(session=base, group=group_from_session(base), **rec))
    return pd.DataFrame(rows)


def across_minus_within_gap(fig6_df, r2col='r2_cv'):
    """Per-session gap = across r^2 - mean(within_LH, within_RH) r^2."""
    g = fig6_df.pivot_table(index=['session', 'group'], columns='component', values=r2col)
    within = g[['within_LH', 'within_RH']].mean(axis=1)
    gap = (g['across'] - within).rename('gap').reset_index()
    gap['across'] = g['across'].values
    gap['within_mean'] = within.values
    return gap
