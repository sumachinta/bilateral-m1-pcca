"""
fig5_angles.py
==============
Figure 5 (McDonnell, Umakantha, Williamson, Smith & Yu, Nat Commun 2026)
re-implemented for the bilateral-M1 pCCA-FA project.

Figure 5 asks: for a given hemisphere, is the *top across-area co-fluctuation
pattern* (the direction along which this hemisphere co-fluctuates with the OTHER
hemisphere) the same direction as the *top within-area co-fluctuation pattern*
(the direction along which this hemisphere co-fluctuates with itself)?

That is quantified by the principal angle theta between the two weight vectors:
    theta ~   0 deg  ->  same neurons drive both  (aligned mechanisms)
    theta ~  90 deg  ->  distinct directions       (separate mechanisms)

The math here is a faithful port of
    figures-dual-pfc-main/main_analyses/dual_pfc_funcs.get_top_angle  (across_mode='cov')
    figures-dual-pfc-main/main_analyses/dual_pfc_funcs.prinangle
so numbers are directly comparable to the paper.

Unit of analysis is a HEMISPHERE, not a session: each session yields two angles,
one for the left hemisphere (area 1) and one for the right hemisphere (area 2).
This mirrors the paper's "# of brain areas".

The functions read the fitted `.joblib` payloads saved by this project
(run_save_model.py), whose `model_params` dict already contains the loading
matrices W_1, W_2, L_1, L_2 -- so no refitting is needed.
"""

import os
import glob
import math

import numpy as np
import scipy.linalg as slin
import pandas as pd


# ---------------------------------------------------------------------------
# core geometry (ports of dual_pfc_funcs.prinangle / get_top_angle)
# ---------------------------------------------------------------------------
def prinangle(A, B):
    """Principal angles (degrees) between the column spaces of A and B.

    Direct port of dual_pfc_funcs.prinangle. Returns a list of angles ordered
    smallest-first; for the top-pattern comparison we use element [0].
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.ndim == 1:
        A = A.reshape((len(A), 1))
    if B.ndim == 1:
        B = B.reshape((len(B), 1))
    A_orth = slin.orth(A)
    B_orth = slin.orth(B)
    _, sv, _ = slin.svd(A_orth.T @ B_orth)
    # guard against tiny numerical overshoot before acos
    for i, val in enumerate(sv):
        if math.isclose(1.0, val, abs_tol=1e-5):
            sv[i] = 1.0
    return [math.acos(x) * 180.0 / math.pi for x in sv]


def top_pattern(loadings):
    """Top co-fluctuation pattern for a loading matrix.

    SVD-orthonormalize the loadings and return the leading left singular vector
    (the direction capturing the most shared variance). Matches the
    `slin.svd(W_1)[0][:, 0]` step inside get_top_angle(..., across_mode='cov').
    """
    u, _, _ = slin.svd(np.asarray(loadings, dtype=float))
    return u[:, 0]


def top_pattern_signed(loadings):
    """Top pattern with a deterministic sign convention (>=50% weights positive).

    Used only for the example-weight bar plots (paper Fig 5b); the angle itself
    is sign-invariant so this does not affect theta.
    """
    vec = top_pattern(loadings)
    n = len(vec)
    if (vec >= 0).sum() / n < 0.5:
        vec = -vec
    return vec


def _has_pattern(loadings):
    """True if a top co-fluctuation pattern is well defined.

    Requires at least one column AND non-zero energy. When the cross-validated
    within-area dimensionality is 0 (d1==0 or d2==0) the loading matrix has
    shape (n, 0): there is no within-area shared latent, so the angle is
    undefined and must NOT be reported.
    """
    L = np.asarray(loadings, dtype=float)
    return L.ndim == 2 and L.shape[1] >= 1 and np.linalg.norm(L) > 0


def top_angle_from_params(model_params):
    """Return (theta_LH, theta_RH) in degrees for one fitted session.

    model_params must contain W_1, W_2 (across-area loadings) and
    L_1, L_2 (within-area loadings), as saved in each results/*.joblib.

    A hemisphere with no across-area latent (W empty) or no within-area latent
    (L empty; i.e. cross-validated d1 or d2 == 0) has an undefined angle and
    returns np.nan for that hemisphere.
    """
    W_1, W_2 = model_params['W_1'], model_params['W_2']
    L_1, L_2 = model_params['L_1'], model_params['L_2']

    theta_lh = (prinangle(top_pattern(W_1), top_pattern(L_1))[0]
                if _has_pattern(W_1) and _has_pattern(L_1) else np.nan)
    theta_rh = (prinangle(top_pattern(W_2), top_pattern(L_2))[0]
                if _has_pattern(W_2) and _has_pattern(L_2) else np.nan)
    return theta_lh, theta_rh


# ---------------------------------------------------------------------------
# session bookkeeping
# ---------------------------------------------------------------------------
def session_id_from_path(path):
    """'.../P1_w0.0-1.0.joblib' -> 'P1'."""
    base = os.path.basename(path)
    return base.split('_')[0].replace('.joblib', '')


def group_from_session(session_id):
    """P* -> bilateral, U* -> unilateral."""
    if session_id.startswith('P'):
        return 'bilateral'
    if session_id.startswith('U'):
        return 'unilateral'
    return 'unknown'


# ---------------------------------------------------------------------------
# table builders
# ---------------------------------------------------------------------------
def build_angle_table(results_dir, window_suffix='_w0.0-1.0'):
    """Long-format table: one row per (session, hemisphere).

    Columns: session, group, hemisphere ('LH'/'RH'), area (1/2), theta (deg),
             n_neurons, d, d1, d2, file.
    """
    import joblib

    pattern = os.path.join(results_dir, '*{}.joblib'.format(window_suffix))
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            'No fits matching {} in {}'.format(pattern, results_dir))

    rows = []
    for f in files:
        sid = session_id_from_path(f)
        grp = group_from_session(sid)
        mp = joblib.load(f)['model_params']
        theta_lh, theta_rh = top_angle_from_params(mp)
        rows.append(dict(session=sid, group=grp, hemisphere='LH', area=1,
                         theta=theta_lh, n_neurons=mp['W_1'].shape[0],
                         d=mp['d'], d1=mp['d1'], d2=mp['d2'], file=os.path.basename(f)))
        rows.append(dict(session=sid, group=grp, hemisphere='RH', area=2,
                         theta=theta_rh, n_neurons=mp['W_2'].shape[0],
                         d=mp['d'], d1=mp['d1'], d2=mp['d2'], file=os.path.basename(f)))

    df = pd.DataFrame(rows)
    return df.sort_values(['group', 'session', 'hemisphere']).reset_index(drop=True)


def session_level_table(angle_df):
    """Collapse the two hemispheres to one mean theta per session.

    Useful for a statistical test that avoids treating the two hemispheres of
    the same animal as independent samples.
    """
    g = (angle_df.groupby(['session', 'group'])['theta']
         .mean().reset_index().rename(columns={'theta': 'theta_mean'}))
    return g.sort_values(['group', 'session']).reset_index(drop=True)


# ===========================================================================
# Supplementary Figure 6  --  how much shared variance each pattern captures
# ("eigenspectrum"). Justifies analysing only the TOP co-fluctuation pattern.
# Ports of plot_figureS6.ipynb.
# ===========================================================================
def _canonical_directions(model_params):
    """Across-area canonical directions for both areas (port of the math in
    pcca_fa_mdl.get_canonical_directions). Reimplemented here so the module has
    no dependency on the model class. Returns (dirs_x1, dirs_x2, d)."""
    W_1, W_2 = model_params['W_1'], model_params['W_2']
    L_1, L_2 = model_params['L_1'], model_params['L_2']
    psi_1, psi_2 = model_params['psi_1'], model_params['psi_2']
    d = int(model_params['d'])

    cov1 = W_1 @ W_1.T + L_1 @ L_1.T + np.diag(psi_1)
    cov2 = W_2 @ W_2.T + L_2 @ L_2.T + np.diag(psi_2)
    crosscov = W_1 @ W_2.T
    inv1 = slin.inv(slin.sqrtm(cov1))
    inv2 = slin.inv(slin.sqrtm(cov2))
    K = inv1 @ crosscov @ inv2
    u, _, vt = slin.svd(K)
    dirs_x1 = np.real(inv1 @ u[:, :d])
    dirs_x2 = np.real(inv2 @ vt[:d, :].T)
    return dirs_x1, dirs_x2, d


def across_variance_spectrum(model_params):
    """Proportion of ACROSS-area shared variance captured by each across-area
    pattern (canonical direction), per area. Each returned array sums to 1 and
    is ordered by canonical correlation. Port of plot_figureS6 cell 2."""
    dirs_x1, dirs_x2, d = _canonical_directions(model_params)
    out = []
    for dirs, W in ((dirs_x1, model_params['W_1']), (dirs_x2, model_params['W_2'])):
        nd = dirs / slin.norm(dirs, axis=0)
        v = np.array([nd[:, j].T @ (W @ W.T) @ nd[:, j] for j in range(d)])
        out.append(v / v.sum())
    return out[0], out[1]


def within_variance_spectrum(model_params):
    """Proportion of WITHIN-area shared variance captured by each within-area
    pattern, per area. Ordered by shared variance; empty array if that area's
    within-dim is 0. Port of plot_figureS6 cell 3."""
    out = []
    for L, dd in ((model_params['L_1'], int(model_params['d1'])),
                  (model_params['L_2'], int(model_params['d2']))):
        if dd == 0 or np.linalg.norm(L) == 0:
            out.append(np.array([]))
            continue
        vals = slin.svdvals(L @ L.T)
        total = np.trace(L @ L.T)
        out.append(vals[:dd] / total)
    return out[0], out[1]


def build_variance_spectrum_table(results_dir, window_suffix='_w0.0-1.0'):
    """Long table: session, group, area, hemisphere, component ('across'/
    'within'), rank (1-based pattern index), prop_sv (fraction, 0..1)."""
    import joblib

    files = sorted(glob.glob(os.path.join(results_dir, '*{}.joblib'.format(window_suffix))))
    rows = []
    for f in files:
        sid = session_id_from_path(f)
        grp = group_from_session(sid)
        mp = joblib.load(f)['model_params']
        ax1, ax2 = across_variance_spectrum(mp)
        wn1, wn2 = within_variance_spectrum(mp)
        for hemi, area, ax_sp, wn_sp in (('LH', 1, ax1, wn1), ('RH', 2, ax2, wn2)):
            for r, v in enumerate(ax_sp, start=1):
                rows.append(dict(session=sid, group=grp, hemisphere=hemi, area=area,
                                 component='across', rank=r, prop_sv=float(v)))
            for r, v in enumerate(wn_sp, start=1):
                rows.append(dict(session=sid, group=grp, hemisphere=hemi, area=area,
                                 component='within', rank=r, prop_sv=float(v)))
    return pd.DataFrame(rows)


# ===========================================================================
# Supplementary Figure 7  --  weight diversity of the top co-fluctuation
# pattern. % of top-pattern weights that are positive:  ~50% = diverse
# (mixed sign, structured);  ~100% = low diversity (global on/off).
# Ports of get_top_vec / plot_figureS7.ipynb.
# ===========================================================================
def top_vec_pct_positive(loadings):
    """% of the top pattern's weights that share the majority sign (>=50).

    Mirrors dual_pfc_funcs.get_top_vec: orthonormalize, take leading vector,
    flip so most weights are positive, return that positive fraction * 100.
    Returns np.nan when the pattern is undefined (empty / zero loadings)."""
    if not _has_pattern(loadings):
        return np.nan
    vec = top_pattern(loadings)
    n = len(vec)
    pct_pos = (vec >= 0).sum() / n
    if pct_pos < 0.5:
        pct_pos = 1 - pct_pos
    return pct_pos * 100.0


def build_weight_diversity_table(results_dir, window_suffix='_w0.0-1.0'):
    """Long table: session, group, hemisphere, pct_pos_across, pct_pos_within.

    pct_pos_* is the % of the top across-/within-area pattern's weights that
    are the majority sign (the paper's weight-diversity readout)."""
    import joblib

    files = sorted(glob.glob(os.path.join(results_dir, '*{}.joblib'.format(window_suffix))))
    rows = []
    for f in files:
        sid = session_id_from_path(f)
        grp = group_from_session(sid)
        mp = joblib.load(f)['model_params']
        for hemi, W, L in (('LH', mp['W_1'], mp['L_1']), ('RH', mp['W_2'], mp['L_2'])):
            rows.append(dict(session=sid, group=grp, hemisphere=hemi,
                             pct_pos_across=top_vec_pct_positive(W),
                             pct_pos_within=top_vec_pct_positive(L)))
    df = pd.DataFrame(rows)
    return df.sort_values(['group', 'session', 'hemisphere']).reset_index(drop=True)
