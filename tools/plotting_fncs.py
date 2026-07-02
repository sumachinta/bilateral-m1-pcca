import matplotlib.pyplot as plt

def plot_trial_variable(session_data, trial_start_frames, var_name='run_speed', trial_idx=1, xlim=None, ylim=None):
    """
    Plot a variable for a single trial.

    Parameters
    ----------
    session_data : dict-like
        Must contain:
        - 'absolute_frames'
        - 'time'
        - var_name
    trial_start_frames : array-like
        Start frame of each trial.
    var_name : str, optional
        Variable to plot (e.g. 'hits', 'misses', 'run_speed').
    trial_idx : int, optional
        Trial number (0-based indexing).
    """

    frame_start = trial_start_frames[trial_idx]
    frame_end = trial_start_frames[trial_idx + 1] - 1 if trial_idx + 1 < len(trial_start_frames) else len(session_data['absolute_frames'])

    mask = (
        (session_data['absolute_frames'] >= frame_start)
        & (session_data['absolute_frames'] <= frame_end)
    )

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(
        session_data['time'][mask],
        session_data[var_name][mask],
        marker='_',
        markersize=1, 
        alpha=0.5,
    )

    ax.set_xlabel('Time (s)')
    ax.set_ylabel(var_name)
    ax.set_title(f'{var_name} | Trial {trial_idx}')

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    plt.tight_layout()
    plt.show()



import matplotlib.gridspec as gridspec
import numpy as np


def plot_session_metrics(metrics, summary, session_id=None):
    """
    4-panel summary figure for one session's pCCA-FA results.

    Panel A — Variance decomposition: stacked bar per hemisphere showing
               what fraction of spike count variance is across-, within-,
               or independently explained. This is the headline result.

    Panel B — Per-neuron %sv scatter: one dot per neuron, x = across %sv,
               y = within %sv. Shows whether the across/within split is
               uniform across neurons or driven by a subset.

    Panel C — Primary metric: the normalised within_ratio for this session,
               shown as a gauge bar so you can immediately see where it sits
               between 0 (all across) and 1 (all within).

    Panel D — Cross-validated canonical correlations: how correlated are the
               across-hemisphere latent variables between the two hemispheres,
               evaluated on held-out folds. A sanity check that the across-
               hemisphere structure the model found is real.
    """
    psv = metrics['psv']
    fig = plt.figure(figsize=(7, 7))
    fig.suptitle(f"pCCA-FA metrics — session {session_id or ''}",
                 fontsize=14, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

    # colour palette
    c_across = '#D85A30'   # coral — across-hemisphere
    c_within = '#185FA5'   # blue  — within-hemisphere
    c_indep  = '#888780'   # gray  — independent noise

    # ── Panel A: stacked variance decomposition ────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])

    hemis      = ['LH', 'RH']
    avg_W      = [psv['avg_psv_W_1'], psv['avg_psv_W_2']]
    avg_L      = [psv['avg_psv_L_1'], psv['avg_psv_L_2']]
    avg_ind    = [np.mean(psv['ind_var_x1']), np.mean(psv['ind_var_x2'])]

    x = np.arange(len(hemis))
    w = 0.45

    bar_across = ax_a.bar(x, avg_W, width=w, color=c_across, label='Across-hemisphere')
    bar_within = ax_a.bar(x, avg_L, width=w, bottom=avg_W, color=c_within,
                          label='Within-hemisphere')
    bar_indep  = ax_a.bar(x, avg_ind, width=w,
                          bottom=[avg_W[i] + avg_L[i] for i in range(2)],
                          color=c_indep, alpha=0.5, label='Independent')

    # value labels inside bars
    for i in range(2):
        ax_a.text(x[i], avg_W[i] / 2,
                  f'{avg_W[i]:.1f}%', ha='center', va='center',
                  fontsize=9, color='white', fontweight='bold')
        ax_a.text(x[i], avg_W[i] + avg_L[i] / 2,
                  f'{avg_L[i]:.1f}%', ha='center', va='center',
                  fontsize=9, color='white', fontweight='bold')

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(hemis, fontsize=11)
    ax_a.set_ylabel('% spike count variance', fontsize=10)
    ax_a.set_title('A  Variance decomposition per hemisphere', fontsize=10,
                   loc='left', fontweight='bold')
    ax_a.set_ylim(0, 105)
    ax_a.legend(fontsize=8, loc='upper right')
    ax_a.spines[['top', 'right']].set_visible(False)

    # ── Panel B: per-neuron %sv scatter ───────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])

    # LH neurons
    ax_b.scatter(psv['psv_W_1'], psv['psv_L_1'],
                 color=c_across, alpha=0.5, s=20, label='LH neurons',
                 edgecolors='none')
    # RH neurons
    ax_b.scatter(psv['psv_W_2'], psv['psv_L_2'],
                 color=c_within, alpha=0.5, s=20, label='RH neurons',
                 edgecolors='none', marker='s')

    # diagonal: equal across and within
    lim = max(ax_b.get_xlim()[1], ax_b.get_ylim()[1])
    ax_b.plot([0, lim], [0, lim], '--', color='#B4B2A9', lw=1,
              label='across = within')

    # mean crosshairs
    ax_b.axvline(psv['avg_psv_W_total'], color='#B4B2A9', lw=1, ls=':')
    ax_b.axhline(psv['avg_psv_L_total'], color='#B4B2A9', lw=1, ls=':')

    ax_b.set_xlabel('Across-hemisphere %sv (per neuron)', fontsize=10)
    ax_b.set_ylabel('Within-hemisphere %sv (per neuron)', fontsize=10)
    ax_b.set_title('B  Per-neuron %sv', fontsize=10,
                   loc='left', fontweight='bold')
    ax_b.legend(fontsize=8)
    ax_b.spines[['top', 'right']].set_visible(False)

    # ── Panel C: primary metric gauge ─────────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])

    within_ratio = summary['within_ratio']

    # background track
    ax_c.barh(0, 1.0, height=0.5, color='#F1EFE8', edgecolor='#D3D1C7')
    # filled portion
    ax_c.barh(0, within_ratio, height=0.5,
              color=c_within if within_ratio > 0.5 else c_across)

    ax_c.set_xlim(0, 1)
    ax_c.set_ylim(-0.6, 0.6)
    ax_c.set_yticks([])
    ax_c.set_xlabel('within_ratio  =  within / (within + across)', fontsize=10)
    ax_c.axvline(0.5, color='#888780', lw=1, ls='--')
    ax_c.text(0.02, 0.32, 'all across →', fontsize=8, color='#888780',
              transform=ax_c.transAxes)
    ax_c.text(0.75, 0.32, '← all within', fontsize=8, color='#888780',
              transform=ax_c.transAxes)
    ax_c.text(within_ratio, 0,
              f'  {within_ratio:.3f}', va='center', fontsize=12,
              fontweight='bold', color='#2C2C2A')
    ax_c.set_title('C  Primary metric (this session)', fontsize=10,
                   loc='left', fontweight='bold')
    ax_c.spines[['top', 'right', 'left']].set_visible(False)

    # annotation box with selected dims
    dim_text = (f"d={summary['d']}  d1={summary['d1']}  d2={summary['d2']}\n"
                f"avg W={summary['avg_psv_W']:.1f}%   "
                f"avg L={summary['avg_psv_L']:.1f}%")
    ax_c.text(0.5, -0.38, dim_text, ha='center', va='center',
              fontsize=9, color='#5F5E5A',
              transform=ax_c.transAxes)

    # ── Panel D: cross-validated canonical correlations ───────────────────
    ax_d = fig.add_subplot(gs[1, 1])

    if 'cv_rho' in metrics and len(metrics['cv_rho']) > 0:
        cv_rho = metrics['cv_rho']
        rho    = metrics['rho']        # model (non-CV) canonical corrs
        dims   = np.arange(1, len(cv_rho) + 1)

        ax_d.bar(dims, cv_rho, color=c_across, alpha=0.85,
                 label='CV canonical corr.')
        ax_d.plot(dims, rho[:len(cv_rho)], 'o--', color='#444441',
                  ms=5, lw=1, label='Model canonical corr.')
        ax_d.axhline(0, color='#B4B2A9', lw=0.8)
        ax_d.set_xlabel('Across-hemisphere latent dimension', fontsize=10)
        ax_d.set_ylabel('Canonical correlation (ρ)', fontsize=10)
        ax_d.set_xticks(dims)
        ax_d.set_ylim(-0.05, 1.05)
        ax_d.legend(fontsize=8)
    else:
        ax_d.text(0.5, 0.5, 'cv_rho not available\n(crossvalidate() required)',
                  ha='center', va='center', transform=ax_d.transAxes,
                  fontsize=10, color='#888780')

    ax_d.set_title('D  Cross-validated canonical correlations', fontsize=10,
                   loc='left', fontweight='bold')
    ax_d.spines[['top', 'right']].set_visible(False)

    plt.savefig(f'figures/session_{session_id or "metrics"}.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    return fig


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches


def plot_psv_partitioning(metrics, session_id, order_by='across'):
    """
    Plot per-neuron variance partitioning and across-vs-within scatter
    for both hemispheres.

    Parameters
    ----------
    metrics    : dict returned by model.compute_metrics()
    session_id : str, used in the figure title (e.g. 'U02')
    order_by   : str, how to sort neurons in the bar plots.
                 One of 'across' | 'within' | 'independent'

    Returns
    -------
    fig : matplotlib Figure — display with plt.show(), save with fig.savefig()

    Example
    -------
    fig = plot_psv_partitioning(metrics, session_id='U02', order_by='across')
    plt.show()
    fig.savefig('psv_U02.png', dpi=150, bbox_inches='tight')
    """
    teal   = '#0F6E56'   # across-hemisphere
    coral  = '#D85A30'   # within-hemisphere
    silver = '#C4CDD6'   # independent

    # ── unpack psv ────────────────────────────────────────────────────────────
    psv = metrics['psv']

    psv_W_1   = psv['psv_W_1'];    psv_W_2   = psv['psv_W_2']
    psv_L_1   = psv['psv_L_1'];    psv_L_2   = psv['psv_L_2']
    ind_var_1 = psv['ind_var_x1']; ind_var_2 = psv['ind_var_x2']
    avg_W_1   = psv['avg_psv_W_1']; avg_L_1  = psv['avg_psv_L_1']
    avg_W_2   = psv['avg_psv_W_2']; avg_L_2  = psv['avg_psv_L_2']

    n1 = len(psv_W_1)
    n2 = len(psv_W_2)

    # ── sort order ────────────────────────────────────────────────────────────
    sort_options_1 = {'across': psv_W_1, 'within': psv_L_1, 'independent': ind_var_1}
    sort_options_2 = {'across': psv_W_2, 'within': psv_L_2, 'independent': ind_var_2}

    if order_by not in sort_options_1:
        raise ValueError(f"order_by must be one of {list(sort_options_1.keys())}, got '{order_by}'")

    ord1 = np.argsort(sort_options_1[order_by])[::-1]
    ord2 = np.argsort(sort_options_2[order_by])[::-1]

    # ── figure ────────────────────────────────────────────────────────────────
    fig_h = max(n1, n2) * 0.1 + 2
    fig   = plt.figure(figsize=(10, fig_h))
    gs    = gridspec.GridSpec(2, 2, figure=fig,width_ratios=[2, 1.4])

    ax_bar_rh  = fig.add_subplot(gs[0, 0])
    ax_bar_lh  = fig.add_subplot(gs[1, 0])
    ax_scat_rh = fig.add_subplot(gs[0, 1])
    ax_scat_lh = fig.add_subplot(gs[1, 1])

    # ── bar plots ─────────────────────────────────────────────────────────────
    for ax, psv_W, psv_L, ind_var, order, avg_W, avg_L, hemi in [
        (ax_bar_rh, psv_W_2, psv_L_2, ind_var_2, ord2, avg_W_2, avg_L_2, 'RH'),
        (ax_bar_lh, psv_W_1, psv_L_1, ind_var_1, ord1, avg_W_1, avg_L_1, 'LH'),
    ]:
        n  = len(psv_W)
        y  = np.arange(n)
        sW = psv_W[order]; sL = psv_L[order]; sI = ind_var[order]

        ax.barh(y, sW, height=0.8, color=teal,   zorder=3)
        ax.barh(y, sL, height=0.8, color=coral,  left=sW,      zorder=3)
        ax.barh(y, sI, height=0.8, color=silver, left=sW + sL, zorder=3)

        ax.set_xlim(0, 100)
        ax.set_ylim(-0.5, n - 0.5)
        ax.invert_yaxis()
        ax.set_yticks([])
        ax.set_xlabel('% variance', fontsize=9)
        ax.set_ylabel(f'{hemi} neurons', fontsize=9, labelpad=5)
        ax.set_title(f'partitioning of variance — {hemi}', fontsize=10, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='x', linewidth=0.3, alpha=0.5, zorder=0)

        ax.text(0.98, 0.10, f'across %sv = {avg_W:.2f}%',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=8.5, fontweight='bold', color=teal)
        ax.text(0.98, 0.04, f'within %sv  = {avg_L:.2f}%',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=8.5, fontweight='bold', color=coral)

    # ── scatter plots ─────────────────────────────────────────────────────────
    for ax, psv_W, psv_L, color, hemi in [
        (ax_scat_rh, psv_W_2, psv_L_2, teal,  'RH'),
        (ax_scat_lh, psv_W_1, psv_L_1, coral, 'LH'),
    ]:
        ax.scatter(psv_L, psv_W, color=color, alpha=0.55, s=18,
                   edgecolors='white', linewidths=0.4, zorder=3)

        lim = max(psv_W.max(), psv_L.max()) * 1.15
        ax.plot([0, lim], [0, lim], color='#555', linestyle='--',
                linewidth=1, zorder=2)
        ax.text(lim * 0.08, lim * 0.88, 'across > within',
                fontsize=7, color='#333', style='italic')

        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel('within-area %sv', fontsize=9, color=coral)
        ax.set_ylabel('across-area %sv', fontsize=9, color=teal)
        ax.tick_params(axis='x', labelsize=7, colors=coral)
        ax.tick_params(axis='y', labelsize=7, colors=teal)
        ax.spines['bottom'].set_color(coral)
        ax.spines['left'].set_color(teal)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title(f'{hemi} — across vs within %sv', fontsize=10, fontweight='bold')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(linewidth=0.3, alpha=0.4, zorder=0)

    # ── legend + title ────────────────────────────────────────────────────────
    fig.legend(
        handles=[
            mpatches.Patch(color=teal,   label='across-hemisphere'),
            mpatches.Patch(color=coral,  label='within-hemisphere'),
            mpatches.Patch(color=silver, label='independent'),
        ],
        loc='lower left', bbox_to_anchor=(0.02, -0.01),
        ncol=3, fontsize=9, frameon=False,
    )

    fig.suptitle(
        f'Partitioning of variance — {session_id}  (ordered by: {order_by})',
        fontsize=10, fontweight='bold')
    fig.tight_layout()

    return fig


import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import joblib


def plot_dshared_scatter(window_suffix, results_dir='results'):
    """
    Load all sessions matching window_suffix and plot a 2×4 grid of dshared
    scatter plots — rows separate P (bilateral) and U (unilateral) groups.

    Layout
    ------
    Col 0 : across vs within dshared — LH
    Col 1 : across vs within dshared — RH
    Col 2 : LH vs RH across-area dshared   (x=LH, y=RH)
    Col 3 : LH vs RH within-area dshared   (x=LH, y=RH)

    Parameters
    ----------
    window_suffix : str   e.g. '_w0.0-1.0'
    results_dir   : str or Path

    Returns
    -------
    fig : matplotlib Figure

    Example
    -------
    fig = plot_dshared_scatter('_w0.0-1.0')
    plt.show()
    fig.savefig('dshared_w0.0-1.0.png', dpi=150, bbox_inches='tight')
    """
    teal  = '#0F6E56'   # unilateral (U) / across-hemisphere
    coral = '#D85A30'   # bilateral  (P) / within-hemisphere

    results_dir = Path(results_dir)
    files = sorted(results_dir.glob(f'*{window_suffix}.joblib'))
    if not files:
        raise FileNotFoundError(
            f"No files found matching '*{window_suffix}.joblib' in {results_dir}"
        )

    # ── load ─────────────────────────────────────────────────────────────────
    groups = {
        'P': {'W1': [], 'L1': [], 'W2': [], 'L2': [], 'ids': []},
        'U': {'W1': [], 'L1': [], 'W2': [], 'L2': [], 'ids': []},
    }

    for f in files:
        session_id = f.stem.replace(window_suffix, '')
        group      = session_id[0].upper()
        if group not in groups:
            continue

        payload = joblib.load(f)
        ds      = payload['metrics']['dshared']

        groups[group]['W1'].append(ds['dshared_W_1'])
        groups[group]['L1'].append(ds['dshared_L_1'])
        groups[group]['W2'].append(ds['dshared_W_2'])
        groups[group]['L2'].append(ds['dshared_L_2'])
        groups[group]['ids'].append(session_id)

    # ── global axis limit (all panels share same scale) ───────────────────────
    all_vals = [v for g in groups.values()
                  for key in ('W1', 'L1', 'W2', 'L2')
                  for v in g[key]]
    lim = max(all_vals) * 1.25 if all_vals else 10

    # ── figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.subplots_adjust(hspace=0.4, wspace=0.35)

    group_colors = {'P': coral, 'U': teal}
    group_names  = {'P': 'Bilateral (P)', 'U': 'Unilateral (U)'}

    # panel definitions:
    # (col, x_key, y_key, title_suffix, x_label, y_label, diag_label,
    #  x_spine_color, y_spine_color)
    col_specs = [
        (0, 'L1', 'W1',
         'LH  —  across vs within',
         'within dshared — LH', 'across dshared — LH',
         'across > within',
         coral, teal),
        (1, 'L2', 'W2',
         'RH  —  across vs within',
         'within dshared — RH', 'across dshared — RH',
         'across > within',
         coral, teal),
        (2, 'W1', 'W2',
         'across dshared  —  LH vs RH',
         'across dshared — LH', 'across dshared — RH',
         'RH > LH',
         teal, teal),
        (3, 'L1', 'L2',
         'within dshared  —  LH vs RH',
         'within dshared — LH', 'within dshared — RH',
         'RH > LH',
         coral, coral),
    ]

    for row, grp in enumerate(('P', 'U')):
        color = group_colors[grp]

        for col, xkey, ykey, title_sfx, xlabel, ylabel, diag_label, \
                x_spine_col, y_spine_col in col_specs:

            ax  = axes[row, col]
            xs  = np.array(groups[grp][xkey])
            ys  = np.array(groups[grp][ykey])
            ids = groups[grp]['ids']

            if len(xs):
                ax.scatter(xs, ys, s=60, alpha=0.85, color=color,
                           marker='o', edgecolors='white',
                           linewidths=0.5, zorder=4)

                for x, y, sid in zip(xs, ys, ids):
                    ax.text(x + 0.1, y + 0.1, sid, fontsize=7,
                            color=color, va='bottom')

            # y = x diagonal
            ax.plot([0, lim], [0, lim], color='#555', linestyle='--',
                    linewidth=1, zorder=2)
            ax.text(lim * 0.05, lim * 0.90, diag_label,
                    fontsize=7, color='#333', style='italic')

            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
            ax.set_aspect('equal', adjustable='box')
            ax.set_title(f'{group_names[grp]}\n{title_sfx}',
                         fontsize=9, fontweight='bold', color=color)
            ax.set_xlabel(xlabel, fontsize=8, color=x_spine_col)
            ax.set_ylabel(ylabel, fontsize=8, color=y_spine_col)
            ax.tick_params(axis='x', colors=x_spine_col, labelsize=7)
            ax.tick_params(axis='y', colors=y_spine_col, labelsize=7)
            ax.spines['bottom'].set_color(x_spine_col)
            ax.spines['left'].set_color(y_spine_col)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(linewidth=0.3, alpha=0.4, zorder=0)

    fig.suptitle(
        f'Shared dimensionality (dshared)  —  window: {window_suffix}',
        fontsize=12, fontweight='bold',
    )

    return fig


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import joblib


def plot_psv_pooled(window_suffix, results_dir='results'):
    """
    Pool all sessions matching window_suffix, split into P (bilateral) and
    U (unilateral) groups, and return one figure per group.

    Each figure has 2 rows (LH top, RH bottom) × 3 columns:
      col 0 — histogram of across + within %sv  (x: 0–100%)
      col 1 — histogram of independent %sv      (x: 0–100%)
      col 2 — per-neuron scatter across vs within %sv, colored by session

    Parameters
    ----------
    window_suffix : str   e.g. '_w0.0-1.0'
    results_dir   : str or Path

    Returns
    -------
    fig_P, fig_U : matplotlib Figures  (bilateral, unilateral)
                   Either may be None if no sessions of that type were found.

    Example
    -------
    fig_P, fig_U = plot_psv_pooled('_w0.0-1.0')
    fig_P.savefig('psv_pooled_bilateral.png',   dpi=150, bbox_inches='tight')
    fig_U.savefig('psv_pooled_unilateral.png',  dpi=150, bbox_inches='tight')
    plt.show()
    """
    teal   = '#0F6E56'
    coral  = '#D85A30'
    silver = '#C4CDD6'

    results_dir = Path(results_dir)
    files = sorted(results_dir.glob(f'*{window_suffix}.joblib'))
    if not files:
        raise FileNotFoundError(
            f"No files found matching '*{window_suffix}.joblib' in {results_dir}"
        )

    # ── load and group ────────────────────────────────────────────────────────
    groups = {'P': {}, 'U': {}}

    for f in files:
        session_id = f.stem.replace(window_suffix, '')
        group      = session_id[0].upper()
        if group not in groups:
            continue

        payload = joblib.load(f)
        psv     = payload['metrics']['psv']

        groups[group][session_id] = {
            'psv_W_1':   psv['psv_W_1'],
            'psv_L_1':   psv['psv_L_1'],
            'ind_var_1': psv['ind_var_x1'],
            'psv_W_2':   psv['psv_W_2'],
            'psv_L_2':   psv['psv_L_2'],
            'ind_var_2': psv['ind_var_x2'],
        }

    # ── session color palette (tab10, up to 10 sessions) ─────────────────────
    cmap = plt.cm.tab10

    group_labels = {'P': 'Bilateral (P)', 'U': 'Unilateral (U)'}
    figs = {'P': None, 'U': None}

    for grp in ('P', 'U'):
        sessions = groups[grp]
        if not sessions:
            continue

        session_ids    = list(sessions.keys())
        session_colors = {sid: cmap(i % 10) for i, sid in enumerate(session_ids)}

        fig, axes = plt.subplots(2, 3, figsize=(13, 7))

        hemis = [
            ('psv_W_1', 'psv_L_1', 'ind_var_1', 'LH'),
            ('psv_W_2', 'psv_L_2', 'ind_var_2', 'RH'),
        ]

        for row, (wkey, lkey, ikey, hemi) in enumerate(hemis):

            ax_sv  = axes[row, 0]   # across + within histogram
            ax_ind = axes[row, 1]   # independent histogram
            ax_sc  = axes[row, 2]   # scatter

            # pool neurons across all sessions for histograms
            all_W = np.concatenate([s[wkey] for s in sessions.values()])
            all_L = np.concatenate([s[lkey] for s in sessions.values()])
            all_I = np.concatenate([s[ikey] for s in sessions.values()])
            n_neurons = len(all_W)

            # ── histogram: across + within (zoomed 0–100%) ─────────────────
            bins_sv = np.linspace(0, 100, 28)

            ax_sv.hist(all_W, bins=bins_sv, color=teal,  alpha=0.65,
                       zorder=3, label='across')
            ax_sv.hist(all_L, bins=bins_sv, color=coral, alpha=0.65,
                       zorder=3, label='within')

            # mean lines
            ax_sv.axvline(all_W.mean(), color=teal,  linestyle='--',
                          linewidth=1.2, zorder=4)
            ax_sv.axvline(all_L.mean(), color=coral, linestyle='--',
                          linewidth=1.2, zorder=4)

            # mean annotations
            ax_sv.text(0.98, 0.97,
                       f'across mean = {all_W.mean():.2f}%',
                       transform=ax_sv.transAxes, ha='right', va='top',
                       fontsize=8, color=teal, fontweight='bold')
            ax_sv.text(0.98, 0.90,
                       f'within mean  = {all_L.mean():.2f}%',
                       transform=ax_sv.transAxes, ha='right', va='top',
                       fontsize=8, color=coral, fontweight='bold')

            ax_sv.set_xlim(0, 100)
            ax_sv.set_xlabel('% shared variance', fontsize=9)
            ax_sv.set_ylabel('neuron count', fontsize=9)
            ax_sv.set_title(f'{hemi} — across & within %sv\n(n = {n_neurons} neurons)',
                            fontsize=9.5, fontweight='bold')
            ax_sv.spines['top'].set_visible(False)
            ax_sv.spines['right'].set_visible(False)
            ax_sv.grid(axis='y', linewidth=0.3, alpha=0.5, zorder=0)

            # ── histogram: independent (0–100%) ───────────────────────────
            bins_ind = np.linspace(0, 100, 28)

            ax_ind.hist(all_I, bins=bins_ind, color=silver, alpha=0.8, zorder=3)
            ax_ind.axvline(all_I.mean(), color='#555', linestyle='--',
                           linewidth=1.2, zorder=4)
            ax_ind.text(0.98, 0.97,
                        f'mean = {all_I.mean():.2f}%',
                        transform=ax_ind.transAxes, ha='right', va='top',
                        fontsize=8, color='#444', fontweight='bold')

            ax_ind.set_xlim(0, 100)
            ax_ind.set_xlabel('% variance', fontsize=9)
            ax_ind.set_ylabel('neuron count', fontsize=9)
            ax_ind.set_title(f'{hemi} — independent %sv',
                             fontsize=9.5, fontweight='bold')
            ax_ind.spines['top'].set_visible(False)
            ax_ind.spines['right'].set_visible(False)
            ax_ind.grid(axis='y', linewidth=0.3, alpha=0.5, zorder=0)

            # ── scatter: per session colored ───────────────────────────────
            lim_vals = []
            for sid in session_ids:
                s  = sessions[sid]
                xs = s[lkey]; ys = s[wkey]
                ax_sc.scatter(xs, ys,
                              color=session_colors[sid],
                              alpha=0.45, s=10,
                              label=sid, zorder=3,
                              edgecolors='none')
                lim_vals.extend(xs.tolist() + ys.tolist())

            # use 99th percentile to avoid outliers stretching the axes
            lim = np.percentile(lim_vals, 99) * 1.2 if lim_vals else 40

            ax_sc.plot([0, lim], [0, lim], color='#555', linestyle='--',
                       linewidth=1, zorder=2)
            ax_sc.text(lim * 0.05, lim * 0.90, 'across > within',
                       fontsize=7, color='#333', style='italic')

            ax_sc.set_xlim(0, lim); ax_sc.set_ylim(0, lim)
            ax_sc.set_xlabel('within-area %sv', fontsize=9, color=coral)
            ax_sc.set_ylabel('across-area %sv', fontsize=9, color=teal)
            ax_sc.tick_params(axis='x', colors=coral, labelsize=7)
            ax_sc.tick_params(axis='y', colors=teal,  labelsize=7)
            ax_sc.spines['bottom'].set_color(coral)
            ax_sc.spines['left'].set_color(teal)
            ax_sc.spines['top'].set_visible(False)
            ax_sc.spines['right'].set_visible(False)
            ax_sc.set_aspect('equal', adjustable='box')
            ax_sc.set_title(f'{hemi} — across vs within %sv',
                            fontsize=9.5, fontweight='bold')
            ax_sc.legend(fontsize=7, frameon=False,
                         ncol=2, loc='lower right',
                         markerscale=1.5)
            ax_sc.grid(linewidth=0.3, alpha=0.4, zorder=0)

        # ── shared legend for histogram colors ────────────────────────────
        fig.legend(
            handles=[
                mpatches.Patch(color=teal,   alpha=0.75, label='across-hemisphere'),
                mpatches.Patch(color=coral,  alpha=0.75, label='within-hemisphere'),
                mpatches.Patch(color=silver, alpha=0.80, label='independent'),
            ],
            loc='lower center', ncol=3, fontsize=9,
            bbox_to_anchor=(0.38, -0.02), frameon=False,
        )

        fig.suptitle(
            f'{group_labels[grp]} sessions — PSV distributions'
            f'  (window: {window_suffix},  {len(session_ids)} sessions)',
            fontsize=11, fontweight='bold',
        )
        fig.tight_layout()
        figs[grp] = fig

    return figs['P'], figs['U']


import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import joblib


def plot_psv_scatter(window_suffix, results_dir='results'):
    """
    Load all sessions matching window_suffix and plot a 2×4 grid of
    average %sv scatter plots — rows separate P (bilateral) and U
    (unilateral) groups, one point per session.

    Layout
    ------
    Col 0 : across %sv vs within %sv  — LH   (x=within, y=across)
    Col 1 : across %sv vs within %sv  — RH   (x=within, y=across)
    Col 2 : LH vs RH across %sv              (x=LH, y=RH)
    Col 3 : LH vs RH within %sv              (x=LH, y=RH)

    Rows  : row 0 = Bilateral (P),  row 1 = Unilateral (U)

    Parameters
    ----------
    window_suffix : str   e.g. '_w0.0-1.0'
    results_dir   : str or Path

    Returns
    -------
    fig : matplotlib Figure

    Example
    -------
    fig = plot_psv_scatter('_w0.0-1.0')
    plt.show()
    fig.savefig('figures/psv_scatter_w0.0-1.0.png', dpi=150, bbox_inches='tight')
    """
    teal  = '#0F6E56'
    coral = '#D85A30'

    results_dir = Path(results_dir)
    files = sorted(results_dir.glob(f'*{window_suffix}.joblib'))
    if not files:
        raise FileNotFoundError(
            f"No files found matching '*{window_suffix}.joblib' in {results_dir}"
        )

    # ── load ─────────────────────────────────────────────────────────────────
    groups = {
        'P': {'W1': [], 'L1': [], 'W2': [], 'L2': [], 'ids': []},
        'U': {'W1': [], 'L1': [], 'W2': [], 'L2': [], 'ids': []},
    }

    for f in files:
        session_id = f.stem.replace(window_suffix, '')
        group      = session_id[0].upper()
        if group not in groups:
            continue

        payload = joblib.load(f)
        summary = payload['summary']

        groups[group]['W1'].append(summary['avg_psv_W_lh'])
        groups[group]['L1'].append(summary['avg_psv_L_lh'])
        groups[group]['W2'].append(summary['avg_psv_W_rh'])
        groups[group]['L2'].append(summary['avg_psv_L_rh'])
        groups[group]['ids'].append(session_id)

    # ── shared axis limit ─────────────────────────────────────────────────────
    all_vals = [v for g in groups.values()
                  for key in ('W1', 'L1', 'W2', 'L2')
                  for v in g[key]]
    lim = max(all_vals) * 1.25 if all_vals else 30

    # ── figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.subplots_adjust(hspace=0.4, wspace=0.35)

    group_colors = {'P': coral, 'U': teal}
    group_names  = {'P': 'Bilateral (P)', 'U': 'Unilateral (U)'}

    col_specs = [
        (0, 'L1', 'W1',
         'LH  —  across vs within %sv',
         'within %sv — LH', 'across %sv — LH',
         'across > within',
         coral, teal),
        (1, 'L2', 'W2',
         'RH  —  across vs within %sv',
         'within %sv — RH', 'across %sv — RH',
         'across > within',
         coral, teal),
        (2, 'W1', 'W2',
         'across %sv  —  LH vs RH',
         'across %sv — LH', 'across %sv — RH',
         'RH > LH',
         teal, teal),
        (3, 'L1', 'L2',
         'within %sv  —  LH vs RH',
         'within %sv — LH', 'within %sv — RH',
         'RH > LH',
         coral, coral),
    ]

    for row, grp in enumerate(('P', 'U')):
        color = group_colors[grp]

        for col, xkey, ykey, title_sfx, xlabel, ylabel, diag_label, \
                x_spine_col, y_spine_col in col_specs:

            ax  = axes[row, col]
            xs  = np.array(groups[grp][xkey])
            ys  = np.array(groups[grp][ykey])
            ids = groups[grp]['ids']

            if len(xs):
                ax.scatter(xs, ys, s=60, alpha=0.85, color=color,
                           marker='o', edgecolors='white',
                           linewidths=0.5, zorder=4)

                for x, y, sid in zip(xs, ys, ids):
                    ax.text(x + lim * 0.02, y + lim * 0.02, sid,
                            fontsize=7, color=color, va='bottom')

            # y = x diagonal
            ax.plot([0, lim], [0, lim], color='#555', linestyle='--',
                    linewidth=1, zorder=2)
            ax.text(lim * 0.05, lim * 0.90, diag_label,
                    fontsize=7, color='#333', style='italic')

            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
            ax.set_aspect('equal', adjustable='box')
            ax.set_title(f'{group_names[grp]}\n{title_sfx}',
                         fontsize=9, fontweight='bold', color=color)
            ax.set_xlabel(xlabel, fontsize=8, color=x_spine_col)
            ax.set_ylabel(ylabel, fontsize=8, color=y_spine_col)
            ax.tick_params(axis='x', colors=x_spine_col, labelsize=7)
            ax.tick_params(axis='y', colors=y_spine_col, labelsize=7)
            ax.spines['bottom'].set_color(x_spine_col)
            ax.spines['left'].set_color(y_spine_col)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(linewidth=0.3, alpha=0.4, zorder=0)

    fig.suptitle(
        f'Average % shared variance  —  window: {window_suffix}',
        fontsize=12, fontweight='bold',
    )

    return fig