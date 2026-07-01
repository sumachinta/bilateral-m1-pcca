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