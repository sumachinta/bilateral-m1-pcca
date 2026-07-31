import time
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # non-interactive backend; suppresses plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from tools.load_session import load_session
from tools.trial_epoching import compute_derived
from tools.temporal_profile import build_binned_spike_tensor
from tools.temporal_decoding import run_all_neurons
from tools.neuron_behavior_analysis import attach_temporal_psv_columns, base_session_id
from tools.session_group_analysis import discover_sessions
from tools.run_save_model import RESULTS_DIR

# ── Configuration ─────────────────────────────────────────────────────────────
BIN_WIDTH_S    = 0.05          # 50 ms bins
WINDOW         = (0.0, 2.0)    # seconds relative to trial start
N_FOLDS        = 5
N_PERMUTATIONS = 50            # matches temporal_pattern_discrimination.ipynb's full-sweep setting
SEED           = 0
MIN_RELIABLE_N = 15            # pairs with fewer trials than this on either side are flagged low_n

UNILATERAL_STIMULI = ['rightC', 'rightD', 'leftC', 'leftD']
BILATERAL_STIMULI  = ['rightC+leftC', 'rightC+leftD', 'rightD+leftC', 'rightD+leftD']
OUTCOMES            = ['hit', 'miss', 'false_alarm', 'correct_rej']
CONDITION_FAMILIES = {
    'stimulus_unilateral': UNILATERAL_STIMULI,
    'stimulus_bilateral':  BILATERAL_STIMULI,
    'outcome':             OUTCOMES,
}

# every session with a saved pCCA-FA fit at the 0.0-1.0s window (needed both
# to confirm the raw .mat exists and to attach psv_across/psv_within later)
SESSION_IDS = ['P4', 'P5', 'P6']#['P11', 'P14', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'U2', 'U3', 'U5', 'U7']
#[base_session_id(sid) for sid in discover_sessions()]
# ──────────────────────────────────────────────────────────────────────────────

for session_id in SESSION_IDS:
    print(f'\n{"=" * 60}')
    print(f'Session: {session_id}')
    print(f'{"=" * 60}')
    t_session = time.time()

    try:
        figures_dir = Path('figures') / f'{session_id}_temporal'
        figures_dir.mkdir(parents=True, exist_ok=True)

        # ── load + trial labels ─────────────────────────────────────────────
        session_data = load_session(f'data/{session_id}.mat')
        derived      = compute_derived(session_data)
        stimulus = np.array(derived['trial_stimulus'])
        outcome  = np.array(derived['trial_outcome'])
        labels_by_family = {
            'stimulus_unilateral': stimulus,
            'stimulus_bilateral':  stimulus,
            'outcome':             outcome,
        }

        # ── binned spike tensors ────────────────────────────────────────────
        reference_frames = derived['trial_start_frames']
        lh_tensor, bin_centers = build_binned_spike_tensor(
            session_data, reference_frames, BIN_WIDTH_S, WINDOW, session_data['loc_lh_mask'])
        rh_tensor, _ = build_binned_spike_tensor(
            session_data, reference_frames, BIN_WIDTH_S, WINDOW, session_data['loc_rh_mask'])
        print(f'LH tensor: {lh_tensor.shape}, RH tensor: {rh_tensor.shape}')

        # ── full sweep: every neuron x every pair (+ per-bin classifier weights) ─
        lh_discrim, lh_weights = run_all_neurons(
            lh_tensor, 'LH', labels_by_family, CONDITION_FAMILIES, session_id,
            n_folds=N_FOLDS, n_permutations=N_PERMUTATIONS, seed=SEED, collect_weights=True)
        rh_discrim, rh_weights = run_all_neurons(
            rh_tensor, 'RH', labels_by_family, CONDITION_FAMILIES, session_id,
            n_folds=N_FOLDS, n_permutations=N_PERMUTATIONS, seed=SEED, collect_weights=True)
        temporal_discrim_all = pd.concat([lh_discrim, rh_discrim], ignore_index=True)
        temporal_discrim_all['low_n'] = (temporal_discrim_all[['n_a', 'n_b']].min(axis=1) < MIN_RELIABLE_N)
        print(f'Full sweep: {len(temporal_discrim_all)} rows in {time.time() - t_session:.0f}s')

        weights_path = RESULTS_DIR / f'temporal_pattern_weights_{session_id}.npz'
        np.savez(weights_path, bin_centers=bin_centers, **lh_weights, **rh_weights)
        print(f'Saved {len(lh_weights) + len(rh_weights)} weight vectors to {weights_path}')

        # ── session-level per-pair summary ──────────────────────────────────
        pair_summary = (
            temporal_discrim_all
            .groupby(['variable', 'group_a', 'group_b'])
            .agg(
                n_neurons=('auc', 'size'),
                frac_above_0_5=('auc', lambda x: (x > 0.5).mean()),
                frac_above_0_75=('auc', lambda x: (x > 0.75).mean()),
                n_a=('n_a', 'first'),
                n_b=('n_b', 'first'),
                low_n=('low_n', 'first'),
            )
            .reset_index()
            .sort_values('frac_above_0_5', ascending=False)
        )
        pair_summary_path = RESULTS_DIR / f'temporal_pattern_pair_summary_{session_id}.csv'
        pair_summary.to_csv(pair_summary_path, index=False)
        print(f'Saved {len(pair_summary)} rows to {pair_summary_path}')

        # ── figure: per-pair discriminability, one subplot per pair ────────
        n_pairs = 6  # 4 choose 2, per family
        fig, axes = plt.subplots(len(CONDITION_FAMILIES), n_pairs, figsize=(24, 9), sharex=True)
        for row_idx, (family, categories) in enumerate(CONDITION_FAMILIES.items()):
            family_df = temporal_discrim_all[temporal_discrim_all['variable'] == family]
            for col_idx, (a, b) in enumerate(combinations(categories, 2)):
                ax = axes[row_idx, col_idx]
                pair_df = family_df[(family_df['group_a'] == a) & (family_df['group_b'] == b)]
                reliable_vals = pair_df.loc[~pair_df['low_n'], 'auc']
                low_n_vals    = pair_df.loc[pair_df['low_n'], 'auc']
                ax.hist(reliable_vals, bins=20, color='steelblue', label=f'reliable (n >= {MIN_RELIABLE_N})')
                ax.hist(low_n_vals, bins=20, color='crimson', alpha=0.7, label='low n')
                ax.axvline(0.5, color='gray', ls='--', lw=1)
                ax.set_title(f'{a}\nvs {b}', fontsize=8)
                if col_idx == 0:
                    ax.set_ylabel(f'{family}\ncount (neurons)', fontsize=9)
        axes[0, 0].legend(fontsize=7, loc='upper left')
        fig.suptitle(f'{session_id}: per-neuron discriminability, one subplot per pair')
        fig.tight_layout()
        fig.savefig(figures_dir / 'summary_auc_distribution.png', dpi=150)
        plt.close(fig)

        # ── figure: top-10 example neuron/pair traces ───────────────────────
        reliable = temporal_discrim_all[~temporal_discrim_all['low_n']]
        top10 = reliable.sort_values('auc', ascending=False).head(10).reset_index(drop=True)
        fig, axes = plt.subplots(5, 2, figsize=(9, 14), sharex=True)
        for row, ax in zip(top10.itertuples(), axes.flat):
            hemisphere, idx = row.neuron.split('_neuron_')
            idx    = int(idx)
            tensor = lh_tensor if hemisphere == 'LH' else rh_tensor
            labels = labels_by_family[row.variable]
            X_neuron = tensor[:, :, idx]
            X_a = X_neuron[labels == row.group_a]
            X_b = X_neuron[labels == row.group_b]
            ax.plot(bin_centers, X_a.mean(axis=0) / BIN_WIDTH_S, label=f'{row.group_a} (n={len(X_a)})')
            ax.plot(bin_centers, X_b.mean(axis=0) / BIN_WIDTH_S, label=f'{row.group_b} (n={len(X_b)})')
            ax.set_title(f'{row.neuron}  {row.variable}\nAUC={row.auc:.3f}, p={row.p_value:.4f}', fontsize=9)
            ax.legend(fontsize=8)
            ax.set_ylabel('firing rate (Hz)')
        for ax in axes[-1]:
            ax.set_xlabel('time from trial start (s)')
        fig.suptitle(f'{session_id}: top 10 most discriminating neuron/pair combos')
        fig.tight_layout()
        fig.savefig(figures_dir / 'top10_trial_averaged_traces.png', dpi=150)
        plt.close(fig)

        # ── attach % shared variance, final save ────────────────────────────
        temporal_discrim_all = attach_temporal_psv_columns(temporal_discrim_all, session_id)
        out_path = RESULTS_DIR / f'temporal_pattern_discriminability_{session_id}.csv'
        temporal_discrim_all.to_csv(out_path, index=False)
        print(f'Saved {len(temporal_discrim_all)} rows (with psv_across/psv_within) to {out_path}')

        # ── figure: discriminability vs shared variance ─────────────────────
        neuron_summary = (
            temporal_discrim_all[~temporal_discrim_all['low_n']]
            .groupby('neuron')
            .agg(max_auc=('auc', 'max'), psv_across=('psv_across', 'first'), psv_within=('psv_within', 'first'))
            .reset_index()
            .dropna(subset=['psv_across', 'psv_within'])
        )
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for ax, psv_col, label in zip(axes, ['psv_across', 'psv_within'], ['across-hemisphere', 'within-hemisphere']):
            if len(neuron_summary) >= 2:
                rho, p = spearmanr(neuron_summary[psv_col], neuron_summary['max_auc'])
                ax.set_title(f'rho={rho:.2f}, p={p:.4f}')
            ax.scatter(neuron_summary[psv_col], neuron_summary['max_auc'], alpha=0.5, s=15)
            ax.axhline(0.5, color='gray', ls='--', lw=0.8)
            ax.set_xlabel(f'% shared variance ({label})')
            ax.set_ylabel('max discriminability AUC\n(best reliable pair, any family)')
        fig.suptitle(f'{session_id}: does high discriminability track high shared variance?')
        fig.tight_layout()
        fig.savefig(figures_dir / 'psv_vs_max_discriminability.png', dpi=150)
        plt.close(fig)

        print(f'Session {session_id} done in {time.time() - t_session:.0f}s')

    except Exception as e:
        print(f'ERROR: session {session_id} failed — {e}')
