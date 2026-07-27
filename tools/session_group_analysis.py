from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .load_session import load_session
from .trial_epoching import compute_derived
from .run_save_model import RESULTS_DIR
from .neuron_behavior_analysis import (
    load_saved_session,
    base_session_id,
    build_trial_variable_table,
    get_trial_latents,
    get_neuron_psv_values,
)
from .tuning_analysis import add_stimulus_group_column, build_tuning_strength_table
from .plotting_fncs import plot_tuning_heatmap, plot_tuning_vs_psv

# ── group / stimulus defaults (match neuron_behavior_analysis.ipynb's Config cell) ─────
UNILATERAL_STIMULI = ['rightC', 'rightD', 'leftC', 'leftD']
BILATERAL_STIMULI  = ['rightC+leftC', 'rightC+leftD', 'rightD+leftC', 'rightD+leftD']

CATEGORICAL_SPECS = [
    ('stim_group', 'stimulus', None),
    ('stimulus', 'stimulus_unilateral', UNILATERAL_STIMULI),
    ('stimulus', 'stimulus_bilateral', BILATERAL_STIMULI),
    ('choice', 'choice', None),
    ('outcome', 'outcome', None),
]
CONTINUOUS_VARS = ['run_speed', 'whisker_angle', 'curvature', 'lick_latency']

HEATMAP_COLUMN_ORDER = [
    'choice', 'outcome', 'lick_latency', 'stimulus', 'stimulus_unilateral', 'stimulus_bilateral',
    'run_speed', 'whisker_angle', 'curvature',
]

OUTCOME_COMPARISONS = [
    'correct_rej_vs_false_alarm', 'correct_rej_vs_hit', 'correct_rej_vs_miss',
    'false_alarm_vs_hit', 'false_alarm_vs_miss', 'hit_vs_miss',
]
PSV_SCATTER_SPECS = (
    [(v, None) for v in CONTINUOUS_VARS]
    + [('choice', None), ('stimulus_unilateral', 'rightC_vs_rightD')]
    + [('outcome', c) for c in OUTCOME_COMPARISONS]
)


def session_group(session_id):
    """'bilateral' (P*) or 'unilateral' (U*), from the first letter of the base session id.

    Same convention already used throughout plotting_fncs.py (P = bilateral, U = unilateral).
    """
    letter = base_session_id(session_id)[0].upper()
    return {'P': 'bilateral', 'U': 'unilateral'}.get(letter, 'unknown')


def discover_sessions(window_suffix='_w0.0-1.0', results_dir=RESULTS_DIR):
    """Session ids (without .joblib) for every saved result matching window_suffix."""
    results_dir = Path(results_dir)
    files = sorted(results_dir.glob(f'*{window_suffix}.joblib'))
    return [f.stem for f in files]


def build_session_tables(session_id, unilateral_stimuli=UNILATERAL_STIMULI,
                          bilateral_stimuli=BILATERAL_STIMULI,
                          categorical_specs=CATEGORICAL_SPECS, continuous_vars=CONTINUOUS_VARS,
                          results_dir=RESULTS_DIR):
    """
    Load one saved session and build its all-neuron tuning-strength tables
    (LH, RH, latent dimensions), tagged with session_id/group columns so they
    can be concatenated across sessions.

    Returns
    -------
    dict with keys:
      neuron_tuning : DataFrame, LH+RH rows combined, columns include
                      session_id, group, hemisphere, neuron_uid, psv_W, psv_L
                      (that neuron's own across/within %shared variance)
      latent_tuning : DataFrame, one row per (latent dim, variable, comparison),
                      columns include session_id, group, latent_type
                      ('across'/'within_LH'/'within_RH')
      metrics, lh_neuron_cols, rh_neuron_cols, latent_cols : needed for plotting
    """
    payload         = load_saved_session(session_id, results_dir=results_dir)
    metrics         = payload['metrics']
    pcca_input_data = payload['pcca_input_data']

    session_data = load_session(f'data/{base_session_id(session_id)}.mat')
    derived      = compute_derived(session_data)

    lh_df = build_trial_variable_table(session_id, metrics, pcca_input_data, session_data, derived,
                                        hemisphere='LH', psv_threshold=None)
    rh_df = build_trial_variable_table(session_id, metrics, pcca_input_data, session_data, derived,
                                        hemisphere='RH', psv_threshold=None)
    lh_df = add_stimulus_group_column(lh_df, unilateral_stimuli, bilateral_stimuli)
    rh_df = add_stimulus_group_column(rh_df, unilateral_stimuli, bilateral_stimuli)

    lh_neuron_cols = [c for c in lh_df.columns if c.startswith('LH_neuron_')]
    rh_neuron_cols = [c for c in rh_df.columns if c.startswith('RH_neuron_')]

    lh_tuning = build_tuning_strength_table(lh_df, lh_neuron_cols, categorical_specs, continuous_vars)
    rh_tuning = build_tuning_strength_table(rh_df, rh_neuron_cols, categorical_specs, continuous_vars)

    lh_psv_W, lh_psv_L = get_neuron_psv_values(metrics, 'LH', 'across'), get_neuron_psv_values(metrics, 'LH', 'within')
    rh_psv_W, rh_psv_L = get_neuron_psv_values(metrics, 'RH', 'across'), get_neuron_psv_values(metrics, 'RH', 'within')
    lh_tuning = lh_tuning.assign(hemisphere='LH', psv_W=lh_tuning['neuron'].map(lh_psv_W), psv_L=lh_tuning['neuron'].map(lh_psv_L))
    rh_tuning = rh_tuning.assign(hemisphere='RH', psv_W=rh_tuning['neuron'].map(rh_psv_W), psv_L=rh_tuning['neuron'].map(rh_psv_L))

    neuron_tuning = pd.concat([lh_tuning, rh_tuning], ignore_index=True)
    neuron_tuning.insert(0, 'session_id', session_id)
    neuron_tuning.insert(1, 'group', session_group(session_id))
    neuron_tuning['neuron_uid'] = neuron_tuning['session_id'] + '_' + neuron_tuning['neuron']

    latents_df  = get_trial_latents(payload, pcca_input_data)
    latent_cols = list(latents_df.columns)
    behavior_cols = ['stimulus', 'stim_group', 'outcome', 'choice', 'lick_latency',
                      'run_speed', 'whisker_angle', 'curvature']
    latents_df = pd.concat(
        [latents_df.reset_index(drop=True), lh_df[behavior_cols].reset_index(drop=True)], axis=1,
    )
    latent_tuning = build_tuning_strength_table(latents_df, latent_cols, categorical_specs, continuous_vars)
    latent_tuning.insert(0, 'session_id', session_id)
    latent_tuning.insert(1, 'group', session_group(session_id))

    def _latent_type(name):
        if name.startswith('z_across'):
            return 'across'
        if name.startswith('z_within_LH'):
            return 'within_LH'
        return 'within_RH'
    latent_tuning['latent_type'] = latent_tuning['neuron'].map(_latent_type)
    latent_tuning['neuron_uid']  = latent_tuning['session_id'] + '_' + latent_tuning['neuron']

    return {
        'neuron_tuning': neuron_tuning,
        'latent_tuning': latent_tuning,
        'metrics': metrics,
        'lh_neuron_cols': lh_neuron_cols,
        'rh_neuron_cols': rh_neuron_cols,
        'latent_cols': latent_cols,
    }


def _save_heatmap_pair(lh_tuning, rh_tuning, metrics, sort_by, column_order, n_lh, n_rh, out_path):
    fig_h = max(6, 0.25 * max(n_lh, n_rh, 1))
    fig, axes = plt.subplots(1, 2, figsize=(12, fig_h))
    plot_tuning_heatmap(lh_tuning, hemisphere='LH', sort_by=sort_by, metrics=metrics, column_order=column_order, ax=axes[0])
    plot_tuning_heatmap(rh_tuning, hemisphere='RH', sort_by=sort_by, metrics=metrics, column_order=column_order, ax=axes[1])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_session_figures(session_id, tables, out_dir, heatmap_column_order=HEATMAP_COLUMN_ORDER,
                          psv_scatter_specs=PSV_SCATTER_SPECS):
    """
    Reproduces, for one session, the heatmap / psv-scatter / latent-heatmap
    cells from neuron_behavior_analysis.ipynb, saved as PNGs under
    out_dir/<session_id>/.
    """
    fig_dir = Path(out_dir) / session_id
    fig_dir.mkdir(parents=True, exist_ok=True)

    neuron_tuning = tables['neuron_tuning']
    lh_tuning = neuron_tuning[neuron_tuning['hemisphere'] == 'LH']
    rh_tuning = neuron_tuning[neuron_tuning['hemisphere'] == 'RH']
    metrics   = tables['metrics']
    n_lh, n_rh = len(tables['lh_neuron_cols']), len(tables['rh_neuron_cols'])

    is_stim = lambda df: df['variable'].str.startswith('stimulus')
    splits = {'stimulus': is_stim, 'other': lambda df: ~is_stim(df)}

    for sort_label, sort_by in [('default', None), ('across', 'across'), ('within', 'within')]:
        for split_name, mask_fn in splits.items():
            _save_heatmap_pair(
                lh_tuning[mask_fn(lh_tuning)], rh_tuning[mask_fn(rh_tuning)], metrics,
                sort_by, heatmap_column_order, n_lh, n_rh,
                fig_dir / f'heatmap_{sort_label}_{split_name}.png',
            )

    for psv_mode in ('across', 'within'):
        fig, axes = plt.subplots(2, len(psv_scatter_specs),
                                  figsize=(4 * len(psv_scatter_specs), 7), squeeze=False)
        for col, (var, comparison) in enumerate(psv_scatter_specs):
            plot_tuning_vs_psv(lh_tuning, psv_mode, variable=var, comparison=comparison,
                                metrics=metrics, hemisphere='LH', ax=axes[0][col])
            plot_tuning_vs_psv(rh_tuning, psv_mode, variable=var, comparison=comparison,
                                metrics=metrics, hemisphere='RH', ax=axes[1][col])
        axes[0][0].set_ylabel('LH — effect size')
        axes[1][0].set_ylabel('RH — effect size')
        fig.suptitle(f'{session_id} — effect size vs. psv_{"W" if psv_mode == "across" else "L"} ({psv_mode})')
        fig.tight_layout()
        fig.savefig(fig_dir / f'psv_scatter_{psv_mode}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

    latent_tuning = tables['latent_tuning']
    fig, ax = plt.subplots(figsize=(10, max(3, 0.5 * len(tables['latent_cols']))))
    plot_tuning_heatmap(latent_tuning, column_order=heatmap_column_order, ax=ax)
    ax.set_title(f'{session_id} — tuning strength (effect size), latent dimensions', fontsize=10, fontweight='bold')
    fig.tight_layout()
    fig.savefig(fig_dir / 'latent_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def run_group_analysis(session_ids=None, window_suffix='_w0.0-1.0', results_dir=RESULTS_DIR,
                        out_dir='group_analysis', unilateral_stimuli=UNILATERAL_STIMULI,
                        bilateral_stimuli=BILATERAL_STIMULI, categorical_specs=CATEGORICAL_SPECS,
                        continuous_vars=CONTINUOUS_VARS, heatmap_column_order=HEATMAP_COLUMN_ORDER,
                        psv_scatter_specs=PSV_SCATTER_SPECS, save_figures=True):
    """
    Loop over every session (or an explicit list), build its tuning-strength
    tables (neurons + latents), optionally save the per-session figures used
    in neuron_behavior_analysis.ipynb, and concatenate everything into two
    long-format tables — one row per (neuron-or-latent, variable, comparison,
    session) — saved for later cross-session / population-level analysis.

    Returns
    -------
    neuron_tuning_all, latent_tuning_all : concatenated DataFrames
    failed : list of (session_id, exception) for sessions that errored out
    """
    out_dir = Path(out_dir)
    (out_dir / 'tables').mkdir(parents=True, exist_ok=True)

    if session_ids is None:
        session_ids = discover_sessions(window_suffix, results_dir)

    neuron_tuning_all, latent_tuning_all, failed = [], [], []

    for session_id in session_ids:
        print(f'--- {session_id} ({session_group(session_id)}) ---')
        try:
            tables = build_session_tables(session_id, unilateral_stimuli, bilateral_stimuli,
                                           categorical_specs, continuous_vars, results_dir)
            neuron_tuning_all.append(tables['neuron_tuning'])
            latent_tuning_all.append(tables['latent_tuning'])

            if save_figures:
                save_session_figures(session_id, tables, out_dir / 'figures',
                                      heatmap_column_order, psv_scatter_specs)
        except Exception as e:
            print(f'  ERROR: {e}')
            failed.append((session_id, e))

    neuron_tuning_all = pd.concat(neuron_tuning_all, ignore_index=True) if neuron_tuning_all else pd.DataFrame()
    latent_tuning_all = pd.concat(latent_tuning_all, ignore_index=True) if latent_tuning_all else pd.DataFrame()

    neuron_tuning_all.to_pickle(out_dir / 'tables' / 'neuron_tuning_all_sessions.pkl')
    latent_tuning_all.to_pickle(out_dir / 'tables' / 'latent_tuning_all_sessions.pkl')

    print(f'\nDone. {len(neuron_tuning_all)} neuron-tuning rows, {len(latent_tuning_all)} latent-tuning rows '
          f'across {len(session_ids) - len(failed)}/{len(session_ids)} sessions.')
    if failed:
        print(f'Failed sessions: {[sid for sid, _ in failed]}')

    return neuron_tuning_all, latent_tuning_all, failed
