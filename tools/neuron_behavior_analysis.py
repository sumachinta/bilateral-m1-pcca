import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .load_session import TIME_STEP
from .input_data_generation import build_windowed_variable_means
from .run_save_model import RESULTS_DIR

CHOICE_MAP = {
    'hit'        : 'lick',
    'false_alarm': 'lick',
    'miss'       : 'no_lick',
    'correct_rej': 'no_lick',
}


def base_session_id(session_id):
    """Strip a trailing '_wSTART-END' window suffix, e.g. 'P1_w0.0-1.0' -> 'P1'."""
    return re.sub(r'_w[\d.]+-[\d.]+$', '', session_id)


def load_saved_session(session_id, results_dir=RESULTS_DIR):
    """
    Load a saved session's joblib payload by exact filename.

    Unlike run_save_model.load_session_results(), this does not hardcode a
    '_w0.0-1.0' suffix, so it works for any session_id exactly as it was
    saved (e.g. 'P1_w0.5-1.0'). Only metrics/summary/pcca_input_data are
    needed for this analysis, so no live pcca_fa model is reconstructed.

    Returns
    -------
    payload : dict with keys session_id, model_params, cv_results, metrics,
        summary, pcca_input_data
    """
    path = Path(results_dir) / f'{session_id}.joblib'
    if not path.exists():
        raise FileNotFoundError(f"No saved results for session '{session_id}' at {path}")
    return joblib.load(path)


def _get_psv_array(metrics, hemisphere, component):
    """
    component : 'across' (psv_W, across-hemisphere shared variance) or
        'within' (psv_L, within-hemisphere shared variance).
    """
    prefix = {'across': 'psv_W', 'within': 'psv_L'}[component]
    suffix = '_1' if hemisphere == 'LH' else '_2'
    return metrics['psv'][prefix + suffix]


def get_high_psv_neuron_indices(metrics, hemisphere='LH', component='across', threshold=50.0):
    """
    Indices (into psv_W_*/psv_L_* and pcca_input_data['lh_raw']/['rh_raw']
    columns) of neurons whose %shared variance exceeds threshold.
    """
    return np.where(_get_psv_array(metrics, hemisphere, component) > threshold)[0]


def get_neuron_psv_values(metrics, hemisphere='LH', component='across'):
    """
    {f'{hemisphere}_neuron_{i}': psv_value} for every neuron in this
    hemisphere. Useful as `sort_by` for plotting_fncs.plot_tuning_heatmap,
    or any other neuron-name-keyed lookup.
    """
    values = _get_psv_array(metrics, hemisphere, component)
    return {f'{hemisphere}_neuron_{i}': v for i, v in enumerate(values)}


def _compute_lick_latency(derived, trial_indices, time_step=TIME_STEP):
    """
    Seconds from trial start to first lick, per trial. NaN if no lick
    occurred in that trial. Generalizes the hit-only calculation in
    data_eda.ipynb to all trials in trial_indices.
    """
    latency_all = (derived['trial_first_lick_frames'] - derived['trial_start_frames']) * time_step
    return latency_all[trial_indices]


def build_trial_variable_table(session_id, metrics, pcca_input_data, session_data, derived,
                                hemisphere='LH', psv_component='across', psv_threshold=50.0,
                                extra_signals=('run_speed', 'whisker_angle', 'curvature')):
    """
    One row per trial: spike counts (raw, within the pCCA window) for neurons
    with high %shared-variance in this hemisphere, plus behavioral variables
    (stimulus, choice, outcome, lick latency, and any extra_signals) aligned
    to the same trial_indices and window used to fit pCCA-FA.

    Parameters
    ----------
    session_id       : str, used only for error messages / provenance
    metrics           : dict from the saved payload ('metrics')
    pcca_input_data   : dict from the saved payload ('pcca_input_data')
    session_data      : dict from load_session() on the raw .mat file
    derived           : dict from compute_derived(session_data)
    hemisphere        : 'LH' or 'RH'
    psv_component     : 'across' or 'within' (passed to get_high_psv_neuron_indices)
    psv_threshold     : float, % shared variance cutoff. None includes every
                         neuron in the hemisphere (no filtering) — useful for
                         later comparing tuning strength between high-psv and
                         low-psv neurons.
    extra_signals     : session_data keys to window-average via
                         build_windowed_variable_means (e.g. 'run_speed')

    Returns
    -------
    pandas.DataFrame, one row per trial. Neuron columns are named
    '{hemisphere}_neuron_{i}', where i is the neuron's index into
    metrics['psv']['psv_W_1'/'psv_W_2'/...] (so it can be traced back to its
    %sv value).
    """
    hemi_key = 'lh' if hemisphere == 'LH' else 'rh'
    raw      = pcca_input_data[f'{hemi_key}_raw']

    if psv_threshold is None:
        neuron_idx = np.arange(raw.shape[1])
    else:
        neuron_idx = get_high_psv_neuron_indices(metrics, hemisphere, psv_component, psv_threshold)

    trial_indices    = pcca_input_data['trial_indices']
    window           = pcca_input_data['window']
    reference_frames = derived['trial_start_frames']

    data = {f'{hemisphere}_neuron_{i}': raw[:, i] for i in neuron_idx}

    data['stimulus']     = np.array(derived['trial_stimulus'])[trial_indices]
    data['outcome']      = pcca_input_data['outcome_labels']
    data['choice']       = np.array([CHOICE_MAP.get(o, 'unknown') for o in data['outcome']])
    data['lick_latency'] = _compute_lick_latency(derived, trial_indices)

    for signal in extra_signals:
        data[signal] = build_windowed_variable_means(
            session_data, trial_indices, reference_frames, window, signal
        )

    return pd.DataFrame(data)


def get_trial_latents(payload, pcca_input_data):
    """
    Reconstructs the fitted pCCA-FA model from payload['model_params'] and
    runs its E-step on the preprocessed trial data (pcca_input_data['lh']/
    ['rh'], the same matrices the model was fit on) to recover each trial's
    latent variable values: across-hemisphere (shared by both hemispheres)
    and within-hemisphere (private to LH / RH).

    Parameters
    ----------
    payload          : dict from load_saved_session() (needs 'model_params')
    pcca_input_data  : dict from the same payload ('pcca_input_data')

    Returns
    -------
    pandas.DataFrame, one row per trial (same order as pcca_input_data),
    columns 'z_across_0'..'z_across_{d-1}', 'z_within_LH_0'..'z_within_LH_{d1-1}',
    'z_within_RH_0'..'z_within_RH_{d2-1}'. These can be treated exactly like
    the neuron spike-count columns from build_trial_variable_table — pass
    them as `neuron_cols` to tuning_analysis.build_tuning_strength_table to
    test whether the latent trajectories themselves relate to behavior.
    """
    from pcca_fa_mdl import pcca_fa

    model = pcca_fa()
    model.set_params(payload['model_params'])
    z, _ = model.estep(pcca_input_data['lh'], pcca_input_data['rh'])

    data = {}
    for i in range(z['z_mu'].shape[1]):
        data[f'z_across_{i}'] = z['z_mu'][:, i]
    for i in range(z['zx1_mu'].shape[1]):
        data[f'z_within_LH_{i}'] = z['zx1_mu'][:, i]
    for i in range(z['zx2_mu'].shape[1]):
        data[f'z_within_RH_{i}'] = z['zx2_mu'][:, i]

    return pd.DataFrame(data)
