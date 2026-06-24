from load_session import TIME_STEP
import numpy as np
from scipy.ndimage import uniform_filter1d

BOXCAR_WIDTH = 25      # trials, per paper section 4.8

def get_neuron_mask(session_data, hemisphere='LH', fsrs=None, min_rate_hz=None,
                    time_step=TIME_STEP):
    """
    Returns a boolean mask scoped to this hemisphere's neurons.
    True = neuron passes all filters.
    """
    loc_mask = (session_data['loc_lh_mask'] if hemisphere == 'LH'
                else session_data['loc_rh_mask'])
    mask = loc_mask.copy()

    if fsrs is not None:
        mask &= np.isin(session_data['fsrs'], fsrs)

    if min_rate_hz is not None:
        hemi_spikes = session_data['spikes'][:, loc_mask]
        mean_rate   = hemi_spikes.mean(axis=0) / time_step
        rate_filter = np.zeros(len(loc_mask), dtype=bool)
        rate_filter[loc_mask] = mean_rate >= min_rate_hz
        mask &= rate_filter

    return mask[loc_mask]   # scoped to hemisphere



def build_spike_count_matrices(session_data, trial_indices, reference_frames,
                                 window, lh_neuron_mask, rh_neuron_mask,
                                 time_step=TIME_STEP):
    """
    Raw (unpreprocessed) spike count matrices.
    Returns lh_matrix (T, N_L) and rh_matrix (T, N_R).
    """
    win_start = int(round(window[0] / time_step))
    win_end   = int(round(window[1] / time_step))
    n_frames  = session_data['spikes'].shape[0]
    T         = len(trial_indices)

    lh_matrix = np.zeros((T, int(lh_neuron_mask.sum())), dtype=float)
    rh_matrix = np.zeros((T, int(rh_neuron_mask.sum())), dtype=float)

    for row, trial_idx in enumerate(trial_indices):
        ref   = int(reference_frames[trial_idx])
        start = max(ref + win_start, 0)
        end   = min(ref + win_end,   n_frames)

        lh_spikes = session_data['spikes'][start:end][:, session_data['loc_lh_mask']]
        rh_spikes = session_data['spikes'][start:end][:, session_data['loc_rh_mask']]

        lh_matrix[row] = lh_spikes[:, lh_neuron_mask].sum(axis=0)
        rh_matrix[row] = rh_spikes[:, rh_neuron_mask].sum(axis=0)

    return lh_matrix, rh_matrix


def _subtract_condition_means(matrix, condition_labels):
    """
    For each stimulus condition c and each neuron n, subtract the mean spike
    count across trials of condition c. Operates on assembled (T, N) matrix.
    """
    residual = matrix.copy().astype(float)
    for cond in np.unique(condition_labels):
        idx = condition_labels == cond
        residual[idx] -= residual[idx].mean(axis=0, keepdims=True)
    return residual


def _remove_slow_drift(matrix, boxcar_width=BOXCAR_WIDTH):
    """
    Centered boxcar filter of width `boxcar_width` trials along the trial axis.
    Subtracts the smoothed (slow) component from each neuron independently.
    """
    slow = uniform_filter1d(matrix.astype(float),
                            size=boxcar_width, axis=0, mode='nearest')
    return matrix - slow


# ── helpers ────────────────────────────────────────────────────────────────────
def _make_stimulus_labels(trial_outcomes):
    """
    Derive stimulus-identity labels from trial outcomes.

    For both unilateral and bilateral mice:
      hit + miss  → 'go'   (Go stimulus was presented, animal licked or not)
      correct_rej → 'nogo' (NoGo stimulus was presented, correctly withheld)

    False alarms are excluded from the primary analysis so not handled here.
    Extend this function if you have finer-grained stimulus coding in your data
    (e.g. which specific whisker pair for bilateral mice).
    """
    labels = np.empty(len(trial_outcomes), dtype=object)
    for i, outcome in enumerate(trial_outcomes):
        if outcome in ('hit', 'miss'):
            labels[i] = 'go'
        elif outcome == 'correct_rej':
            labels[i] = 'nogo'
        else:
            labels[i] = 'other'   # false_alarm or unknown
    return labels


# ── HIGH-LEVEL ENTRY POINT ────────────────────────────────────────────────────
def prepare_session_for_pcca(session_data, derived,
                              trial_indices,
                              window,
                              lh_neuron_mask,
                              rh_neuron_mask,
                              boxcar_width=BOXCAR_WIDTH,
                              time_step=TIME_STEP):
    """

    Builds raw spike count matrices, attaches trial metadata, applies
    condition-mean subtraction and boxcar detrending,
    and returns everything in one self-contained bundle.

    Parameters
    ----------
    session_data     : dict from load_session()
    derived          : dict with trial_outcomes, trial_start_indices, etc.
    trial_indices    : (T,) int array — which trials to include (already filtered)
    window           : (start_s, end_s) tuple, seconds relative to stimulus onset
    lh_neuron_mask   : (N_L,) bool — from get_neuron_mask('LH')
    rh_neuron_mask   : (N_R,) bool — from get_neuron_mask('RH')
    boxcar_width     : int, trials (default 25)

    Returns
    -------
    bundle : dict with keys
        'lh_raw'            (T, N_L)  raw spike counts
        'rh_raw'            (T, N_R)  raw spike counts
        'lh'                (T, N_L)  preprocessed (mean-sub + detrended)
        'rh'                (T, N_R)  preprocessed
        'trial_indices'     (T,)      which trials were used
        'outcome_labels'    (T,)      hit / miss / correct_rej per trial
        'stimulus_labels'   (T,)      go / nogo per trial  ← for mean subtraction
        'window'            tuple     the window used
        'n_lh'              int       neurons kept in LH
        'n_rh'              int       neurons kept in RH
    """
    # 1. grab per-trial metadata for the selected trials
    all_outcomes   = np.array(derived['trial_outcomes'])
    outcome_labels = all_outcomes[trial_indices]
    stim_labels    = _make_stimulus_labels(outcome_labels)

    # 2. build raw spike count matrices
    lh_raw, rh_raw = build_spike_count_matrices(
        session_data,
        trial_indices    = trial_indices,
        reference_frames = derived['stimulus_onset_frame'],   # align to stim, not trial start
        window           = window,
        lh_neuron_mask   = lh_neuron_mask,
        rh_neuron_mask   = rh_neuron_mask,
        time_step        = time_step,
    )

    # 3. condition mean subtraction  (uses stimulus identity, not outcome)
    lh_resid = _subtract_condition_means(lh_raw, stim_labels)
    rh_resid = _subtract_condition_means(rh_raw, stim_labels)

    # 4. slow drift removal
    lh_pre = _remove_slow_drift(lh_resid, boxcar_width)
    rh_pre = _remove_slow_drift(rh_resid, boxcar_width)

    return {
        'lh_raw'          : lh_raw,
        'rh_raw'          : rh_raw,
        'lh'              : lh_pre,
        'rh'              : rh_pre,
        'trial_indices'   : trial_indices,
        'outcome_labels'  : outcome_labels,
        'stimulus_labels' : stim_labels,
        'window'          : window,
        'n_lh'            : int(lh_neuron_mask.sum()),
        'n_rh'            : int(rh_neuron_mask.sum()),
    }
