import numpy as np

from .load_session import TIME_STEP


def build_binned_spike_tensor(session_data, reference_frames, bin_width_s, window,
                               neuron_mask, time_step=TIME_STEP):
    """
    Bin each trial's spike counts into fixed-width time bins relative to
    reference_frames, preserving the time axis (unlike
    input_data_generation.build_spike_count_matrices, which sums it away).

    Parameters
    ----------
    session_data      : dict from load_session()
    reference_frames  : (n_trials,) frame indices each trial is aligned to
                         (e.g. derived['trial_start_frames'])
    bin_width_s       : float, seconds per bin
    window            : (start_s, end_s) tuple, seconds relative to
                         reference_frames
    neuron_mask       : boolean mask into session_data['spikes'] columns
                         (e.g. session_data['loc_lh_mask'])
    time_step         : seconds per frame

    Returns
    -------
    spike_tensor : (n_trials, n_bins, n_neurons) float array of summed
                   spike counts per bin. NaN for any bin that falls outside
                   the recorded session (trial too close to session start/end).
    bin_centers  : (n_bins,) array, bin-center offsets in seconds relative
                   to reference_frames.
    """
    bin_frames      = int(round(bin_width_s / time_step))
    win_start_frame = int(round(window[0] / time_step))
    win_end_frame   = int(round(window[1] / time_step))
    n_bins          = (win_end_frame - win_start_frame) // bin_frames

    spikes         = session_data['spikes'][:, neuron_mask]
    n_frames_total = spikes.shape[0]
    n_neurons      = spikes.shape[1]
    n_trials       = len(reference_frames)

    spike_tensor = np.full((n_trials, n_bins, n_neurons), np.nan)
    for row, ref in enumerate(reference_frames):
        if np.isnan(ref):
            continue
        ref = int(ref)
        for b in range(n_bins):
            start = ref + win_start_frame + b * bin_frames
            end   = start + bin_frames
            if start < 0 or end > n_frames_total:
                continue
            spike_tensor[row, b] = spikes[start:end].sum(axis=0)

    bin_centers = (win_start_frame + (np.arange(n_bins) + 0.5) * bin_frames) * time_step
    return spike_tensor, bin_centers
