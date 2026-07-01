import numpy as np

def get_session_trial_start_frames(session_data):
    return np.where(session_data['frames'] == 1)[0]


# def get_session_touch_onset_frames(session_data):
#     return np.where(session_data['first_touch'] == 1)[0]
## New version because some trials have no touch events, so we want to return NaN for those trials instead of skipping them. 
def get_session_touch_onset_frames(session_data, derived):
    trial_start_indices = derived.get('trial_start_frames')
    if trial_start_indices is None:
        trial_start_indices = get_session_trial_start_frames(session_data)
    return _first_event_per_trial(session_data['first_touch'], trial_start_indices)



def get_trial_outcome(session_data, derived):
    trial_start_indices = derived.get('trial_start_frames')
    if trial_start_indices is None:
        trial_start_indices = get_session_trial_start_frames(session_data)
    n_trials = len(trial_start_indices)
    outcomes = []

    for i in range(n_trials):
        start = trial_start_indices[i]
        end = trial_start_indices[i + 1] - 1 if i + 1 < n_trials else len(session_data['hits'])

        if   np.any(session_data['hits'][start:end]):         outcomes.append('hit')
        elif np.any(session_data['misses'][start:end]):        outcomes.append('miss')
        elif np.any(session_data['false_alarms'][start:end]):  outcomes.append('false_alarm')
        elif np.any(session_data['correct_rejs'][start:end]):  outcomes.append('correct_rej')
        else:                                                   outcomes.append('unknown')

    return outcomes


PISTON_NAMES = ['rightC', 'rightD', 'leftC', 'leftD']

def get_trial_stimulus(session_data, derived, threshold=0.1):
    trial_start_indices = derived.get('trial_start_frames')
    if trial_start_indices is None:
        trial_start_indices = get_session_trial_start_frames(session_data)

    piston_frames = session_data['piston_frames']   # (n_frames, 4)
    n_trials = len(trial_start_indices)
    stimuli = []

    for i in range(n_trials):
        start = trial_start_indices[i]
        end = trial_start_indices[i + 1] - 1 if i + 1 < n_trials else len(piston_frames)

        trial_pistons = piston_frames[start:end]    # (trial_frames, 4)
        n_frames = len(trial_pistons)

        active = [
            PISTON_NAMES[p]
            for p in range(4)
            if trial_pistons[:, p].sum() / n_frames > threshold
        ]

        stimuli.append('+'.join(active) if active else 'none')

    return stimuli


def _first_event_per_trial(binary_signal, trial_start_indices):
    """Return the absolute frame index of the first 1 in binary_signal for each trial.
    Returns NaN for trials where no event occurs."""
    n_trials = len(trial_start_indices)
    n_frames = len(binary_signal)
    result = np.full(n_trials, np.nan)
    for i in range(n_trials):
        start = trial_start_indices[i]
        end = trial_start_indices[i + 1]- 1 if i + 1 < n_trials else n_frames
        hits = np.where(binary_signal[start:end])[0]
        if hits.size > 0:
            result[i] = start + hits[0]
    return result


def get_trial_first_lick_frames(session_data, derived):
    trial_start_indices = derived.get('trial_start_frames')
    if trial_start_indices is None:
        trial_start_indices = get_session_trial_start_frames(session_data)
    return _first_event_per_trial(session_data['licks'], trial_start_indices)


def compute_derived(session_data):
    derived = {}
    derived['trial_start_frames'] = get_session_trial_start_frames(session_data)
    derived['touch_onset_frames'] = get_session_touch_onset_frames(session_data, derived)
    derived['trial_outcome'] = get_trial_outcome(session_data, derived)
    derived['trial_stimulus'] = get_trial_stimulus(session_data, derived)
    derived['trial_first_lick_frames'] = get_trial_first_lick_frames(session_data, derived)
    return derived
