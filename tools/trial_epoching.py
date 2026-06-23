import numpy as np

def get_trial_start_indices(session_data):
    return np.where(session_data['frames'] == 1)[0]


def get_touch_start_indices(session_data):
    return np.where(session_data['first_touch'] == 1)[0]


def get_trial_outcomes(session_data, derived):
    trial_start_indices = derived.get('trial_start_indices')
    if trial_start_indices is None:
        trial_start_indices = get_trial_start_indices(session_data)
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

def get_trial_stimuli(session_data, derived, threshold=0.5):
    trial_start_indices = derived.get('trial_start_indices')
    if trial_start_indices is None:
        trial_start_indices = get_trial_start_indices(session_data)

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


def compute_derived(session_data):
    derived = {}
    derived['trial_start_indices'] = get_trial_start_indices(session_data)
    derived['touch_start_indices'] = get_touch_start_indices(session_data)
    derived['trial_outcomes'] = get_trial_outcomes(session_data, derived)
    derived['trial_stimuli'] = get_trial_stimuli(session_data, derived)
    return derived
