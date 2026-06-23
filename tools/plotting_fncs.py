import matplotlib.pyplot as plt

def plot_trial_variable(session_data, trial_start_indices, var_name='run_speed', trial_idx=1, xlim=None, ylim=None):
    """
    Plot a variable for a single trial.

    Parameters
    ----------
    session_data : dict-like
        Must contain:
        - 'absolute_frames'
        - 'time'
        - var_name
    trial_start_indices : array-like
        Start frame of each trial.
    var_name : str, optional
        Variable to plot (e.g. 'hits', 'misses', 'run_speed').
    trial_idx : int, optional
        Trial number (1-based indexing).
    """
    
    frame_start = trial_start_indices[trial_idx - 1]
    frame_end = trial_start_indices[trial_idx] - 1

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