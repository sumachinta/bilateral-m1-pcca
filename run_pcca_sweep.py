import matplotlib
matplotlib.use('Agg')  # non-interactive backend; suppresses plt.show()

import numpy as np
import matplotlib.pyplot as plt

from tools.load_session import load_session
from tools.trial_epoching import compute_derived
from tools.input_data_generation import prepare_session_for_pcca
from tools.run_save_model import fit_session_pcca, extract_session_metrics, save_session_results
from tools.plotting_fncs import plot_session_metrics

# ── Configuration ─────────────────────────────────────────────────────────────
SESSION_IDS = ['U5','P6', 'U7','P11', 'P12', 'P14','U8']          # ['U1','P2','U2','P3','U3','P4', 'P5', 'U5','P6', 'U7','P11', 'P12', 'P14','U8'] 
WINDOWS     = [(0.0, 1.0)]#, (0.0, 0.5), (0.5, 1.0)]
# ──────────────────────────────────────────────────────────────────────────────

for session_id in SESSION_IDS:
    print(f'\n{"=" * 60}')
    print(f'Session: {session_id}')
    print(f'{"=" * 60}')

    try:
        session_data = load_session(f'data/{session_id}.mat')
        derived      = compute_derived(session_data)

        # trial filtering: hits, misses, correct rejections, false alarms; valid stimuli only
        stim_valid_mask  = np.array([s != 'none' for s in derived['trial_stimulus']])
        hit_mask         = np.array([o == 'hit'         for o in derived['trial_outcome']])
        miss_mask        = np.array([o == 'miss'        for o in derived['trial_outcome']])
        correct_rej_mask = np.array([o == 'correct_rej' for o in derived['trial_outcome']])
        false_alarm_mask = np.array([o == 'false_alarm' for o in derived['trial_outcome']])

        filtered_trial_indices = np.where(
            (hit_mask | miss_mask | correct_rej_mask | false_alarm_mask) & stim_valid_mask
        )[0]
        print(f'Filtered trials: {len(filtered_trial_indices)}')

        neuron_filter_params = {'fsrs': [1, -1], 'min_rate_hz': 5.0}

        for window in WINDOWS:
            win_label = f'{window[0]:.1f}-{window[1]:.1f}'
            sid       = f'{session_id}_w{win_label}'
            print(f'\n--- Window {win_label} s ---')

            pcca_input_data = prepare_session_for_pcca(
                session_data         = session_data,
                derived              = derived,
                trial_indices        = filtered_trial_indices,
                window               = window,
                neuron_filter_params = neuron_filter_params,
            )

            model, cv_results = fit_session_pcca(pcca_input_data, d_max=6, n_folds=10, rand_seed=42)
            metrics, summary  = extract_session_metrics(model, session_id=sid)
            save_session_results(sid, model, cv_results, metrics, summary, pcca_input_data)

            fig = plot_session_metrics(metrics, summary, session_id=sid)
            plt.close(fig)

    except Exception as e:
        print(f'ERROR: session {session_id} failed — {e}')
