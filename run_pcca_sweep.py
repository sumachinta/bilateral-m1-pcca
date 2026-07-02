import matplotlib
matplotlib.use('Agg')  # non-interactive backend; suppresses plt.show()

import numpy as np
import matplotlib.pyplot as plt

from tools.load_session import load_session, TIME_STEP
from tools.trial_epoching import compute_derived
from tools.input_data_generation import get_neuron_mask, prepare_session_for_pcca
from tools.run_save_model import fit_session_pcca, extract_session_metrics, save_session_results
from tools.plotting_fncs import plot_session_metrics

# ── Configuration ─────────────────────────────────────────────────────────────
SESSION_IDS = ['P6', 'P11', 'P12', 'P14']#['P5', 'P6', 'P11', 'P12', 'P14', 'U1', 'U2', 'U3', 'U5', 'U7', 'U8']          # 'P1', 'P2', 'P3', 'P4', 
WINDOWS     = [(0.0, 1.0)]#, (0.0, 0.5), (0.5, 1.0)]
# ──────────────────────────────────────────────────────────────────────────────

for session_id in SESSION_IDS:
    print(f'\n{"=" * 60}')
    print(f'Session: {session_id}')
    print(f'{"=" * 60}')

    try:
        session_data = load_session(f'data/{session_id}.mat')
        derived      = compute_derived(session_data)

        # lick latency on hit trials (needed for the fast-lick filter below)
        reference_frames = derived['trial_start_frames']
        hit_mask         = np.array([o == 'hit' for o in derived['trial_outcome']])
        lick_latency_s   = (
            derived['trial_first_lick_frames'][hit_mask] - reference_frames[hit_mask]
        ) * TIME_STEP

        # trial filtering: hits with lick > 1 s, misses, correct rejections; valid stimuli only
        stim_valid_mask  = np.array([s != 'none' for s in derived['trial_stimulus']])
        miss_mask        = np.array([o == 'miss'        for o in derived['trial_outcome']])
        correct_rej_mask = np.array([o == 'correct_rej' for o in derived['trial_outcome']])

        hit_fast_lick_mask          = hit_mask.copy()
        hit_fast_lick_mask[hit_mask] = lick_latency_s > 1.0

        filtered_trial_indices = np.where(
            (hit_fast_lick_mask | miss_mask | correct_rej_mask) & stim_valid_mask
        )[0]
        print(f'Filtered trials: {len(filtered_trial_indices)}')

        lh_neuron_mask = get_neuron_mask(session_data, hemisphere='LH', fsrs=[1, -1], min_rate_hz=5.0)
        rh_neuron_mask = get_neuron_mask(session_data, hemisphere='RH', fsrs=[1, -1], min_rate_hz=5.0)

        for window in WINDOWS:
            win_label = f'{window[0]:.1f}-{window[1]:.1f}'
            sid       = f'{session_id}_w{win_label}'
            print(f'\n--- Window {win_label} s ---')

            bundle = prepare_session_for_pcca(
                session_data   = session_data,
                derived        = derived,
                trial_indices  = filtered_trial_indices,
                window         = window,
                lh_neuron_mask = lh_neuron_mask,
                rh_neuron_mask = rh_neuron_mask,
            )

            model, cv_results = fit_session_pcca(bundle, d_max=6, n_folds=10, rand_seed=42)
            metrics, summary  = extract_session_metrics(model, session_id=sid)
            save_session_results(sid, model, cv_results, metrics, summary)

            fig = plot_session_metrics(metrics, summary, session_id=sid)
            plt.close(fig)

    except Exception as e:
        print(f'ERROR: session {session_id} failed — {e}')
