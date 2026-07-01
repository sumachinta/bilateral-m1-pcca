import numpy as np
from pcca_fa_mdl import pcca_fa
import joblib
import os
from pathlib import Path


RESULTS_DIR = Path('results')


def fit_session_pcca(bundle, d_max=10, n_folds=10, rand_seed=42, verbose=True):
    """
    Runs pCCA-FA on one session's preprocessed spike count matrices.

    Uses crossvalidate() which:
      1. Searches a grid of (d, d1, d2) from 0 to d_max using k-fold CV
      2. Selects the dimensionality triplet maximising cross-validated log-likelihood
      3. Fits the final model on all trials with the selected dimensionalities

    Parameters
    ----------
    bundle   : dict returned by prepare_session_for_pcca()
    d_max    : int, maximum dimensionality to search (paper uses 15; 6 is
               sufficient for your trial count ~400)
    n_folds  : int, CV folds (paper uses 10)
    rand_seed: int, for reproducibility

    Returns
    -------
    model   : fitted pcca_fa instance
    cv_results : dict from crossvalidate() — includes selected d, d1, d2,
                 per-fold log-likelihoods, and final LL
    """
    X_1 = bundle['lh']   # (T, N_L) preprocessed
    X_2 = bundle['rh']   # (T, N_R) preprocessed

    d_list = np.arange(0, d_max + 1, dtype=int)   # [0, 1, 2, ..., d_max]

    model = pcca_fa()
    cv_results = model.crossvalidate(
        X_1, X_2,
        d_list  = d_list,
        d1_list = d_list,
        d2_list = d_list,
        n_folds = n_folds,
        verbose = verbose,
        rand_seed = rand_seed,
        parallelize = False,   # set True if you want fold-level parallelism
    )

    if verbose:
        print(f"\nSelected dimensionalities:  d={cv_results['d']}, "
              f"d1={cv_results['d1']}, d2={cv_results['d2']}")
        print(f"Final cross-validated LL:   {cv_results['final_LL']:.2f}")

    return model, cv_results



def extract_session_metrics(model, session_id=None):
    """
    Extracts all pCCA-FA metrics from a fit model and computes the
    primary within:across normalised ratio for Suma's hypothesis.

    Primary metric (from project notes):
        within_ratio = avg_psv_L_total / (avg_psv_L_total + avg_psv_W_total)

    High within_ratio → more within-hemisphere shared variance (expected: unilateral)
    Low  within_ratio → more across-hemisphere shared variance (expected: bilateral)

    Returns
    -------
    metrics : dict — all raw metrics from compute_metrics()
    summary : dict — the key numbers you will compare across sessions/groups
    """
    metrics = model.compute_metrics()   # runs psv, dshared, part_ratio, load_sim, rho

    psv = metrics['psv']
    avg_W = psv['avg_psv_W_total']   # % across-hemisphere variance, all neurons
    avg_L = psv['avg_psv_L_total']   # % within-hemisphere variance, all neurons

    # normalised ratio — robust when denominator is small (avoids raw L/W blowup)
    within_ratio = avg_L / (avg_L + avg_W) if (avg_L + avg_W) > 0 else np.nan

    summary = {
        'session_id'     : session_id,
        'avg_psv_W'      : avg_W,              # across-hemisphere %sv
        'avg_psv_L'      : avg_L,              # within-hemisphere %sv
        'within_ratio'   : within_ratio,       # PRIMARY METRIC
        # area-specific (for hemispheric asymmetry secondary analysis)
        'avg_psv_W_lh'   : psv['avg_psv_W_1'],
        'avg_psv_L_lh'   : psv['avg_psv_L_1'],
        'avg_psv_W_rh'   : psv['avg_psv_W_2'],
        'avg_psv_L_rh'   : psv['avg_psv_L_2'],
        'within_ratio_lh': (psv['avg_psv_L_1'] /
                            (psv['avg_psv_L_1'] + psv['avg_psv_W_1'])),
        'within_ratio_rh': (psv['avg_psv_L_2'] /
                            (psv['avg_psv_L_2'] + psv['avg_psv_W_2'])),
        # dimensionality
        'd'  : model.params['d'],
        'd1' : model.params['d1'],
        'd2' : model.params['d2'],
    }

    return metrics, summary



def save_session_results(session_id, model, cv_results, metrics, summary,
                          results_dir=RESULTS_DIR):
    """
    Save everything needed to reproduce all figures and analyses
    without re-running crossvalidate().

    Saves one .joblib file per session containing:
        model_params  — the fit pCCA-FA parameters (W, L, psi, mu, d values)
        cv_results    — the full CV grid search output (LLs per fold, selected dims)
        metrics       — all computed metrics (psv, dshared, rho, etc.)
        summary       — the extracted scalar summary for this session
    """
    results_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        'session_id'  : session_id,
        'model_params': model.get_params(),   # dict of numpy arrays — everything
        'cv_results'  : cv_results,
        'metrics'     : metrics,
        'summary'     : summary,
    }

    path = results_dir / f'{session_id}.joblib'
    joblib.dump(payload, path, compress=3)    # compress=3 is a good size/speed tradeoff
    print(f"Saved → {path}  ({path.stat().st_size / 1024:.1f} KB)")


def load_session_results(session_id, results_dir=RESULTS_DIR):
    """
    Load a saved session and reconstruct a usable pcca_fa model object.

    Returns the same objects as fit_session_pcca + extract_session_metrics:
        model, cv_results, metrics, summary

    The returned model is fully functional — you can call
    compute_psv(), estep(), get_loading_matrices(), etc. on it.
    """
    path = results_dir / f'{session_id}.joblib'
    if not path.exists():
        raise FileNotFoundError(f"No saved results for session '{session_id}' at {path}")

    payload = joblib.load(path)

    # reconstruct a live pcca_fa object from the saved params
    model = pcca_fa()
    model.set_params(payload['model_params'])

    print(f"Loaded ← {path}  (d={model.params['d']}, "
          f"d1={model.params['d1']}, d2={model.params['d2']})")

    return model, payload['cv_results'], payload['metrics'], payload['summary']


def session_is_saved(session_id, results_dir=RESULTS_DIR):
    """Quick check before deciding whether to run or load."""
    return (results_dir / f'{session_id}.joblib').exists()


# def run_or_load_session(session_id, bundle, d_max=6, n_folds=10, rand_seed=42):
#     """
#     Run pCCA-FA if results don't exist yet, otherwise load from disk.
#     Drop-in replacement for run_session().
#     """
#     if session_is_saved(session_id):
#         print(f"[{session_id}] Found saved results — loading.")
#         return load_session_results(session_id)

#     print(f"[{session_id}] No saved results — fitting model (this will take a while).")
#     model, cv_results, metrics, summary = run_session(
#         bundle, session_id=session_id, d_max=d_max,
#         n_folds=n_folds, rand_seed=rand_seed
#     )
#     save_session_results(session_id, model, cv_results, metrics, summary)
#     return model, cv_results, metrics, summary