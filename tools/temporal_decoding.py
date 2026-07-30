from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _cv_auc(X, y, n_folds, rng, solver='liblinear'):
    """Cross-validated AUC of a z-scored L2 logistic regression on X -> y.
    Scaling is fit per training fold (inside the pipeline) to avoid leakage.
    liblinear is the default solver purely for speed (~2x faster than lbfgs
    at this problem size) since this runs inside a permutation loop."""
    n_per_class = np.bincount(y.astype(int))
    n_folds_eff = min(n_folds, n_per_class.min())
    if n_folds_eff < 2:
        return np.nan
    skf   = StratifiedKFold(n_splits=n_folds_eff, shuffle=True,
                             random_state=int(rng.integers(1_000_000_000)))
    clf   = make_pipeline(StandardScaler(), LogisticRegression(penalty='l2', C=1.0, max_iter=1000, solver=solver))
    probs = cross_val_predict(clf, X, y, cv=skf, method='predict_proba')[:, 1]
    return roc_auc_score(y, probs)


def temporal_pattern_auc(X_a, X_b, n_folds=5, n_permutations=500, seed=0, solver='liblinear',
                          return_null=False):
    """
    Discriminability of two conditions from a neuron's full temporal
    firing-rate pattern (trials x time-bins), using a cross-validated
    L2-regularized logistic regression scored by AUC, plus a label-
    permutation p-value.

    Parameters
    ----------
    X_a, X_b : (n_trials_a, n_bins), (n_trials_b, n_bins) arrays — one
        neuron's binned spike counts per trial, for the two conditions.
    n_folds        : int, stratified CV folds (reduced automatically if a
                      class has fewer trials than n_folds)
    n_permutations : int, label-shuffle permutations for the null distribution
    seed           : int, RNG seed
    return_null    : bool, if True also return the (n_permutations,) null
                      AUC array (for plotting the null distribution)

    Returns
    -------
    auc, p_value [, null_aucs] : floats (+ optional array). auc=0.5 is
        chance, 1.0 is perfect separation. p_value is the fraction of
        permuted-label AUCs >= the observed AUC.
    """
    X = np.vstack([X_a, X_b])
    y = np.concatenate([np.zeros(len(X_a)), np.ones(len(X_b))])
    rng = np.random.default_rng(seed)

    auc = _cv_auc(X, y, n_folds, rng, solver=solver)
    if np.isnan(auc):
        return (np.nan, np.nan, np.full(n_permutations, np.nan)) if return_null else (np.nan, np.nan)

    null_aucs = np.empty(n_permutations)
    for i in range(n_permutations):
        y_perm = rng.permutation(y)
        null_aucs[i] = _cv_auc(X, y_perm, n_folds, rng, solver=solver)

    p_value = (np.sum(null_aucs >= auc) + 1) / (n_permutations + 1)
    return (auc, p_value, null_aucs) if return_null else (auc, p_value)


def fit_temporal_weights(X_a, X_b, solver='liblinear'):
    """
    Single L2-regularized logistic regression fit on ALL trials of X_a vs
    X_b (not cross-validated), returning the (n_bins,) standardized
    coefficient vector — which time bins the classifier relies on to tell
    the two conditions apart. For interpretability only: the AUC/p-value
    discriminability score comes from the cross-validated fits in
    temporal_pattern_auc, not this fit.
    """
    X = np.vstack([X_a, X_b])
    y = np.concatenate([np.zeros(len(X_a)), np.ones(len(X_b))])
    clf = make_pipeline(StandardScaler(), LogisticRegression(penalty='l2', C=1.0, max_iter=1000, solver=solver))
    clf.fit(X, y)
    return clf.named_steps['logisticregression'].coef_.ravel()


def compute_pairwise_discriminability(spike_tensor, labels, categories,
                                       metric_fn=temporal_pattern_auc, weights_out=None, **metric_kwargs):
    """
    Runs metric_fn on every pair of levels in `categories` for one neuron's
    temporal firing pattern — the temporal-pattern analogue of
    tuning_analysis.compute_categorical_pairs.

    Parameters
    ----------
    spike_tensor : (n_trials, n_bins) array — ONE neuron's binned spike
                   counts per trial (a single slice of the (n_trials,
                   n_bins, n_neurons) tensor from build_binned_spike_tensor)
    labels       : (n_trials,) array-like of condition labels per trial
    categories   : list of levels to compare pairwise, e.g. UNILATERAL_STIMULI
    metric_fn    : (X_a, X_b, **kwargs) -> (auc, p_value)
    weights_out  : optional dict; if provided, populated with
                   {(group_a, group_b): (n_bins,) weight_vector} via
                   fit_temporal_weights for each pair with enough trials

    Returns
    -------
    DataFrame with columns: group_a, group_b, auc, p_value, n_a, n_b
    """
    labels = np.asarray(labels)
    rows = []
    for a, b in combinations(categories, 2):
        X_a = spike_tensor[labels == a]
        X_b = spike_tensor[labels == b]
        if len(X_a) < 2 or len(X_b) < 2:
            auc, p = np.nan, np.nan
        else:
            auc, p = metric_fn(X_a, X_b, **metric_kwargs)
            if weights_out is not None:
                weights_out[(a, b)] = fit_temporal_weights(X_a, X_b)
        rows.append({
            'group_a': a, 'group_b': b,
            'auc': auc, 'p_value': p,
            'n_a': len(X_a), 'n_b': len(X_b),
        })
    return pd.DataFrame(rows)


def run_all_neurons(tensor, hemisphere, labels_by_family, condition_families, session_id,
                     n_folds=5, n_permutations=50, seed=0, metric_fn=temporal_pattern_auc,
                     collect_weights=False):
    """
    Sweep compute_pairwise_discriminability across every neuron in `tensor`
    and every condition family, tagging each row with neuron/hemisphere/
    session identity. One long-format row per (neuron, family, pair).

    Parameters
    ----------
    tensor              : (n_trials, n_bins, n_neurons) array — one hemisphere
    hemisphere          : 'LH' or 'RH' (used to name neurons and tag rows)
    labels_by_family    : {family_name: (n_trials,) array of condition labels}
    condition_families  : {family_name: [levels...]}
    session_id          : str, tagged onto every row
    collect_weights     : bool, if True also fit and return each pair's
                           (n_bins,) classifier weight vector (see
                           fit_temporal_weights) — cheap (one extra fit per
                           pair, no permutations), useful for later "when
                           does each neuron start discriminating" analysis

    Returns
    -------
    discrim_df, or (discrim_df, weights) if collect_weights is True.
    discrim_df columns: group_a, group_b, auc, p_value, n_a, n_b, neuron,
    hemisphere, session_id, variable. weights is a dict keyed by
    '{neuron}__{variable}__{group_a}_vs_{group_b}' -> (n_bins,) array.
    """
    n_neurons = tensor.shape[2]
    rows = []
    weights = {} if collect_weights else None
    for i in range(n_neurons):
        neuron_name = f'{hemisphere}_neuron_{i}'
        for family, categories in condition_families.items():
            pair_weights = {} if collect_weights else None
            pairwise = compute_pairwise_discriminability(
                tensor[:, :, i], labels_by_family[family], categories,
                metric_fn=metric_fn, n_folds=n_folds, n_permutations=n_permutations, seed=seed,
                weights_out=pair_weights)
            pairwise['neuron']     = neuron_name
            pairwise['hemisphere'] = hemisphere
            pairwise['session_id'] = session_id
            pairwise['variable']   = family
            rows.append(pairwise)
            if collect_weights:
                for (a, b), w in pair_weights.items():
                    weights[f'{neuron_name}__{family}__{a}_vs_{b}'] = w
    discrim_df = pd.concat(rows, ignore_index=True)
    return (discrim_df, weights) if collect_weights else discrim_df
