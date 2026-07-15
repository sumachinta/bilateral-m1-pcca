from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr


def add_stimulus_group_column(df, unilateral_stimuli, bilateral_stimuli):
    """
    Derive a pooled binary 'stim_group' column ('unilateral'/'bilateral') from
    the existing 'stimulus' column. Trials whose stimulus is in neither list
    (e.g. 'none') get NaN and are dropped by downstream comparisons.
    """
    stim_group = pd.Series(np.nan, index=df.index, dtype=object)
    stim_group[df['stimulus'].isin(unilateral_stimuli)] = 'unilateral'
    stim_group[df['stimulus'].isin(bilateral_stimuli)]  = 'bilateral'
    return df.assign(stim_group=stim_group)


# ── metric functions ──────────────────────────────────────────────────────────
# Contract: metric_fn(x, y) -> (value, effect_size, p_value)
#   categorical : x, y are two groups' spike-count arrays
#   continuous  : x, y are (spike_counts, variable_values)
# A future mutual-information method (with a shuffle-based null for p_value)
# can be written as a drop-in replacement for either of these, following the
# same 3-tuple contract, without touching the extraction/orchestration code
# below or the plotting functions in plotting_fncs.py.

def _auc_mannwhitney(x, y):
    """AUC (= U / (n_x * n_y)) + Mann-Whitney p-value. 0.5 = chance."""
    stat, p = mannwhitneyu(x, y, alternative='two-sided')
    value = stat / (len(x) * len(y))
    effect_size = abs(value - 0.5) * 2
    return value, effect_size, p


def _spearman(x, y):
    """Spearman rho + p-value. NaNs (e.g. lick_latency on no-lick trials) are omitted."""
    rho, p = spearmanr(x, y, nan_policy='omit')
    return rho, abs(rho), p


# ── data extraction (kept separate from metric computation for reuse) ─────────

def _get_categorical_groups(df, neuron_col, category_col, categories=None):
    """Returns {level: spike_count_array} for every level present (or given)."""
    if categories is None:
        categories = sorted(df[category_col].dropna().unique())
    return {level: df.loc[df[category_col] == level, neuron_col].to_numpy() for level in categories}


def _get_continuous_pair(df, neuron_col, variable):
    """Returns (spike_counts, variable_values), NaN rows in either dropped."""
    pair = df[[neuron_col, variable]].dropna()
    return pair[neuron_col].to_numpy(), pair[variable].to_numpy()


# ── per-neuron computations ────────────────────────────────────────────────────

def compute_categorical_pairs(df, neuron_col, category_col, categories=None, metric_fn=_auc_mannwhitney):
    """
    Runs metric_fn on every pair of levels present in category_col (1 pair for
    binary columns like 'choice'/'stim_group', 3 for 'outcome').

    Returns a DataFrame with columns: group_a, group_b, value, effect_size,
    p_value, n_a, n_b.
    """
    groups = _get_categorical_groups(df, neuron_col, category_col, categories)
    rows = []
    for a, b in combinations(groups.keys(), 2):
        x, y = groups[a], groups[b]
        value, effect_size, p = metric_fn(x, y)
        rows.append({
            'group_a': a, 'group_b': b,
            'value': value, 'effect_size': effect_size, 'p_value': p,
            'n_a': len(x), 'n_b': len(y),
        })
    return pd.DataFrame(rows)


def compute_continuous_metric(df, neuron_col, variable, metric_fn=_spearman):
    """
    Runs metric_fn on (spike_counts, variable) for one neuron/variable pair.

    Returns (value, effect_size, p_value, n).
    """
    x, y = _get_continuous_pair(df, neuron_col, variable)
    value, effect_size, p = metric_fn(x, y)
    return value, effect_size, p, len(x)


# ── top-level orchestrator ──────────────────────────────────────────────────────

def build_tuning_strength_table(df, neuron_cols, categorical_specs, continuous_vars,
                                 categorical_metric_fn=_auc_mannwhitney,
                                 continuous_metric_fn=_spearman):
    """
    One row per (neuron, variable, comparison): how strongly that neuron's
    spike counts relate to that behavioral variable.

    Parameters
    ----------
    df                 : DataFrame from build_trial_variable_table()
    neuron_cols        : list of spike-count column names to score
    categorical_specs  : list of (column_name, variable_label, categories)
                          triples, e.g.
                          [('stim_group', 'stimulus', None),
                           ('stimulus', 'stimulus_unilateral', unilateral_stimuli),
                           ('stimulus', 'stimulus_bilateral', bilateral_stimuli),
                           ('choice', 'choice', None),
                           ('outcome', 'outcome', None)]
                          `categories=None` compares every level present in
                          that column; pass a list to restrict/order which
                          levels are compared — e.g. the two rows above reuse
                          the full 'stimulus' column (8 identities) but each
                          restrict the pairwise comparison to just the 4
                          unilateral (or bilateral) identities, so you get
                          within-group discriminability instead of only the
                          pooled unilateral-vs-bilateral comparison.
    continuous_vars    : list of continuous column names, e.g.
                          ['run_speed', 'whisker_angle', 'curvature', 'lick_latency']
    categorical_metric_fn, continuous_metric_fn : pluggable metric_fn
        (see contract above). Defaults are the AUC/Spearman first pass; a
        future method (e.g. mutual information) can be swapped in here
        without changing this function's structure.

    Returns
    -------
    pandas.DataFrame with columns: neuron, variable, comparison, method,
    value, effect_size, p_value, n
    """
    categorical_method = categorical_metric_fn.__name__.lstrip('_')
    continuous_method  = continuous_metric_fn.__name__.lstrip('_')

    rows = []
    for neuron_col in neuron_cols:
        for column_name, variable_label, categories in categorical_specs:
            pairs = compute_categorical_pairs(df, neuron_col, column_name, categories=categories,
                                               metric_fn=categorical_metric_fn)
            for _, pair_row in pairs.iterrows():
                rows.append({
                    'neuron': neuron_col,
                    'variable': variable_label,
                    'comparison': f"{pair_row['group_a']}_vs_{pair_row['group_b']}",
                    'method': categorical_method,
                    'value': pair_row['value'],
                    'effect_size': pair_row['effect_size'],
                    'p_value': pair_row['p_value'],
                    'n': pair_row['n_a'] + pair_row['n_b'],
                })

        for variable in continuous_vars:
            value, effect_size, p, n = compute_continuous_metric(df, neuron_col, variable, metric_fn=continuous_metric_fn)
            rows.append({
                'neuron': neuron_col,
                'variable': variable,
                'comparison': 'all',
                'method': continuous_method,
                'value': value,
                'effect_size': effect_size,
                'p_value': p,
                'n': n,
            })

    return pd.DataFrame(rows)
