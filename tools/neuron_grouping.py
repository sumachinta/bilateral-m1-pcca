import pandas as pd

# collapse the fine-grained 'variable' values onto 4 broad functional buckets
CATEGORY_MAP = {
    'stimulus':            'stimulus',
    'stimulus_unilateral': 'stimulus',
    'stimulus_bilateral':  'stimulus',
    'outcome':              'outcome',
    'choice':               'choice',
    'run_speed':            'movement',
    'whisker_angle':        'movement',
    'curvature':            'movement',
    'lick_latency':         'movement',
}
CATEGORIES = sorted(set(CATEGORY_MAP.values()))

IDENTITY_COLS = ['neuron_uid', 'neuron', 'session_id', 'group', 'hemisphere', 'psv_W', 'psv_L']


def add_category_column(tuning_df, category_map=CATEGORY_MAP):
    """Maps each row's fine-grained 'variable' onto one of the broad functional
    categories in `category_map` (stimulus / outcome / choice / movement by
    default), so neurons can be grouped by *what kind* of thing they're tuned
    to rather than by every individual variable/comparison.
    """
    return tuning_df.assign(category=tuning_df['variable'].map(category_map))


def summarize_neuron_selectivity(tuning_df, alpha=0.05, category_map=CATEGORY_MAP,
                                  identity_cols=IDENTITY_COLS):
    """
    Collapse a long-format tuning table (one row per neuron x variable x
    comparison) down to one row per neuron, summarizing which broad
    behavioral category (if any) it's significantly tuned to.

    For each neuron and category, takes the max effect_size among that
    category's significant (p_value < alpha) rows -- 0 if none are
    significant. 'dominant_category' is whichever category has the highest
    such score ('untuned' if every category is 0). 'selectivity_breadth'
    counts how many categories the neuron is significantly tuned to at all
    (0..len(categories)) -- a measure of mixed selectivity.

    Parameters
    ----------
    tuning_df     : DataFrame, e.g. neuron_tuning_all / latent_tuning_all
    alpha         : float, p-value cutoff for "significantly tuned"
    category_map  : dict, variable name -> broad category name
    identity_cols : which per-neuron columns (constant across a neuron's
                    rows) to carry through into the output

    Returns
    -------
    DataFrame, one row per neuron_uid: identity_cols (whichever are present
    in tuning_df) + one column per category (max significant effect_size,
    0 if none) + dominant_category + selectivity_breadth.
    """
    df = add_category_column(tuning_df, category_map)
    categories = sorted(set(category_map.values()))

    identity = (df[[c for c in identity_cols if c in df.columns]]
                .drop_duplicates(subset='neuron_uid')
                .set_index('neuron_uid'))

    sig = df[df['p_value'] < alpha]
    cat_scores = (sig.pivot_table(index='neuron_uid', columns='category',
                                   values='effect_size', aggfunc='max')
                  .reindex(index=identity.index, columns=categories)
                  .fillna(0.0))

    dominant = cat_scores.idxmax(axis=1)
    dominant[cat_scores.max(axis=1) == 0] = 'untuned'

    summary = identity.join(cat_scores)
    summary['dominant_category'] = dominant
    summary['selectivity_breadth'] = (cat_scores > 0).sum(axis=1)
    return summary.reset_index()


def build_neuron_profile_matrix(tuning_df, identity_cols=IDENTITY_COLS):
    """
    Pivot a long-format tuning table to one row per neuron x one column per
    (variable, comparison), values = effect_size -- each neuron's full tuning
    "fingerprint", for clustering. Missing combinations (e.g. an outcome pair
    a session has ~zero trials for) are filled with 0.

    Returns
    -------
    profile  : DataFrame, index=neuron_uid, columns='variable: comparison'
    identity : DataFrame, index=neuron_uid (same order as profile), the
               identity_cols (psv_W, group, hemisphere, ...)
    """
    df = tuning_df.copy()
    df['column'] = df['variable'] + ': ' + df['comparison']
    profile = df.pivot_table(index='neuron_uid', columns='column', values='effect_size', aggfunc='max')
    profile = profile.fillna(0.0)

    identity = (df[[c for c in identity_cols if c in df.columns]]
                .drop_duplicates(subset='neuron_uid')
                .set_index('neuron_uid')
                .loc[profile.index])
    return profile, identity


def cluster_neuron_profiles(profile, n_clusters=4, random_state=0):
    """
    K-means clustering of neuron tuning profiles (see
    build_neuron_profile_matrix). Returns a Series of integer cluster labels
    (0..n_clusters-1), indexed the same as `profile` (neuron_uid), named
    'cluster'.
    """
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(profile.to_numpy())
    return pd.Series(labels, index=profile.index, name='cluster')
