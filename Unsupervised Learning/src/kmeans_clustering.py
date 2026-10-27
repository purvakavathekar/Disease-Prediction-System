"""
K-Means Symptom Clustering Module (unsupervised learning)

Groups 132-dimensional binary symptom vectors into clusters with k-means.
No labels are used to fit the model. The `prognosis` column is only used
afterwards, as descriptive metadata, to report which recorded conditions
occur inside each discovered cluster.

Public API
----------
fit_kmeans(X, k)                 -> fitted sklearn KMeans
select_k(X, k_values)            -> (best_k, sweep_results)
build_cluster_profiles(...)      -> per-cluster summary (size, top symptoms, conditions)
save_artifacts / load_artifacts  -> persist and reload the model + profiles
assign_cluster(vector)           -> cluster assignment for one symptom vector
"""
import json
import os

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score

try:
    from . import config
except ImportError:  # executed as a script from src/
    import config


# --------------------------------------------------------------------------
# Training helpers
# --------------------------------------------------------------------------

def fit_kmeans(X, k=None, random_state=None):
    """Fit k-means (k-means++ initialisation, several restarts) on X."""
    k = k or config.DEFAULT_K
    model = KMeans(
        n_clusters=int(k),
        init='k-means++',
        n_init=config.KMEANS_N_INIT,
        max_iter=config.KMEANS_MAX_ITER,
        random_state=config.RANDOM_STATE if random_state is None else random_state,
    )
    model.fit(np.asarray(X, dtype=float))
    return model


def select_k(X, k_values=None):
    """
    Choose the number of clusters with the silhouette score (no labels needed).
    Returns (best_k, sweep) where sweep is a list of dicts with inertia,
    silhouette and Davies-Bouldin for every k tried.
    """
    X = np.asarray(X, dtype=float)
    k_values = list(k_values or config.K_SEARCH_RANGE)
    sweep = []
    for k in k_values:
        model = fit_kmeans(X, k)
        labels = model.labels_
        sweep.append({
            'k': int(k),
            'inertia': float(model.inertia_),
            'silhouette': float(silhouette_score(X, labels)),
            'davies_bouldin': float(davies_bouldin_score(X, labels)),
        })
    best = max(sweep, key=lambda r: r['silhouette'])
    return best['k'], sweep


def build_cluster_profiles(model, X, feature_names, descriptive_labels=None, top_symptoms=8):
    """
    Summarise every cluster:
      * size and share of the data
      * most characteristic symptoms (highest share of members reporting them)
      * conditions recorded for its members (descriptive only; labels were
        never used to build the clusters)
    """
    X = np.asarray(X, dtype=float)
    labels = model.labels_
    labels_arr = None if descriptive_labels is None else np.asarray(descriptive_labels)
    profiles = []

    for c in range(model.n_clusters):
        members = labels == c
        size = int(members.sum())
        if size == 0:
            profiles.append({
                'cluster_id': c, 'size': 0, 'share': 0.0,
                'top_symptoms': [], 'conditions': [], 'dominant_condition': None,
                'dominant_share': 0.0,
            })
            continue

        freq = X[members].mean(axis=0)
        order = np.argsort(freq)[::-1][:top_symptoms]
        symptoms = [
            {'symptom': feature_names[i], 'prevalence': round(float(freq[i]), 4)}
            for i in order if freq[i] > 0
        ]

        conditions = []
        if labels_arr is not None:
            values, counts = np.unique(labels_arr[members], return_counts=True)
            for j in np.argsort(counts)[::-1]:
                conditions.append({
                    'condition': str(values[j]).strip(),
                    'share': round(float(counts[j]) / size, 4),
                    'count': int(counts[j]),
                })

        profiles.append({
            'cluster_id': c,
            'size': size,
            'share': round(size / len(X), 4),
            'top_symptoms': symptoms,
            'conditions': conditions,
            'dominant_condition': conditions[0]['condition'] if conditions else None,
            'dominant_share': conditions[0]['share'] if conditions else 0.0,
        })
    return profiles


def cluster_quality(model, X):
    """Unsupervised quality metrics for a fitted model."""
    X = np.asarray(X, dtype=float)
    labels = model.labels_
    return {
        'inertia': float(model.inertia_),
        'silhouette': float(silhouette_score(X, labels)),
        'davies_bouldin': float(davies_bouldin_score(X, labels)),
        'n_clusters': int(model.n_clusters),
        'n_iter': int(model.n_iter_),
    }


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------

def save_artifacts(model, profiles, feature_names, model_path=None, profiles_path=None):
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    joblib.dump(model, model_path or config.KMEANS_MODEL_PATH)
    with open(profiles_path or config.CLUSTER_PROFILES_PATH, 'w', encoding='utf-8') as f:
        json.dump({'features': list(feature_names), 'clusters': profiles}, f, indent=2)


_CACHE = {'model': None, 'profiles': None, 'features': None}


def load_artifacts(model_path=None, profiles_path=None):
    """Load (and cache) the fitted k-means model and its cluster profiles."""
    if _CACHE['model'] is not None and model_path is None and profiles_path is None:
        return _CACHE['model'], _CACHE['profiles'], _CACHE['features']

    m_path = model_path or config.KMEANS_MODEL_PATH
    p_path = profiles_path or config.CLUSTER_PROFILES_PATH
    if not os.path.exists(m_path) or not os.path.exists(p_path):
        raise FileNotFoundError(
            "K-means artifacts not found. Run 'python src/train_and_evaluate.py' to "
            f"create '{m_path}' and '{p_path}'."
        )

    model = joblib.load(m_path)
    with open(p_path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    profiles = {c['cluster_id']: c for c in payload['clusters']}
    features = payload['features']

    if model_path is None and profiles_path is None:
        _CACHE.update(model=model, profiles=profiles, features=features)
    return model, profiles, features


def reset_cache():
    _CACHE.update(model=None, profiles=None, features=None)


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------

def _soft_memberships(sq_dists, temperature):
    """Softmax over negative squared distances (nearest centroid dominates)."""
    shifted = sq_dists - sq_dists.min()
    w = np.exp(-shifted / max(temperature, 1e-6))
    return w / w.sum()


def assign_cluster(vector, top_clusters=None, temperature=None):
    """
    Assign one symptom vector to its nearest k-means cluster.

    Returns a dict with:
      cluster_id, cluster_size, distance, membership (0-100 %),
      cluster_symptoms (characteristic symptoms of the cluster),
      nearest_clusters (top-N clusters with membership),
      associated_conditions (ranked conditions, blended over nearby clusters),
      matched_symptoms / cluster_overlap (how many reported symptoms the
      cluster typically contains).
    """
    top_clusters = top_clusters or config.TOP_CLUSTERS
    temperature = temperature or config.MEMBERSHIP_TEMPERATURE

    model, profiles, features = load_artifacts()
    x = np.asarray(vector, dtype=float).reshape(1, -1)
    if x.shape[1] != model.cluster_centers_.shape[1]:
        raise ValueError(
            f"Expected {model.cluster_centers_.shape[1]} symptom features, got {x.shape[1]}."
        )

    sq = ((model.cluster_centers_ - x) ** 2).sum(axis=1)
    memberships = _soft_memberships(sq, temperature)
    order = np.argsort(sq)[:top_clusters]
    best = int(order[0])

    nearest = []
    for c in order:
        prof = profiles.get(int(c), {})
        nearest.append({
            'cluster_id': int(c),
            'distance': round(float(np.sqrt(sq[c])), 4),
            'membership': round(float(memberships[c]) * 100, 1),
            'dominant_condition': prof.get('dominant_condition'),
        })

    # Blend the condition mix of the nearest clusters, weighted by membership
    scores = {}
    weights = memberships[order]
    weights = weights / weights.sum()
    for w, c in zip(weights, order):
        for cond in profiles.get(int(c), {}).get('conditions', []):
            scores[cond['condition']] = scores.get(cond['condition'], 0.0) + w * cond['share']
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    associated = [
        {'condition': name, 'probability': round(float(sc) * 100, 1)}
        for name, sc in ranked[:3]
    ]

    best_profile = profiles.get(best, {})
    active = [features[i] for i, v in enumerate(np.asarray(vector).ravel()) if v]
    cluster_symptom_names = {s['symptom'] for s in best_profile.get('top_symptoms', [])}
    overlap = [s for s in active if s in cluster_symptom_names]

    return {
        'cluster_id': best,
        'cluster_size': best_profile.get('size', 0),
        'distance': round(float(np.sqrt(sq[best])), 4),
        'membership': round(float(memberships[best]) * 100, 1),
        'cluster_symptoms': best_profile.get('top_symptoms', []),
        'nearest_clusters': nearest,
        'associated_conditions': associated,
        'matched_symptoms': overlap,
        'n_active_symptoms': len(active),
    }
