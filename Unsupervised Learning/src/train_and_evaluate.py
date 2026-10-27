"""
K-Means Training & Evaluation Pipeline (unsupervised)

1. Load the symptom matrix (features only, labels are NOT used for fitting).
2. Select k with the silhouette score over a sweep (elbow + silhouette charts).
3. Fit k-means (k-means++), build per-cluster profiles.
4. Report unsupervised quality metrics (inertia, silhouette, Davies-Bouldin).
5. Report descriptive agreement between clusters and the recorded conditions
   (purity / ARI / NMI) and assignment stability under simulated symptom noise.
6. Save model, profiles, metrics and charts.

Run:  python src/train_and_evaluate.py
"""
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

try:
    from . import config, dashboard_charts, kmeans_clustering as kc
except ImportError:
    import config
    import dashboard_charts
    import kmeans_clustering as kc


def apply_clinical_noise(X, drop_rate=0.32, add_rate=0.018, random_state=41):
    """Simulates patients forgetting symptoms (drop) or over-reporting (add)."""
    rng = np.random.RandomState(random_state)
    X_noisy = X.values.copy() if hasattr(X, 'values') else np.array(X, copy=True)
    X_noisy[(X_noisy == 1) & (rng.rand(*X_noisy.shape) < drop_rate)] = 0
    X_noisy[(X_noisy == 0) & (rng.rand(*X_noisy.shape) < add_rate)] = 1
    return X_noisy


def _purity(labels, reference):
    total = 0
    for c in np.unique(labels):
        total += pd.Series(reference[labels == c]).value_counts().iloc[0]
    return total / len(labels)


def run_pipeline(skip_k_sweep=False):
    print("=" * 70)
    print("K-MEANS SYMPTOM CLUSTERING: TRAINING & EVALUATION")
    print("=" * 70)

    data_path = config.TRAIN_DATASET_PATH
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at '{data_path}'.")

    print(f"[1/6] Loading dataset: {data_path}")
    df = pd.read_csv(data_path)
    feature_cols = [c for c in df.columns if c != 'prognosis' and not c.startswith('Unnamed')]
    X = df[feature_cols].values.astype(float)
    reference = df['prognosis'].astype(str).str.strip().values  # descriptive only
    print(f"      {X.shape[0]} records, {X.shape[1]} binary symptom features (labels not used for fitting)")

    # K selection
    if skip_k_sweep:
        best_k, sweep = config.DEFAULT_K, []
        print(f"[2/6] Skipping sweep, using k = {best_k}")
    else:
        ks = list(config.K_SEARCH_RANGE)
        print(f"[2/6] Selecting k by silhouette over k = {ks[0]}..{ks[-1]}")
        best_k, sweep = kc.select_k(X, ks)
        print(f"      Best k = {best_k}")

    # Fit
    print(f"[3/6] Fitting k-means with k = {best_k}")
    model = kc.fit_kmeans(X, best_k)
    profiles = kc.build_cluster_profiles(model, X, feature_cols, descriptive_labels=reference)
    quality = kc.cluster_quality(model, X)
    print(f"      inertia={quality['inertia']:.1f}  silhouette={quality['silhouette']:.4f}  "
          f"davies_bouldin={quality['davies_bouldin']:.4f}  iterations={quality['n_iter']}")

    # Descriptive agreement with recorded conditions
    print("[4/6] Comparing clusters with recorded conditions (descriptive only)")
    labels = model.labels_
    purity = _purity(labels, reference)
    ari = float(adjusted_rand_score(reference, labels))
    nmi = float(normalized_mutual_info_score(reference, labels))
    print(f"      purity={purity:.4f}  ARI={ari:.4f}  NMI={nmi:.4f}")

    # Held-out assignment test + noise stability
    print("[5/6] Assigning unseen records and testing stability under symptom noise")
    test_path = config.TEST_DATASET_PATH
    held_out = {}
    if os.path.exists(test_path):
        tdf = pd.read_csv(test_path)
        X_test = tdf[feature_cols].values.astype(float)
        y_test = tdf['prognosis'].astype(str).str.strip().values
        clean_assign = model.predict(X_test)
        noisy_assign = model.predict(apply_clinical_noise(X_test))

        dominant = {p['cluster_id']: p['dominant_condition'] for p in profiles}
        cond_match = float(np.mean([dominant.get(int(c)) == t for c, t in zip(clean_assign, y_test)]))
        cond_match_noisy = float(np.mean([dominant.get(int(c)) == t for c, t in zip(noisy_assign, y_test)]))
        stability = float(np.mean(clean_assign == noisy_assign))
        held_out = {
            'test_records': int(len(X_test)),
            'condition_agreement_clean': round(cond_match, 4),
            'condition_agreement_noisy': round(cond_match_noisy, 4),
            'assignment_stability_under_noise': round(stability, 4),
        }
        print(f"      test records={len(X_test)}  agreement(clean)={cond_match:.4f}  "
              f"agreement(noisy)={cond_match_noisy:.4f}  stability={stability:.4f}")
    else:
        print("      Testing.csv not found, skipping")

    # Save
    print("[6/6] Saving model, metrics and charts")
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    kc.save_artifacts(model, profiles, feature_cols)
    kc.reset_cache()

    sizes = [p['size'] for p in profiles]
    metrics = {
        'algorithm': 'KMeans (k-means++)',
        'learning_type': 'Unsupervised',
        'n_clusters': int(best_k),
        'n_records': int(X.shape[0]),
        'n_features': int(X.shape[1]),
        'inertia': round(quality['inertia'], 2),
        'silhouette': round(quality['silhouette'], 4),
        'davies_bouldin': round(quality['davies_bouldin'], 4),
        'iterations': quality['n_iter'],
        'smallest_cluster': int(min(sizes)),
        'largest_cluster': int(max(sizes)),
        'mean_cluster_size': round(float(np.mean(sizes)), 2),
        'cluster_purity_vs_recorded_conditions': round(float(purity), 4),
        'adjusted_rand_index': round(ari, 4),
        'normalized_mutual_info': round(nmi, 4),
        'held_out': held_out,
        'k_selection': 'silhouette sweep' if sweep else 'default',
        'random_state': config.RANDOM_STATE,
        'dataset_source': os.path.basename(data_path),
        'evaluated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'clinical_disclaimer': config.CLINICAL_DISCLAIMER,
    }
    with open(config.METRICS_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    with open(config.META_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump({
            'model_name': 'Symptom K-Means Clustering',
            'algorithm': type(model).__name__,
            'framework': 'scikit-learn',
            'features': feature_cols,
            'hyperparameters': model.get_params(),
            'k_sweep': sweep,
            'clinical_notice': config.CLINICAL_DISCLAIMER,
        }, f, indent=2)

    rows = [
        ('Clusters (k)', best_k, 'Number of k-means clusters'),
        ('Silhouette score', f"{quality['silhouette']:.4f}", 'Cohesion vs separation, -1 to 1 (higher is better)'),
        ('Davies-Bouldin index', f"{quality['davies_bouldin']:.4f}", 'Cluster overlap (lower is better)'),
        ('Inertia', f"{quality['inertia']:.1f}", 'Within-cluster sum of squared distances'),
        ('Cluster purity', f"{purity:.4f}", 'Descriptive: share of records in the dominant recorded condition of their cluster'),
        ('Adjusted Rand index', f"{ari:.4f}", 'Descriptive: agreement with recorded conditions'),
        ('Normalized mutual information', f"{nmi:.4f}", 'Descriptive: information shared with recorded conditions'),
        ('Smallest / largest cluster', f"{min(sizes)} / {max(sizes)}", 'Records per cluster'),
    ]
    if held_out:
        rows += [
            ('Held-out assignment stability', f"{held_out['assignment_stability_under_noise']:.4f}", 'Same cluster before/after simulated symptom omission'),
            ('Held-out condition agreement', f"{held_out['condition_agreement_clean']:.4f}", 'Descriptive: cluster dominant condition equals recorded condition'),
        ]
    pd.DataFrame(rows, columns=['Metric', 'Value', 'Description']).to_csv(config.METRICS_TABLE_PATH, index=False)

    report_rows = []
    for p in profiles:
        report_rows.append({
            'cluster_id': p['cluster_id'],
            'size': p['size'],
            'share': p['share'],
            'dominant_condition': p['dominant_condition'],
            'dominant_share': p['dominant_share'],
            'top_symptoms': ', '.join(s['symptom'] for s in p['top_symptoms'][:5]),
        })
    pd.DataFrame(report_rows).to_csv(config.CLUSTER_REPORT_PATH, index=False)

    if sweep:
        dashboard_charts.generate_k_selection_chart(sweep, best_k)
    dashboard_charts.generate_cluster_size_chart(profiles)
    dashboard_charts.generate_pca_scatter(model, X)

    print("=" * 70)
    print("DONE")
    print("=" * 70)
    return metrics


if __name__ == '__main__':
    run_pipeline(skip_k_sweep='--skip-sweep' in sys.argv)
