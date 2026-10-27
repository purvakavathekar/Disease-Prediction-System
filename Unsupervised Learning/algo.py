"""
K-Means Algorithm Exploration Module (unsupervised)
Thin convenience layer over src/kmeans_clustering.py for experimenting with
the number of clusters and checking cluster quality on the datasets.
Note: results are for educational comparison only and do NOT constitute
clinical validation or medical diagnosis accuracy.
"""
import os
import sys

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import config  # noqa: E402
import kmeans_clustering as kc  # noqa: E402
from train_and_evaluate import apply_clinical_noise  # noqa: E402

# Robust paths for Linux/Render
_train_path = os.path.join(BASE_DIR, 'datasets', 'Training.csv')
if not os.path.exists(_train_path):
    _train_path = os.path.join(BASE_DIR, 'Datasets', 'Training.csv')
TRAIN_CSV = _train_path

_test_path = os.path.join(BASE_DIR, 'datasets', 'Testing.csv')
if not os.path.exists(_test_path):
    _test_path = os.path.join(BASE_DIR, 'Datasets', 'Testing.csv')
TEST_CSV = _test_path


def get_training_matrix():
    """Returns (X, feature_names) from Training.csv; the label column is dropped."""
    df = pd.read_csv(TRAIN_CSV)
    cols = [c for c in df.columns if c != 'prognosis' and not c.startswith('Unnamed')]
    return df[cols].values.astype(float), cols


def predict_cluster(symptoms):
    """Assigns a 132-binary symptom vector to its nearest saved k-means cluster."""
    return kc.assign_cluster(symptoms)


def predict(symptoms):
    """Returns the condition most associated with the vector's cluster."""
    assoc = predict_cluster(symptoms)['associated_conditions']
    return assoc[0]['condition'] if assoc else 'Undetermined'


def sweep_k(k_values=None):
    """Silhouette / Davies-Bouldin / inertia for several values of k."""
    X, _ = get_training_matrix()
    return kc.select_k(X, k_values)


def evaluate_kmeans(simulate_clinical_noise=True):
    """
    Quality of the saved model plus how stable held-out Testing.csv assignments are
    when symptoms are randomly omitted (simulated patient under-reporting).
    """
    model, profiles, features = kc.load_artifacts()
    X, _ = get_training_matrix()
    quality = kc.cluster_quality(model, X)

    test = pd.read_csv(TEST_CSV)
    X_test = test[features].values.astype(float)
    clean = model.predict(X_test)
    result = {'quality': quality, 'n_clusters': int(model.n_clusters)}
    if simulate_clinical_noise:
        noisy = model.predict(apply_clinical_noise(X_test))
        result['assignment_stability_under_noise'] = float((clean == noisy).mean())
    return result


if __name__ == '__main__':
    print(evaluate_kmeans())
