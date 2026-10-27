"""
Test Suite for the K-Means Symptom Clustering Pipeline (unsupervised)
Validates:
1. Dataset dimensions and 132 binary symptom features.
2. K-means artifacts (model, cluster profiles, metrics, charts) exist and are valid.
3. Cluster quality metrics and reproducibility of the fit.
4. Cluster assignment interface (src/predict.py), red-flag emergency detection.
5. Flask dashboard routes (/evaluation, /api/evaluation/metrics, /reports/...).
6. No supervised learning estimators remain in the code base.
"""
import os
import sys
import json
import unittest
import numpy as np
import pandas as pd
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

SRC_DIR = os.path.join(BASE_DIR, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    from src import config, predict, dashboard_charts, kmeans_clustering
except ImportError:
    import config
    import predict
    import dashboard_charts
    import kmeans_clustering
from app import app


class TestKMeansPipeline(unittest.TestCase):

    def setUp(self):
        app.testing = True
        self.app = app.test_client()

    # ----------------------------------------------------
    # Dataset
    # ----------------------------------------------------
    def test_01_training_dataset_shape(self):
        """Dataset has 132 binary symptom features plus the descriptive prognosis column."""
        df = pd.read_csv(config.TRAIN_DATASET_PATH)
        feature_cols = [c for c in df.columns if c != 'prognosis' and not c.startswith('Unnamed')]
        self.assertEqual(len(feature_cols), 132)
        values = np.unique(df[feature_cols].values)
        self.assertTrue(set(values.tolist()).issubset({0, 1}), "Features must be binary")
        self.assertEqual(df['prognosis'].nunique(), 41)

    # ----------------------------------------------------
    # Artifacts and metrics
    # ----------------------------------------------------
    def test_02_all_artifacts_generated_and_valid(self):
        status = dashboard_charts.verify_dashboard_assets()
        self.assertTrue(status['all_present'], f"Missing required assets: {status['missing']}")
        for path in (config.KMEANS_MODEL_PATH, config.CLUSTER_PROFILES_PATH,
                     config.METRICS_JSON_PATH, config.META_JSON_PATH,
                     config.METRICS_TABLE_PATH, config.CLUSTER_REPORT_PATH):
            self.assertTrue(os.path.exists(path), path)

        for chart, min_w in ((config.CLUSTER_SCATTER_CHART_PATH, 500),
                             (config.CLUSTER_SIZES_CHART_PATH, 500),
                             (config.K_SELECTION_CHART_PATH, 500)):
            with Image.open(chart) as img:
                self.assertGreater(img.size[0], min_w)

    def test_03_cluster_quality_metrics(self):
        metrics = dashboard_charts.load_metrics()
        self.assertEqual(metrics['learning_type'], 'Unsupervised')
        self.assertGreater(metrics['n_clusters'], 1)
        self.assertGreater(metrics['silhouette'], 0.3)
        self.assertLess(metrics['davies_bouldin'], 1.5)
        self.assertEqual(metrics['n_features'], 132)
        self.assertIn("EDUCATIONAL & RESEARCH PROTOTYPE", metrics['clinical_disclaimer'])
        self.assertGreater(metrics['held_out']['assignment_stability_under_noise'], 0.5)

    def test_04_cluster_profiles_consistent(self):
        model, profiles, features = kmeans_clustering.load_artifacts()
        self.assertEqual(len(profiles), model.n_clusters)
        self.assertEqual(sum(p['size'] for p in profiles.values()), len(model.labels_))
        for p in profiles.values():
            self.assertGreater(p['size'], 0)
            self.assertTrue(p['top_symptoms'])

    def test_05_fit_is_reproducible_and_label_free(self):
        """Fitting only needs the feature matrix and the same seed gives the same centroids."""
        df = pd.read_csv(config.TRAIN_DATASET_PATH)
        cols = [c for c in df.columns if c != 'prognosis' and not c.startswith('Unnamed')]
        X = df[cols].values
        m1 = kmeans_clustering.fit_kmeans(X, 10)
        m2 = kmeans_clustering.fit_kmeans(X, 10)
        np.testing.assert_allclose(m1.cluster_centers_, m2.cluster_centers_)

    def test_06_no_supervised_estimators_in_code(self):
        """No supervised sklearn estimator should be referenced by project code."""
        banned = ['Random' + 'ForestClassifier', 'KNeighbors' + 'Classifier', 'Bernoulli' + 'NB',
                  'Logistic' + 'Regression', 'GradientBoosting' + 'Classifier', 'train_test' + '_split']
        skip_dirs = {'.venv', '__pycache__', '.git', 'node_modules'}
        offenders = []
        for root, dirs, files in os.walk(BASE_DIR):
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            for name in files:
                if not name.endswith('.py'):
                    continue
                with open(os.path.join(root, name), encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                for word in banned:
                    if word in text:
                        offenders.append((name, word))
        self.assertEqual(offenders, [])

    # ----------------------------------------------------
    # Dashboard and UI routes
    # ----------------------------------------------------
    def test_07_flask_evaluation_dashboard_route(self):
        response = self.app.get('/evaluation')
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("K-Means Clustering Evaluation Dashboard", html)
        self.assertIn("Silhouette Score", html)
        self.assertIn("kmeans_pca_scatter.png", html)
        self.assertIn("kmeans_k_selection.png", html)
        self.assertNotIn("Confusion Matrix", html)

    def test_08_flask_reports_serving_route(self):
        for name, ctype in (('kmeans_pca_scatter.png', 'image/png'),
                            ('kmeans_cluster_sizes.png', 'image/png'),
                            ('kmeans_k_selection.png', 'image/png')):
            with self.app.get(f'/reports/{name}') as resp:
                self.assertEqual(resp.status_code, 200)
                self.assertEqual(resp.content_type, ctype)
        with self.app.get('/reports/kmeans_metrics_table.csv') as resp:
            self.assertEqual(resp.status_code, 200)

    def test_09_flask_metrics_api_endpoint(self):
        response = self.app.get('/api/evaluation/metrics')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.get_data(as_text=True))
        self.assertEqual(data.get('algorithm'), 'KMeans (k-means++)')
        self.assertGreater(data.get('silhouette', 0), 0.3)

    # ----------------------------------------------------
    # Predict interface
    # ----------------------------------------------------
    def test_10_predict_interface_binary_vector(self):
        vec = [0] * 132
        vec[0] = 1  # itching
        vec[1] = 1  # skin_rash
        self.assertEqual(predict.predict(vec), 'Fungal infection')

    def test_11_predict_interface_symptom_names(self):
        res = predict.predict_triage(['chest_pain', 'breathlessness', 'sweating'])
        self.assertEqual(res['status'], 'success')
        self.assertTrue(res['is_emergency'])
        self.assertTrue(res['has_red_flags'])
        self.assertEqual(len(res['top_3']), 3)
        self.assertGreaterEqual(res['top_3'][0]['probability'], 50.0)
        self.assertIn('cluster_id', res)
        self.assertIn("CRITICAL MEDICAL ADVISORY", res['emergency_advisory'])

    def test_12_predict_interface_invalid_input_error_handling(self):
        with self.assertRaises(ValueError):
            predict.predict(12345)

    def test_13_assign_cluster_rejects_wrong_length(self):
        with self.assertRaises(ValueError):
            kmeans_clustering.assign_cluster([1, 0, 1])


if __name__ == '__main__':
    unittest.main()
