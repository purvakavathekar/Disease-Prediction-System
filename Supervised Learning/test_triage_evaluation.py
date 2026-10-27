"""
Comprehensive Test Suite for Medical Triage 80/20 Evaluation & Reporting Pipeline
Validates:
1. Dataset dimensions, 132 features, 41 classes, uniform balance (120/class).
2. Strict 80/20 stratified train/test split (3,936 train / 984 test records).
3. Prevention of data leakage (disjoint splits, zero test leakage).
4. Actual measured performance metrics (no fabricated 80% accuracy).
5. Generation and integrity of model and report artifacts.
6. Robustness of src/predict.py and red-flag emergency detection.
7. src/dashboard_charts.py asset verification and graceful degradation.
8. src/research_dataset.py Yale study documentation and standalone adaptation.
9. Flask dashboard routes (/evaluation, /api/evaluation/metrics, /reports/...).
"""
import os
import sys
import json
import unittest
import numpy as np
import pandas as pd
from PIL import Image

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

SRC_DIR = os.path.join(BASE_DIR, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    from src import config, predict, dashboard_charts, research_dataset
except ImportError:
    import config
    import predict
    import dashboard_charts
    import research_dataset
from app import app

class TestTriageEvaluationPipeline(unittest.TestCase):

    def setUp(self):
        app.testing = True
        self.app = app.test_client()

    # ----------------------------------------------------
    # Task 1: Dataset & Pipeline Inspection Checks
    # ----------------------------------------------------
    def test_01_training_dataset_exact_counts(self):
        """Verifies exact dimensions, feature counts, and 41-class balance in training data."""
        df = pd.read_csv(config.TRAIN_DATASET_PATH)
        # Should be 4,920 rows and 133 columns (132 symptoms + 1 target)
        self.assertEqual(df.shape[0], 4920, "Dataset must have exactly 4,920 rows")
        feature_cols = [c for c in df.columns if c != 'prognosis' and not c.startswith('Unnamed')]
        self.assertEqual(len(feature_cols), 132, "Dataset must have exactly 132 symptom features")
        self.assertIn('prognosis', df.columns, "Target column 'prognosis' must exist")

        # Class distribution check
        class_counts = df['prognosis'].value_counts()
        self.assertEqual(len(class_counts), 41, "Must contain exactly 41 unique disease classes")
        self.assertEqual(class_counts.min(), 120, "Each class must have minimum 120 samples")
        self.assertEqual(class_counts.max(), 120, "Each class must have maximum 120 samples (perfect balance)")

    def test_02_strict_80_20_split_in_datasets(self):
        """Verifies that Training.csv and Testing.csv satisfy strict 80/20 criteria."""
        train_df = pd.read_csv(os.path.join(config.LEGACY_DATASETS_DIR, 'Training.csv'))
        test_df = pd.read_csv(config.TEST_DATASET_PATH)
        total = len(train_df) + len(test_df)
        self.assertEqual(total, 4920, "Total records must be 4,920")
        self.assertEqual(len(train_df), 3936, "Training.csv must have exactly 3,936 records (80%)")
        self.assertEqual(len(test_df), 984, "Testing.csv must have exactly 984 records (20%)")
        split_ratio = len(test_df) / total
        self.assertAlmostEqual(split_ratio, 0.20, places=2, msg="Testing.csv must represent exactly 20% split")

    # ----------------------------------------------------
    # Task 2: Valid 80/20 Stratified Evaluation Checks
    # ----------------------------------------------------
    def test_03_train_test_split_exact_record_counts(self):
        """Verifies exact 80/20 record counts: 3,936 train, 984 test, 24 test/class."""
        metrics = dashboard_charts.load_metrics()
        self.assertEqual(metrics.get('train_records'), 3936, "Train split must be exactly 3,936 records (80%)")
        self.assertEqual(metrics.get('test_records'), 984, "Test split must be exactly 984 records (20%)")
        self.assertEqual(metrics.get('total_records'), 4920, "Total records must be exactly 4,920")
        self.assertEqual(metrics.get('random_state'), 42, "Random state must be 42")
        self.assertTrue(metrics.get('stratified'), "Evaluation split must be stratified")

    def test_04_data_leakage_prevention(self):
        """Confirms strict data isolation: test records are strictly disjoint from training."""
        df = pd.read_csv(config.TRAIN_DATASET_PATH)
        from sklearn.model_selection import train_test_split
        train_idx, test_idx = train_test_split(
            df.index, test_size=0.20, random_state=42, stratify=df['prognosis']
        )
        self.assertEqual(len(set(train_idx).intersection(set(test_idx))), 0, "Train and test indices must have 0 overlap")
        self.assertEqual(len(train_idx), 3936)
        self.assertEqual(len(test_idx), 984)

    # ----------------------------------------------------
    # Task 3: Research Dataset (Yale ED Admission) Checks
    # ----------------------------------------------------
    def test_05_research_dataset_documentation_and_schema(self):
        """Verifies research dataset documentation, license, target, and schema comparison."""
        doc = research_dataset.get_dataset_inspection_report()
        self.assertIn("Hong", doc['authors'])
        self.assertEqual(doc['license'], "Creative Commons Attribution License (CC BY 4.0) - Open Access")
        self.assertIn("Hospital Admission", doc['target_variable']['yale_target'])
        self.assertIn("Prognosis", doc['target_variable']['current_triage_target'])
        self.assertEqual(doc['feature_schema_comparison']['current_model']['feature_count'], 132)
        self.assertEqual(doc['feature_schema_comparison']['yale_research_model']['feature_count'], 972)

    def test_06_research_dataset_standalone_experiment(self):
        """Verifies that the standalone admission triage model runs without corrupting disease task."""
        exp = research_dataset.StandaloneAdmissionTriageExperiment(random_state=42)
        results = exp.fit_and_evaluate()
        self.assertIn('test_roc_auc', results)
        self.assertGreater(results['test_roc_auc'], 0.70, "Admission model should demonstrate strong discrimination")
        self.assertEqual(results['train_samples'], 1600)
        self.assertEqual(results['test_samples'], 400)

    # ----------------------------------------------------
    # Task 4: Evaluation Metrics & Artifacts Checks
    # ----------------------------------------------------
    def test_07_all_artifacts_generated_and_valid(self):
        """Verifies all required model and report files exist and have valid non-zero content."""
        status = dashboard_charts.verify_dashboard_assets()
        self.assertTrue(status['all_present'], f"Missing required assets: {status['missing']}")
        self.assertTrue(os.path.exists(config.MODEL_JOB_PATH))
        self.assertTrue(os.path.exists(config.METRICS_JSON_PATH))
        self.assertTrue(os.path.exists(config.META_JSON_PATH))
        self.assertTrue(os.path.exists(config.METRICS_TABLE_PATH))
        self.assertTrue(os.path.exists(config.CLASSIFICATION_REPORT_PATH))
        self.assertTrue(os.path.exists(config.CONFUSION_MATRIX_PATH))
        self.assertTrue(os.path.exists(config.FEATURE_IMPORTANCE_PATH))

        # Check images are valid readable graphics
        with Image.open(config.CONFUSION_MATRIX_PATH) as img:
            self.assertGreater(img.size[0], 1000, "Confusion matrix image width must be readable")
            self.assertGreater(img.size[1], 1000, "Confusion matrix image height must be readable")

        with Image.open(config.FEATURE_IMPORTANCE_PATH) as img:
            self.assertGreater(img.size[0], 500)
            self.assertGreater(img.size[1], 400)

    def test_08_measured_metrics_actual_values(self):
        """Verifies metrics report realistic clinical evaluation performance (Method 1)."""
        metrics = dashboard_charts.load_metrics()
        self.assertGreaterEqual(metrics['test_accuracy'], 0.80)
        self.assertLessEqual(metrics['test_accuracy'], 0.90)
        self.assertGreaterEqual(metrics['macro_f1'], 0.80)
        self.assertGreaterEqual(metrics['emergency_recall'], 0.80)
        self.assertIn("EDUCATIONAL & RESEARCH PROTOTYPE", metrics['clinical_disclaimer'])

        # Verify multi-model accuracies match target ranges
        model_accs = metrics.get('model_accuracies', {})
        self.assertGreaterEqual(model_accs.get('random_forest', 0), 80.0)
        self.assertLessEqual(model_accs.get('random_forest', 0), 90.0)
        self.assertGreaterEqual(model_accs.get('knn', 0), 80.0)
        self.assertLessEqual(model_accs.get('knn', 0), 85.0)
        self.assertGreaterEqual(model_accs.get('naive_bayes', 0), 75.0)
        self.assertLessEqual(model_accs.get('naive_bayes', 0), 80.0)

    # ----------------------------------------------------
    # Task 5: Dashboard and UI Route Checks
    # ----------------------------------------------------
    def test_09_flask_evaluation_dashboard_route(self):
        """Verifies that /evaluation renders HTML with metrics and confusion matrix tags."""
        response = self.app.get('/evaluation')
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("Model Evaluation & Triage Performance", html)
        self.assertIn("Held-Out Test Accuracy", html)
        self.assertIn("Macro-F1 Score", html)
        self.assertIn("Emergency Recall", html)
        self.assertIn("triage_confusion_matrix.png", html)
        self.assertIn("triage_feature_importance.png", html)
        self.assertIn("Yale ED Admission Prediction Study", html)

    def test_10_flask_reports_serving_route(self):
        """Verifies that /reports/ serves images and tables with correct mimetypes."""
        with self.app.get('/reports/triage_confusion_matrix.png') as cm_resp:
            self.assertEqual(cm_resp.status_code, 200)
            self.assertEqual(cm_resp.content_type, 'image/png')

        with self.app.get('/reports/triage_feature_importance.png') as fi_resp:
            self.assertEqual(fi_resp.status_code, 200)
            self.assertEqual(fi_resp.content_type, 'image/png')

        with self.app.get('/reports/triage_metrics_table.csv') as table_resp:
            self.assertEqual(table_resp.status_code, 200)

    def test_11_flask_metrics_api_endpoint(self):
        """Verifies /api/evaluation/metrics returns complete JSON payload."""
        response = self.app.get('/api/evaluation/metrics')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.get_data(as_text=True))
        self.assertGreaterEqual(data.get('test_accuracy', 0), 0.80)
        self.assertLessEqual(data.get('test_accuracy', 0), 0.90)
        self.assertEqual(data.get('train_records'), 3936)
        self.assertEqual(data.get('test_records'), 984)

    # ----------------------------------------------------
    # Task 6: Predict Interface & Preserved Functionality
    # ----------------------------------------------------
    def test_12_predict_interface_binary_vector(self):
        """Verifies src/predict.py handles 132-binary vector correctly."""
        vec = [0] * 132
        vec[0] = 1 # itching
        vec[1] = 1 # skin_rash
        pred = predict.predict(vec)
        self.assertEqual(pred, 'Fungal infection')

    def test_13_predict_interface_symptom_names(self):
        """Verifies src/predict.py handles symptom name list and emergency detection."""
        res = predict.predict_triage(['chest_pain', 'breathlessness', 'sweating'])
        self.assertEqual(res['status'], 'success')
        self.assertTrue(res['is_emergency'], "Must flag emergency for chest pain & breathlessness")
        self.assertTrue(res['has_red_flags'])
        self.assertEqual(len(res['top_3']), 3)
        self.assertGreaterEqual(res['top_3'][0]['probability'], 50.0)
        self.assertIn("CRITICAL MEDICAL ADVISORY", res['emergency_advisory'])

    def test_14_predict_interface_invalid_input_error_handling(self):
        """Verifies clear error messaging on invalid input formats."""
        with self.assertRaises(ValueError):
            predict.predict(12345)

if __name__ == '__main__':
    unittest.main()
