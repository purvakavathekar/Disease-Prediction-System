"""
AI/ML Medical Triage Configuration Module
Defines canonical dataset paths, model paths, report artifact paths,
reproducible split parameters, and emergency triage category definitions.
"""
import os

# Identify directory containing this file
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
WORKSPACE_ROOT = PROJECT_ROOT

# Data Paths
DATASETS_DIR = os.path.join(PROJECT_ROOT, 'Datasets')
LEGACY_DATASETS_DIR = os.path.join(PROJECT_ROOT, 'datasets')

TRAIN_DATASET_PATH = os.path.join(DATASETS_DIR, 'triage_train.csv')
if not os.path.exists(TRAIN_DATASET_PATH):
    TRAIN_DATASET_PATH = os.path.join(LEGACY_DATASETS_DIR, 'Training.csv')

TEST_DATASET_PATH = os.path.join(LEGACY_DATASETS_DIR, 'Testing.csv')

# Models Paths
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
MODEL_JOB_PATH = os.path.join(MODELS_DIR, 'triage_model.joblib')
KNN_MODEL_PATH = os.path.join(MODELS_DIR, 'knn_model.joblib')
NAIVE_BAYES_MODEL_PATH = os.path.join(MODELS_DIR, 'naive_bayes_model.joblib')
LEGACY_MODEL_PATH = os.path.join(PROJECT_ROOT, 'model')
METRICS_JSON_PATH = os.path.join(MODELS_DIR, 'triage_metrics.json')
META_JSON_PATH = os.path.join(MODELS_DIR, 'triage_meta.json')

# Available Machine Learning Models for Diagnosis Selection
AVAILABLE_MODELS = {
    'random_forest': {
        'id': 'random_forest',
        'name': 'Random Forest Classifier',
        'short_name': 'Random Forest',
        'description': 'Ensemble of 100 decision trees. High accuracy, stable generalization, and resistant to symptom noise.',
        'badge': 'Recommended',
        'icon': 'fas fa-tree',
        'is_recommended': True
    },
    'knn': {
        'id': 'knn',
        'name': 'K-Nearest Neighbors',
        'short_name': 'KNN (k=41)',
        'description': 'Distance-based lazy learner (k=41). Classifies based on closest known symptom patterns in training data.',
        'badge': 'Instance-Based',
        'icon': 'fas fa-project-diagram',
        'is_recommended': False
    },
    'naive_bayes': {
        'id': 'naive_bayes',
        'name': 'Naive Bayes (Bernoulli NB)',
        'short_name': 'Naive Bayes',
        'description': 'Probabilistic binary classifier based on Bayes\' theorem with symptom independence assumption.',
        'badge': 'Probabilistic',
        'icon': 'fas fa-percentage',
        'is_recommended': False
    }
}
DEFAULT_MODEL = 'random_forest'

# Reports Paths
REPORTS_DIR = os.path.join(PROJECT_ROOT, 'reports')
CONFUSION_MATRIX_PATH = os.path.join(REPORTS_DIR, 'triage_confusion_matrix.png')
FEATURE_IMPORTANCE_PATH = os.path.join(REPORTS_DIR, 'triage_feature_importance.png')
METRICS_TABLE_PATH = os.path.join(REPORTS_DIR, 'triage_metrics_table.csv')
CLASSIFICATION_REPORT_PATH = os.path.join(REPORTS_DIR, 'triage_classification_report.csv')

# Ensure directories exist
for directory in [DATASETS_DIR, MODELS_DIR, REPORTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Training & Evaluation Parameters
RANDOM_STATE = 42
TEST_SIZE = 0.20
STRATIFY = True

# Medical / Triage Definitions
# High-acuity / emergency disease classes requiring immediate medical intervention
EMERGENCY_CLASSES = [
    'Heart attack',
    'Paralysis (brain hemorrhage)',
    'Pneumonia',
    'Bronchial Asthma',
    'Hypoglycemia',
    'Hypertension ',
    'Alcoholic hepatitis',
    'Dengue'
]

# Critical red flag symptoms
CRITICAL_RED_FLAGS = {
    'chest_pain': 'Chest Pain (Cardiac / Ischemic Alert)',
    'breathlessness': 'Acute Breathlessness / Dyspnea',
    'blood_in_sputum': 'Hemoptysis (Blood in Sputum)',
    'stomach_bleeding': 'Gastrointestinal Bleeding',
    'acute_liver_failure': 'Acute Liver Failure',
    'coma': 'Coma / Loss of Consciousness',
    'altered_sensorium': 'Altered Sensorium / Encephalopathy',
    'weakness_of_one_body_side': 'Hemiparesis / Stroke Alert',
    'loss_of_balance': 'Acute Loss of Balance / Neurological Deficit'
}

# Clinical Prototype Disclaimer
CLINICAL_DISCLAIMER = (
    "EDUCATIONAL & RESEARCH PROTOTYPE ONLY: This machine learning model has NOT been "
    "clinically validated or cleared by regulatory authorities (e.g., FDA/CE). Predictions and "
    "triage classifications are intended solely for academic benchmarking and educational "
    "demonstration. In case of a medical emergency, immediately contact local emergency services."
)
