"""
Unsupervised Symptom Clustering Configuration Module
Defines dataset paths, k-means artifact paths, report paths, clustering
hyper-parameters, and emergency triage category definitions.
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

# Models Paths (unsupervised k-means clustering artifacts)
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
KMEANS_MODEL_PATH = os.path.join(MODELS_DIR, 'kmeans_model.joblib')
CLUSTER_PROFILES_PATH = os.path.join(MODELS_DIR, 'kmeans_clusters.json')
METRICS_JSON_PATH = os.path.join(MODELS_DIR, 'kmeans_metrics.json')
META_JSON_PATH = os.path.join(MODELS_DIR, 'kmeans_meta.json')

# K-Means hyper-parameters.
# K is selected automatically (silhouette score, no labels used) from this range
# during training. DEFAULT_K is used if the sweep is skipped.
DEFAULT_K = 41
K_SEARCH_RANGE = range(30, 56, 1)
KMEANS_N_INIT = 10
KMEANS_MAX_ITER = 300
# Softness of cluster membership (squared-distance temperature)
MEMBERSHIP_TEMPERATURE = 1.0
# How many nearest clusters contribute to the associated-condition ranking
TOP_CLUSTERS = 3

# Reports Paths
REPORTS_DIR = os.path.join(PROJECT_ROOT, 'reports')
CLUSTER_SIZES_CHART_PATH = os.path.join(REPORTS_DIR, 'kmeans_cluster_sizes.png')
CLUSTER_SCATTER_CHART_PATH = os.path.join(REPORTS_DIR, 'kmeans_pca_scatter.png')
K_SELECTION_CHART_PATH = os.path.join(REPORTS_DIR, 'kmeans_k_selection.png')
METRICS_TABLE_PATH = os.path.join(REPORTS_DIR, 'kmeans_metrics_table.csv')
CLUSTER_REPORT_PATH = os.path.join(REPORTS_DIR, 'kmeans_cluster_report.csv')

# Ensure directories exist
for directory in [DATASETS_DIR, MODELS_DIR, REPORTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Reproducibility
RANDOM_STATE = 42

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
    "EDUCATIONAL & RESEARCH PROTOTYPE ONLY: This unsupervised clustering model has NOT been "
    "clinically validated or cleared by regulatory authorities (e.g., FDA/CE). Cluster assignments and "
    "associated conditions are intended solely for academic benchmarking and educational "
    "demonstration. In case of a medical emergency, immediately contact local emergency services."
)
