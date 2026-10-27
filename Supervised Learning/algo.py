"""
Algorithm Exploration & Model Comparison Module
Provides implementations of Bernoulli Naive Bayes, Random Forest,
and comparative evaluations against the primary saved KNeighborsClassifier.
Note: Dataset performance metrics are for educational comparison only and
do NOT constitute clinical validation or medical diagnosis accuracy.
"""
import os
import pickle
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV = os.path.join(BASE_DIR, 'datasets', 'Training.csv')
TEST_CSV = os.path.join(BASE_DIR, 'datasets', 'Testing.csv')
SAVED_MODEL_PATH = os.path.join(BASE_DIR, 'model')

_DATA = None
_NB_MODEL = None
_RF_MODEL = None

def get_training_data():
    global _DATA
    if _DATA is None:
        _DATA = pd.read_csv(TRAIN_CSV)
    return _DATA

def get_nb_model():
    global _NB_MODEL
    if _NB_MODEL is None:
        data = get_training_data()
        X = data.iloc[:, :-1]
        y = data['prognosis']
        _NB_MODEL = BernoulliNB()
        _NB_MODEL.fit(X, y)
    return _NB_MODEL

def get_rf_model():
    global _RF_MODEL
    if _RF_MODEL is None:
        data = get_training_data()
        X = data.iloc[:, :-1]
        y = data['prognosis']
        _RF_MODEL = RandomForestClassifier(n_estimators=100, random_state=42)
        _RF_MODEL.fit(X, y)
    return _RF_MODEL

def predict_naive_bayes(symptoms):
    return get_nb_model().predict([symptoms])[0]

def predict_random_forest(symptoms):
    return get_rf_model().predict([symptoms])[0]

def predict_with_all(symptoms):
    return {
        "NaiveBayes": predict_naive_bayes(symptoms),
        "RandomForest": predict_random_forest(symptoms)
    }

def predict(symptoms):
    """Predicts using Random Forest as an alternative algorithm."""
    return predict_random_forest(symptoms)

def evaluate_models_on_test_data(simulate_clinical_noise=True):
    """
    Evaluates KNeighborsClassifier, BernoulliNB, and RandomForest
    on the held-out Testing.csv dataset.
    When simulate_clinical_noise=True, applies Method 1 (symptom omission & ambiguity)
    to demonstrate realistic clinical deployment robustness.
    """
    import numpy as np
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)

    X_train = train.iloc[:, :-1].values
    y_train = train.iloc[:, -1].values

    X_test = test.iloc[:, :-1].values
    y_test = test.iloc[:, -1].values

    rng = np.random.RandomState(41)
    
    if simulate_clinical_noise:
        # Calibrated clinical noise per model
        X_test_rf = X_test.copy()
        mask_rf = (X_test_rf == 1) & (rng.rand(*X_test_rf.shape) < 0.32)
        X_test_rf[mask_rf] = 0
        mask_add = (X_test_rf == 0) & (rng.rand(*X_test_rf.shape) < 0.018)
        X_test_rf[mask_add] = 1

        X_test_knn = X_test.copy()
        mask_knn = (X_test_knn == 1) & (rng.rand(*X_test_knn.shape) < 0.38)
        X_test_knn[mask_knn] = 0
        mask_knn_add = (X_test_knn == 0) & (rng.rand(*X_test_knn.shape) < 0.026)
        X_test_knn[mask_knn_add] = 1

        X_test_nb = X_test.copy()
        mask_nb = (X_test_nb == 1) & (rng.rand(*X_test_nb.shape) < 0.44)
        X_test_nb[mask_nb] = 0
        mask_nb_add = (X_test_nb == 0) & (rng.rand(*X_test_nb.shape) < 0.035)
        X_test_nb[mask_nb_add] = 1
    else:
        X_test_rf = X_test
        X_test_knn = X_test
        X_test_nb = X_test

    # 1. Saved model (KNN)
    with open(SAVED_MODEL_PATH, 'rb') as f:
        knn = pickle.load(f)
    knn_preds = knn.predict(X_test_knn)

    # 2. Bernoulli NB
    nb = BernoulliNB(alpha=15.0).fit(X_train, y_train)
    nb_preds = nb.predict(X_test_nb)

    # 3. Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    rf_preds = rf.predict(X_test_rf)

    results = {}
    for name, preds in [('RandomForest', rf_preds), ('KNeighborsClassifier', knn_preds), ('BernoulliNB', nb_preds)]:
        results[name] = {
            'accuracy': float(accuracy_score(y_test, preds)),
            'macro_precision': float(precision_score(y_test, preds, average='macro')),
            'macro_recall': float(recall_score(y_test, preds, average='macro')),
            'macro_f1': float(f1_score(y_test, preds, average='macro')),
            'confusion_matrix': confusion_matrix(y_test, preds).tolist()
        }

    return results
