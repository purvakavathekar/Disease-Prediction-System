"""
Explainable AI (XAI) & Model Transparency Module for K-Means Symptom Clustering
Provides transparent, educational context for a cluster assignment:
- Which cluster the submitted symptom pattern was grouped into.
- Which symptoms are characteristic of that cluster and which of the user's
  symptoms match it.
- Which conditions are recorded among the cluster's training records
  (descriptive; the clustering itself never used condition labels).
- Clearly indicates this is an educational tool, not a clinical diagnosis.
"""
import os
import sys

import symptoms_data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import kmeans_clustering  # noqa: E402


def get_model():
    """Cached loader for the fitted k-means model."""
    model, _, _ = kmeans_clustering.load_artifacts()
    return model


def _display_name(raw):
    display = raw.replace('_', ' ').strip().title()
    if display == 'Fluid Overload.1':
        return 'Fluid Overload (Systemic)'
    if display == 'Dischromic  Patches':
        return 'Dischromic Patches'
    if display == 'Spotting  Urination':
        return 'Spotting Urination'
    return display


def predict_condition(symptom_vector):
    """Returns the condition most associated with the vector's k-means cluster."""
    result = kmeans_clustering.assign_cluster(symptom_vector)
    assoc = result['associated_conditions']
    return assoc[0]['condition'] if assoc else 'Undetermined'


def explain_prediction(symptom_vector, predicted_disease=None):
    """
    Explains a cluster assignment for a 132-binary symptom vector.
    Reports the cluster, its characteristic symptoms and how many of the
    user's symptoms match, without causal claims or invented confidence.
    """
    symptoms_raw = symptoms_data.get_symptoms_list()

    if isinstance(symptom_vector, str):
        try:
            symptom_vector = [int(i.strip()) for i in symptom_vector.split(',') if i.strip()]
        except Exception:
            symptom_vector = [0] * 132
    else:
        symptom_vector = [int(x) for x in symptom_vector]

    if len(symptom_vector) != 132 or sum(symptom_vector) == 0:
        return {
            'status': 'error',
            'message': 'Valid 132-symptom binary vector with at least one active symptom is required'
        }

    selected_symptoms = [
        _display_name(symptoms_raw[i])
        for i, v in enumerate(symptom_vector) if v == 1 and i < len(symptoms_raw)
    ]

    cluster = kmeans_clustering.assign_cluster(symptom_vector)
    associated = [c['condition'] for c in cluster['associated_conditions']]
    if predicted_disease is None:
        predicted_disease = associated[0] if associated else 'Undetermined'

    alternative_conditions = [c for c in associated if c != predicted_disease]
    cluster_symptoms = [_display_name(s['symptom']) for s in cluster['cluster_symptoms']]
    matched = [_display_name(s) for s in cluster['matched_symptoms']]

    explanation_text = (
        f"K-means grouped the submitted symptom pattern into cluster #{cluster['cluster_id']} "
        f"({cluster['cluster_size']} similar records in the training data). "
        f"{len(matched)} of your {cluster['n_active_symptoms']} symptoms are among that cluster's most "
        "characteristic symptoms. The condition shown is the one most often recorded for records in the cluster."
    )
    disclaimer_text = (
        "This result is for informational and educational purposes only. It cannot confirm or rule out a "
        "medical condition. Consult a qualified healthcare professional for medical advice."
    )

    return {
        'status': 'success',
        'predicted_disease': predicted_disease,
        'selected_symptoms': selected_symptoms,
        'explanation': explanation_text,
        'disclaimer': disclaimer_text,
        'alternative_conditions': alternative_conditions,
        'cluster_id': cluster['cluster_id'],
        'cluster_size': cluster['cluster_size'],
        'cluster_membership': cluster['membership'],
        'cluster_symptoms': cluster_symptoms,
        'matched_symptoms': matched,
    }
