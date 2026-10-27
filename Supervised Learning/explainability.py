"""
Explainable AI (XAI) & Model Transparency Module for Disease Diagnosis
Provides transparent, educational context for model predictions:
- Displays predicted condition and actual user-selected symptoms.
- Explains model pattern matching against the training dataset.
- Clearly indicates this is an educational tool, not a clinical diagnosis.
- Removes Leave-One-Symptom-Out (LOSO) drop percentages, unvalidated confidence
  scores, and causal claims.
"""
import os
import pickle
import numpy as np
import symptoms_data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model')

_MODEL = None

def get_model():
    """Cached loader for the existing saved model."""
    global _MODEL
    if _MODEL is None:
        with open(MODEL_PATH, 'rb') as f:
            _MODEL = pickle.load(f)
    return _MODEL

def explain_prediction(symptom_vector, predicted_disease=None):
    """
    Computes a clean, transparent explanation for a given 132-binary symptom vector.
    Focuses on pattern-matching transparency without misleading individual attribution
    percentages, causal assertions, or LOSO confidence drops.
    """
    model = get_model()
    symptoms_raw = symptoms_data.get_symptoms_list()

    # Ensure vector format
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

    # Identify user-selected symptoms in human-readable form
    selected_symptoms = []
    for idx, val in enumerate(symptom_vector):
        if val == 1 and idx < len(symptoms_raw):
            raw = symptoms_raw[idx]
            display = raw.replace('_', ' ').strip().title()
            if display == 'Fluid Overload.1':
                display = 'Fluid Overload (Systemic)'
            elif display == 'Dischromic  Patches':
                display = 'Dischromic Patches'
            elif display == 'Spotting  Urination':
                display = 'Spotting Urination'
            selected_symptoms.append(display)

    # Obtain model prediction if not provided
    if predicted_disease is None:
        pred = model.predict([symptom_vector])
        predicted_disease = str(pred[0]) if len(pred) > 0 else 'Undetermined'

    # Alternative pattern matches (unranked candidate conditions without probabilities)
    alternative_conditions = []
    try:
        probabilities = model.predict_proba([symptom_vector])[0]
        classes = model.classes_
        sorted_indices = np.argsort(probabilities)[::-1]
        for idx in sorted_indices[1:4]:
            if probabilities[idx] > 0.05:
                alt_name = str(classes[idx])
                if alt_name != predicted_disease and alt_name not in alternative_conditions:
                    alternative_conditions.append(alt_name)
    except Exception:
        alternative_conditions = []

    explanation_text = "The model generated this prediction by evaluating the submitted symptom pattern against patterns in its training data."
    disclaimer_text = "This result is for informational and educational purposes only. It cannot confirm or rule out a medical condition. Consult a qualified healthcare professional for medical advice."

    return {
        'status': 'success',
        'predicted_disease': predicted_disease,
        'selected_symptoms': selected_symptoms,
        'explanation': explanation_text,
        'disclaimer': disclaimer_text,
        'alternative_conditions': alternative_conditions
    }
