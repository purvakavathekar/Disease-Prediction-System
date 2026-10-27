"""
Medical Triage Prediction Interface Module
Provides robust, reproducible prediction functions for symptom-based triage
and differential disease diagnosis.
"""
import os
import sys
import json
import joblib
import pickle
import numpy as np
import pandas as pd

try:
    from . import config
except ImportError:
    import config

# List of 132 canonical symptom features in exact model and dataset order
CANONICAL_SYMPTOMS = [
    'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering',
    'chills', 'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting',
    'vomiting', 'burning_micturition', 'spotting_ urination', 'fatigue', 'weight_gain',
    'anxiety', 'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness',
    'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever',
    'sunken_eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion',
    'headache', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite',
    'pain_behind_the_eyes', 'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea',
    'mild_fever', 'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure',
    'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise',
    'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes',
    'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs',
    'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region',
    'bloody_stool', 'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps',
    'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes',
    'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties', 'excessive_hunger',
    'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech',
    'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints',
    'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness',
    'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort',
    'foul_smell_of urine', 'continuous_feel_of_urine', 'passage_of_gases',
    'internal_itching', 'toxic_look_(typhos)', 'depression', 'irritability',
    'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain',
    'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes',
    'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum',
    'rusty_sputum', 'lack_of_concentration', 'visual_disturbances',
    'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma',
    'stomach_bleeding', 'distention_of_abdomen', 'history_of_alcohol_consumption',
    'fluid_overload.1', 'blood_in_sputum', 'prominent_veins_on_calf', 'palpitations',
    'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring',
    'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails',
    'inflammatory_nails', 'blister', 'red_sore_around_nose', 'yellow_crust_ooze'
]

_CACHED_MODEL = None
_MODEL_SOURCE = None

def get_symptoms_list():
    """Returns canonical list of 132 symptom names."""
    return list(CANONICAL_SYMPTOMS)

def load_model():
    """
    Safely loads the trained triage model.
    Checks models/triage_model.joblib first; falls back to legacy pickled model if present.
    """
    global _CACHED_MODEL, _MODEL_SOURCE
    if _CACHED_MODEL is not None:
        return _CACHED_MODEL

    job_paths = [
        config.MODEL_JOB_PATH,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'triage_model.joblib')
    ]
    legacy_paths = [
        config.LEGACY_MODEL_PATH,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model')
    ]

    for p in job_paths:
        if os.path.exists(p):
            try:
                _CACHED_MODEL = joblib.load(p)
                _MODEL_SOURCE = f"joblib: {p}"
                return _CACHED_MODEL
            except Exception:
                pass

    for p in legacy_paths:
        if os.path.exists(p):
            try:
                with open(p, 'rb') as f:
                    _CACHED_MODEL = pickle.load(f)
                _MODEL_SOURCE = f"pickle: {p}"
                return _CACHED_MODEL
            except Exception:
                pass

    raise FileNotFoundError(
        f"No triage model found. Expected joblib model at '{config.MODEL_JOB_PATH}' "
        f"or legacy model at '{config.LEGACY_MODEL_PATH}'. "
        "Please run 'src/train_and_evaluate.py' to train and save the model."
    )

def symptom_names_to_vector(symptoms):
    """
    Converts list or set of symptom strings or comma-separated string into a 132-binary vector.
    """
    if isinstance(symptoms, str):
        symptoms = [s.strip() for s in symptoms.replace(',', ' ').split() if s.strip()]

    lookup = {}
    for idx, s in enumerate(CANONICAL_SYMPTOMS):
        lookup[s.lower()] = idx
        lookup[s.lower().replace(' ', '_')] = idx
        lookup[s.lower().replace('_', ' ')] = idx
        lookup[s.lower().replace(' ', '').replace('_', '')] = idx

    vector = [0] * len(CANONICAL_SYMPTOMS)
    matched = []
    unmatched = []

    for item in symptoms:
        raw_str = str(item).strip().lower()
        clean_norm = raw_str.replace(' ', '').replace('_', '')
        if clean_norm in lookup:
            idx = lookup[clean_norm]
            vector[idx] = 1
            matched.append(CANONICAL_SYMPTOMS[idx])
        elif raw_str in lookup:
            idx = lookup[raw_str]
            vector[idx] = 1
            matched.append(CANONICAL_SYMPTOMS[idx])
        else:
            unmatched.append(item)

    return vector, matched, unmatched

def validate_and_format_input(symptoms):
    """
    Validates input symptoms format and returns a 132-binary list and active symptom names.
    Raises ValueError on invalid formats.
    """
    if isinstance(symptoms, (list, tuple, np.ndarray)):
        # Check if already a 132-binary vector
        if len(symptoms) == len(CANONICAL_SYMPTOMS) and all(isinstance(x, (int, float, np.integer, np.floating)) for x in symptoms):
            vector = [1 if int(x) != 0 else 0 for x in symptoms]
            active = [CANONICAL_SYMPTOMS[i] for i, v in enumerate(vector) if v == 1]
            return vector, active

        # Otherwise treat as a collection of symptom string names
        vector, matched, unmatched = symptom_names_to_vector(symptoms)
        return vector, matched

    if isinstance(symptoms, str):
        # Could be comma-separated integers "0,1,0..." or comma-separated symptom names "fever,cough"
        parts = [p.strip() for p in symptoms.split(',') if p.strip()]
        if len(parts) == len(CANONICAL_SYMPTOMS) and all(p in ('0', '1') for p in parts):
            vector = [int(p) for p in parts]
            active = [CANONICAL_SYMPTOMS[i] for i, v in enumerate(vector) if v == 1]
            return vector, active
        vector, matched, unmatched = symptom_names_to_vector(parts)
        return vector, matched

    raise ValueError(
        f"Unsupported symptom input type '{type(symptoms).__name__}'. "
        "Expected 132-binary list, numpy array, list of symptom names, or comma-separated string."
    )

def predict(symptoms):
    """
    Simple prediction interface: returns the top predicted disease name string.
    Preserves backwards compatibility with legacy predict calls.
    """
    vector, _ = validate_and_format_input(symptoms)
    model = load_model()
    X_input = pd.DataFrame([vector], columns=CANONICAL_SYMPTOMS) if hasattr(model, 'feature_names_in_') else [vector]
    pred = model.predict(X_input)[0]
    return str(pred)

def predict_triage(symptoms):
    """
    Comprehensive triage prediction interface:
    Returns structured dictionary with:
    - predicted_disease
    - confidence percentage
    - top_3 differential diagnoses
    - emergency triage status (is_emergency)
    - emergency advisory text
    - detected red-flag symptoms
    - disclaimer
    """
    vector, active_symptoms = validate_and_format_input(symptoms)
    model = load_model()
    X_input = pd.DataFrame([vector], columns=CANONICAL_SYMPTOMS) if hasattr(model, 'feature_names_in_') else [vector]

    # Model predictions and probabilities
    pred = model.predict(X_input)[0]
    primary_disease = str(pred)

    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_input)[0]
        classes = list(model.classes_)
        sorted_indices = np.argsort(probs)[::-1]
        top_3 = []
        for idx in sorted_indices[:3]:
            top_3.append({
                'disease': str(classes[idx]),
                'probability': round(float(probs[idx]) * 100, 2)
            })
    else:
        top_3 = [{'disease': primary_disease, 'probability': 100.0}]

    # Red-flag symptom detection
    detected_red_flags = []
    for s in active_symptoms:
        s_norm = s.lower().replace(' ', '_')
        if s_norm in config.CRITICAL_RED_FLAGS:
            detected_red_flags.append(config.CRITICAL_RED_FLAGS[s_norm])

    # Emergency condition detection
    is_emergency_disease = primary_disease in config.EMERGENCY_CLASSES
    has_red_flags = len(detected_red_flags) > 0
    is_emergency = is_emergency_disease or has_red_flags

    # Build advisory
    if is_emergency:
        advisory = (
            "CRITICAL MEDICAL ADVISORY: High-acuity / emergency pattern detected. "
            f"Condition '{primary_disease}' requires immediate medical evaluation. "
        )
        if detected_red_flags:
            advisory += f"Red-flag symptoms present: {', '.join(detected_red_flags)}. "
        advisory += "Please seek emergency medical attention (911 / 112 / 108) or visit the nearest emergency department."
    else:
        advisory = (
            "Standard Triage Category: Non-emergent symptom pattern. "
            "Schedule a routine consultation with a healthcare provider for clinical evaluation."
        )

    return {
        'status': 'success',
        'predicted_disease': primary_disease,
        'top_3': top_3,
        'is_emergency': is_emergency,
        'is_emergency_disease': is_emergency_disease,
        'has_red_flags': has_red_flags,
        'red_flags': detected_red_flags,
        'active_symptoms': active_symptoms,
        'symptom_count': len(active_symptoms),
        'emergency_advisory': advisory,
        'disclaimer': config.CLINICAL_DISCLAIMER
    }

if __name__ == '__main__':
    print("Testing Medical Triage Predict Interface...")
    sample_symptoms = ['chest_pain', 'breathlessness', 'sweating']
    print(f"Sample Symptoms: {sample_symptoms}")
    try:
        result = predict_triage(sample_symptoms)
        print("Prediction Result:")
        print(json.dumps(result, indent=2))
    except Exception as exc:
        print(f"Prediction failed with error: {exc}")
