"""
Symptoms Utility Module
Provides cached access to the 132 symptoms in exact model order.
Does NOT modify or retrain the model or dataset.
"""
import os
import pandas as pd

_SYMPTOMS_RAW = None
_SYMPTOMS_FORMATTED = None

def get_symptoms_list():
    """Returns the exact 132 symptom feature names in model order."""
    global _SYMPTOMS_RAW
    if _SYMPTOMS_RAW is None:
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets', 'Training.csv')
        df = pd.read_csv(csv_path, nrows=1)
        # Exclude last column 'prognosis'
        _SYMPTOMS_RAW = df.columns[:-1].tolist()
    return list(_SYMPTOMS_RAW)

def get_symptoms_formatted():
    """Returns a list of dicts with raw_name, display_name, index for all 132 symptoms."""
    global _SYMPTOMS_FORMATTED
    if _SYMPTOMS_FORMATTED is None:
        raw_list = get_symptoms_list()
        _SYMPTOMS_FORMATTED = []
        for idx, raw in enumerate(raw_list):
            # Clean up display name
            display = raw.replace('_', ' ').strip().title()
            if display == 'Fluid Overload.1':
                display = 'Fluid Overload (Systemic)'
            elif display == 'Dischromic  Patches':
                display = 'Dischromic Patches'
            elif display == 'Spotting  Urination':
                display = 'Spotting Urination'
            _SYMPTOMS_FORMATTED.append({
                'index': idx,
                'raw_name': raw,
                'display_name': display
            })
    return list(_SYMPTOMS_FORMATTED)

def binary_vector_to_symptoms(vector):
    """Converts a 132-length 0/1 vector to a list of symptom names."""
    raw_list = get_symptoms_list()
    selected = []
    for idx, val in enumerate(vector):
        if idx < len(raw_list) and str(val).strip() in ('1', '1.0', True):
            selected.append(raw_list[idx].replace('_', ' ').strip().title())
    return selected
