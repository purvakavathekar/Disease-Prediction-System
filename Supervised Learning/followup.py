"""
Smart Follow-Up Questions Module
Provides category-based contextual follow-up questions for disease diagnosis.
Answers are kept strictly separate from the 132-symptom ML feature vector.
"""

SYMPTOM_CATEGORIES = {
    'dermatological': [
        'itching', 'skin_rash', 'nodal_skin_eruptions', 'pus_filled_pimples',
        'blackheads', 'scurring', 'skin_peeling', 'silver_like_dusting',
        'small_dents_in_nails', 'inflammatory_nails', 'blister',
        'red_sore_around_nose', 'yellow_crust_ooze', 'dischromic _patches',
        'red_spots_over_body', 'bruising', 'internal_itching'
    ],
    'respiratory': [
        'continuous_sneezing', 'cough', 'breathlessness', 'phlegm',
        'throat_irritation', 'redness_of_eyes', 'sinus_pressure',
        'runny_nose', 'congestion', 'chest_pain', 'loss_of_smell',
        'mucoid_sputum', 'rusty_sputum', 'blood_in_sputum', 'patches_in_throat'
    ],
    'gastrointestinal': [
        'stomach_pain', 'acidity', 'ulcers_on_tongue', 'vomiting',
        'indigestion', 'nausea', 'loss_of_appetite', 'constipation',
        'abdominal_pain', 'diarrhoea', 'acute_liver_failure', 'fluid_overload',
        'swelling_of_stomach', 'passage_of_gases', 'belly_pain',
        'distention_of_abdomen', 'stomach_bleeding', 'yellowish_skin',
        'yellowing_of_eyes', 'pain_during_bowel_movements',
        'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus'
    ],
    'musculoskeletal': [
        'joint_pain', 'muscle_wasting', 'back_pain', 'neck_pain',
        'cramps', 'knee_pain', 'hip_joint_pain', 'muscle_weakness',
        'stiff_neck', 'swelling_joints', 'movement_stiffness',
        'painful_walking', 'muscle_pain', 'weakness_in_limbs',
        'spinning_movements', 'loss_of_balance', 'unsteadiness',
        'weakness_of_one_body_side', 'headache', 'dizziness'
    ],
    'systemic': [
        'shivering', 'chills', 'fatigue', 'high_fever', 'mild_fever',
        'sweating', 'dehydration', 'malaise', 'swelled_lymph_nodes',
        'palpitations', 'toxic_look_(typhos)', 'anxiety', 'restlessness',
        'lethargy', 'depression', 'irritability', 'cold_hands_and_feets',
        'fast_heart_rate'
    ],
    'metabolic_urinary': [
        'burning_micturition', 'spotting_ urination', 'dark_urine',
        'yellow_urine', 'bladder_discomfort', 'foul_smell_of urine',
        'continuous_feel_of_urine', 'polyuria', 'weight_gain',
        'weight_loss', 'irregular_sugar_level', 'excessive_hunger',
        'increased_appetite', 'enlarged_thyroid', 'puffy_face_and_eyes',
        'swollen_legs', 'swollen_extremeties'
    ]
}

def identify_categories(selected_symptoms):
    """
    Identifies which symptom categories are represented in the user's selected symptoms.
    """
    detected_cats = set()
    cleaned_selected = [s.strip().lower().replace(' ', '_') for s in selected_symptoms]

    for cat, symptoms in SYMPTOM_CATEGORIES.items():
        for symp in symptoms:
            for user_s in cleaned_selected:
                if symp in user_s or user_s in symp:
                    detected_cats.add(cat)
                    break
    return list(detected_cats)

def format_symptoms_list(symptoms):
    """Formats symptoms into uniform dicts with name, key, and display label."""
    return [
        {
            'name': s,
            'key': s.strip().lower().replace(' ', '_'),
            'label': s.replace('_', ' ').strip().title()
        } for s in symptoms
    ]

def get_follow_up_questions(selected_symptoms):
    """
    Generates a tailored list of questions based on selected symptoms and active categories.
    Asks each question for each reported symptom individually, matching the per-symptom severity UI.
    """
    categories = identify_categories(selected_symptoms)
    symptoms_meta = format_symptoms_list(selected_symptoms) if selected_symptoms else []

    questions = []

    # Helper to find category-relevant symptoms or fall back to all selected
    def get_cat_symptoms(category_name):
        cat_keys = SYMPTOM_CATEGORIES.get(category_name, [])
        matched = []
        for s in (selected_symptoms or []):
            clean = s.strip().lower().replace(' ', '_')
            if any(k in clean or clean in k for k in cat_keys):
                matched.append(s)
        return format_symptoms_list(matched if matched else selected_symptoms)

    # 1. Symptom Duration (Per-Symptom)
    if symptoms_meta:
        questions.append({
            'id': 'duration',
            'category': 'core',
            'title': 'Symptom Duration',
            'question': 'When did you first notice each of your reported symptoms?',
            'required': False,
            'type': 'per_symptom_choice',
            'symptoms': symptoms_meta,
            'default_index': 1,
            'options': [
                {'value': 'Less than 24 hours (Acute)', 'label': '< 24 Hours (Acute)'},
                {'value': '1 to 3 days', 'label': '1 to 3 Days'},
                {'value': '4 to 7 days', 'label': '4 to 7 Days'},
                {'value': '1 to 2 weeks', 'label': '1 to 2 Weeks'},
                {'value': 'More than 2 weeks (Persistent / Chronic)', 'label': '> 2 Weeks (Chronic)'}
            ],
            'help_text': 'Helps distinguish acute episodes from chronic conditions for each symptom individually.'
        })
    else:
        questions.append({
            'id': 'duration',
            'category': 'core',
            'title': 'Symptom Duration',
            'question': 'When did you first notice these symptoms?',
            'required': False,
            'type': 'select',
            'options': [
                'Less than 24 hours (Acute)',
                '1 to 3 days',
                '4 to 7 days',
                '1 to 2 weeks',
                'More than 2 weeks (Persistent / Chronic)'
            ],
            'help_text': 'Helps distinguish acute episodes from chronic conditions.'
        })

    # 2. Individual Symptom Severity (Per-Symptom)
    if symptoms_meta:
        questions.append({
            'id': 'symptom_severities',
            'category': 'core',
            'title': 'Individual Symptom Severity',
            'question': 'Please select the severity level for each of your reported symptoms:',
            'required': False,
            'type': 'per_symptom_severity',
            'symptoms': symptoms_meta,
            'options': [
                {'value': 1, 'label': '1 - Very Mild'},
                {'value': 2, 'label': '2 - Mild'},
                {'value': 3, 'label': '3 - Moderate'},
                {'value': 4, 'label': '4 - Severe'},
                {'value': 5, 'label': '5 - Very Severe'}
            ],
            'help_text': 'Rate each symptom individually to track its personal severity on your progression timeline.'
        })
    else:
        questions.append({
            'id': 'severity',
            'category': 'core',
            'title': 'Current Severity',
            'question': 'How would you rate the overall severity of your symptoms right now?',
            'required': False,
            'type': 'rating',
            'options': [
                {'value': 1, 'label': '1 - Very Mild (Barely noticeable)'},
                {'value': 2, 'label': '2 - Mild (Noticeable, minor annoyance)'},
                {'value': 3, 'label': '3 - Moderate (Interferes with routine)'},
                {'value': 4, 'label': '4 - Severe (Significantly limiting)'},
                {'value': 5, 'label': '5 - Very Severe (Incapacitating)'}
            ],
            'help_text': 'Used for symptom tracking on your progression timeline.'
        })

    # 3. Frequency / Occurrence (Per-Symptom)
    if symptoms_meta:
        questions.append({
            'id': 'frequency',
            'category': 'core',
            'title': 'Frequency / Occurrence',
            'question': 'How frequently does each of your symptoms occur?',
            'required': False,
            'type': 'per_symptom_choice',
            'symptoms': symptoms_meta,
            'default_index': 1,
            'options': [
                {'value': 'Occasional (Comes up rarely or once a day)', 'label': 'Occasional'},
                {'value': 'Intermittent (Comes and goes in waves)', 'label': 'Intermittent'},
                {'value': 'Frequent (Present most of the day)', 'label': 'Frequent'},
                {'value': 'Continuous (Constant and uninterrupted)', 'label': 'Continuous'}
            ],
            'help_text': 'Indicates whether each symptom is episodic, intermittent, or continuous.'
        })
    else:
        questions.append({
            'id': 'frequency',
            'category': 'core',
            'title': 'Frequency / Occurrence',
            'question': 'How frequently do your symptoms occur?',
            'required': False,
            'type': 'select',
            'options': [
                'Occasional (Comes up rarely or once a day)',
                'Intermittent (Comes and goes in waves)',
                'Frequent (Present most of the day)',
                'Continuous (Constant and uninterrupted)'
            ],
            'help_text': 'Indicates whether the condition is episodic or continuous.'
        })

    # 4. Symptom Progression / Intensity (Per-Symptom)
    if symptoms_meta:
        questions.append({
            'id': 'progression',
            'category': 'core',
            'title': 'Symptom Progression',
            'question': 'How has the intensity of each symptom changed since it started?',
            'required': False,
            'type': 'per_symptom_choice',
            'symptoms': symptoms_meta,
            'default_index': 1,
            'options': [
                {'value': 'Improving (Getting milder)', 'label': 'Improving'},
                {'value': 'Unchanged (Remaining steady)', 'label': 'Unchanged'},
                {'value': 'Gradually worsening', 'label': 'Gradually Worsening'},
                {'value': 'Rapidly worsening', 'label': 'Rapidly Worsening'}
            ],
            'help_text': 'Tracks whether each symptom is resolving, steady, or escalating.'
        })
    else:
        questions.append({
            'id': 'progression',
            'category': 'core',
            'title': 'Symptom Progression',
            'question': 'How has the intensity of your symptoms changed since they started?',
            'required': False,
            'type': 'select',
            'options': [
                'Improving (Getting milder)',
                'Unchanged (Remaining steady)',
                'Gradually worsening',
                'Rapidly worsening'
            ],
            'help_text': 'Tracks whether the condition is resolving or escalating.'
        })

    # Category-specific targeted questions (also asked per relevant symptom)

    if 'dermatological' in categories:
        derm_symps = get_cat_symptoms('dermatological')
        if derm_symps:
            questions.append({
                'id': 'skin_spread',
                'category': 'dermatological',
                'title': 'Skin Condition Context',
                'question': 'Have the skin lesions, rash, or itching spread to larger body areas?',
                'required': False,
                'type': 'per_symptom_choice',
                'symptoms': derm_symps,
                'default_index': 0,
                'options': [
                    {'value': 'No, confined to one isolated spot', 'label': 'Confined to Single Spot'},
                    {'value': 'Gradually spreading to nearby areas', 'label': 'Gradually Spreading'},
                    {'value': 'Widespread across multiple body regions', 'label': 'Widespread Across Body'}
                ],
                'help_text': 'Tracks dermatological spread over time.'
            })

    if 'gastrointestinal' in categories:
        gi_symps = get_cat_symptoms('gastrointestinal')
        if gi_symps:
            questions.append({
                'id': 'gi_hydration',
                'category': 'gastrointestinal',
                'title': 'Digestive & Hydration Context',
                'question': 'Are you able to keep water and light food down comfortably?',
                'required': False,
                'type': 'per_symptom_choice',
                'symptoms': gi_symps,
                'default_index': 0,
                'options': [
                    {'value': 'Yes, eating and drinking without difficulty', 'label': 'Eating & Drinking Comfortably'},
                    {'value': 'Occasional nausea/discomfort, but able to hydrate', 'label': 'Occasional Discomfort'},
                    {'value': 'Difficulty keeping fluids down or severe cramping', 'label': 'Difficulty Keeping Fluids Down'}
                ],
                'help_text': 'Evaluates hydration tolerance.'
            })

    if 'musculoskeletal' in categories:
        musc_symps = get_cat_symptoms('musculoskeletal')
        if musc_symps:
            questions.append({
                'id': 'mobility_impact',
                'category': 'musculoskeletal',
                'title': 'Mobility & Joint Context',
                'question': 'Does joint stiffness or muscle pain impede walking or daily movement?',
                'required': False,
                'type': 'per_symptom_choice',
                'symptoms': musc_symps,
                'default_index': 0,
                'options': [
                    {'value': 'Full mobility, minor stiffness only', 'label': 'Full Mobility'},
                    {'value': 'Moderate restriction (e.g. morning stiffness)', 'label': 'Moderate Restriction'},
                    {'value': 'Significant difficulty walking or standing', 'label': 'Significant Movement Difficulty'}
                ],
                'help_text': 'Assesses physical mobility impact.'
            })


    return {
        'categories': categories,
        'symptom_count': len(selected_symptoms),
        'questions': questions,
        'disclaimer': (
            'Notice: Follow-up answers are gathered solely for contextual tracking, '
            'personal record-keeping, and symptom progression visualization. They do not '
            'alter the underlying machine learning model prediction and do not constitute '
            'clinical medical advice.'
        )
    }
