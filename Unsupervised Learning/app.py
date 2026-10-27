from flask import Flask, render_template, request, jsonify, send_from_directory, session, redirect, url_for, send_file, flash
from functools import wraps
import json, csv, os, io, sys
from datetime import datetime
from werkzeug.middleware.proxy_fix import ProxyFix
import numpy as np
import pandas as pd

# Ensure src directory is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Import modular helper services for features
import database
import followup
import symptoms_data
import explainability
import kmeans_clustering
import preventive_measures
import pdf_generator
try:
    from src import dashboard_charts, config
except ImportError:
    import dashboard_charts
    import config

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'disease-prediction-secret-key-2026-secure')

# Render (and most hosts) terminate HTTPS at a proxy: trust X-Forwarded-* headers so
# url_for / redirects / secure cookies behave correctly.
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

IS_PRODUCTION = bool(os.environ.get('RENDER')) or os.environ.get('FLASK_ENV') == 'production'
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    SESSION_COOKIE_SECURE=IS_PRODUCTION,   # cookie only over HTTPS on Render
    SESSION_PERMANENT=False,               # login ends when the browser is closed
)

# High-risk / emergency red flag symptoms mapping
CRITICAL_RED_FLAGS = {
    'chest_pain': 'Chest Pain',
    'breathlessness': 'Breathlessness',
    'blood_in_sputum': 'Blood in Sputum',
    'stomach_bleeding': 'Stomach Bleeding',
    'acute_liver_failure': 'Acute Liver Failure',
    'coma': 'Coma',
    'altered_sensorium': 'Altered Sensorium',
    'weakness_of_one_body_side': 'Weakness of One Body Side',
    'loss_of_balance': 'Loss of Balance'
}

# Path to datasets and models (using safe os paths)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Robust path check for survey.csv (handles 'datasets' vs 'Datasets')
_survey_path = os.path.join(BASE_DIR, 'datasets', 'survey.csv')
if not os.path.exists(_survey_path):
    _survey_path = os.path.join(BASE_DIR, 'Datasets', 'survey.csv')
SURVEY_CSV_PATH = _survey_path

# ---- K-Means clustering model (unsupervised) ----
def get_kmeans_model():
    """Loads and caches the fitted k-means model and its cluster profiles."""
    return kmeans_clustering.load_artifacts()

# ---- Feature names list (kept in sync with training columns) ----
_FEATURE_NAMES = None
def get_feature_names():
    global _FEATURE_NAMES
    if _FEATURE_NAMES is None:
        import symptoms_data as _sd
        _FEATURE_NAMES = _sd.get_symptoms_list()
    return _FEATURE_NAMES

# Ensure database tables and indexes exist
database.init_db()

def _wants_json():
    """True for AJAX / API calls, which must get JSON (401) instead of an HTML redirect."""
    return (
        request.is_json
        or request.path.startswith('/api/')
        or request.method != 'GET'
        or request.headers.get('X-Requested-With') == 'XMLHttpRequest'
    )


def login_required(f):
    """
    Decorator enforcing that a user must be authenticated before accessing
    disease prediction, history, timeline, or personal health records.
    A session whose user no longer exists in the database is treated as logged out.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user_id = session.get('user_id')
        if user_id and not database.get_user_by_id(user_id):
            session.clear()          # stale cookie (e.g. database was reset on redeploy)
            user_id = None
        if not user_id:
            if _wants_json():
                return jsonify({
                    'status': 'error',
                    'message': 'Authentication required. Please sign in.',
                    'login_url': url_for('login')
                }), 401
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login', next=request.path))
        return f(*args, **kwargs)
    return decorated_function


@app.after_request
def add_no_cache_headers(response):
    """Stop browsers caching signed-in pages, so Back/refresh never shows the app without a login."""
    if request.endpoint not in ('static', 'static_css_fallback', 'serve_report_file'):
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
    return response


@app.route('/healthz')
def healthz():
    """Lightweight health check for Render (does not touch the model or database)."""
    return jsonify({'status': 'ok'})


@app.context_processor
def inject_user():
    """Injects current authenticated user info into all rendered templates."""
    user_id = session.get('user_id')
    user = database.get_user_by_id(user_id) if user_id else None
    return {
        'current_user': user
    }


# ==========================================
# USER AUTHENTICATION & ACCESS CONTROL
# ==========================================

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Handles new user registration with secure salted password hashing."""
    if request.method == 'POST':
        if request.is_json:
            json_body = request.get_json(silent=True)
            data = json_body if isinstance(json_body, dict) else {}
        else:
            data = request.form

        username = data.get('username', '').strip()
        email = data.get('email', '').strip()
        password = data.get('password', '')
        full_name = data.get('full_name', '').strip()
        age = data.get('age', '').strip()
        gender = data.get('gender', '').strip()
        next_url = request.args.get('next') or '/'

        try:
            user = database.create_user(
                username=username,
                email=email,
                password=password,
                full_name=full_name,
                age=age,
                gender=gender
            )
            session['user_id'] = user['id']
            session['username'] = user['username']
            if request.is_json:
                return jsonify({'status': 'success', 'message': 'Account created successfully', 'user': user})
            flash('Account created successfully! Welcome to HealthDiagnose.', 'success')
            return redirect(next_url if next_url.startswith('/') else url_for('get_symp'))
        except ValueError as e:
            if request.is_json:
                return jsonify({'status': 'error', 'message': str(e)}), 400
            flash(str(e), 'danger')
            return render_template('register.html', error=str(e))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """Authenticates existing users and starts a secure session."""
    next_url = request.args.get('next') or '/'
    if not next_url.startswith('/') or next_url.startswith('//'):
        next_url = '/'

    if request.method == 'GET' and session.get('user_id') and database.get_user_by_id(session.get('user_id')):
        return redirect(next_url)

    if request.method == 'POST':
        is_json = request.is_json
        if is_json:
            json_body = request.get_json(silent=True)
            data = json_body if isinstance(json_body, dict) else {}
        else:
            data = request.form

        username_or_email = data.get('username', '').strip()
        password = data.get('password', '')
        form_next = data.get('next')
        if form_next and form_next.startswith('/') and not form_next.startswith('//'):
            next_url = form_next

        user = database.authenticate_user(username_or_email, password)
        if user:
            session['user_id'] = user['id']
            session['username'] = user['username']
            if is_json:
                return jsonify({'status': 'success', 'message': 'Logged in successfully', 'user': user, 'next': next_url})
            flash(f"Welcome back, {user['username']}!", 'success')
            return redirect(next_url if next_url.startswith('/') else url_for('get_symp'))
        else:
            err_msg = 'Invalid username/email or password.'
            if is_json:
                return jsonify({'status': 'error', 'message': err_msg}), 401
            flash(err_msg, 'danger')
            return render_template('login.html', error=err_msg, next=next_url)

    return render_template('login.html', next=next_url)


@app.route('/logout')
def logout():
    """Clears user session and logs out."""
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


@app.route('/api/auth/status', methods=['GET'])
def auth_status():
    """Returns current authentication state."""
    user_id = session.get('user_id')
    if user_id:
        user = database.get_user_by_id(user_id)
        return jsonify({'logged_in': True, 'user': user})
    return jsonify({'logged_in': False, 'user': None})


@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    """
    Renders and manages authenticated user profile with real database records.
    Displays Full name, Email address, Username, Account creation date, and activity stats.
    Allows secure editing of profile details with server-side validation.
    """
    user_id = session.get('user_id')
    user = database.get_user_by_id(user_id)
    if not user:
        flash('User account not found.', 'danger')
        return redirect(url_for('logout'))

    if request.method == 'POST':
        full_name = request.form.get('full_name', '').strip()
        email = request.form.get('email', '').strip()
        age = request.form.get('age', '').strip()
        gender = request.form.get('gender', '').strip()

        try:
            if email and ('@' not in email or '.' not in email):
                raise ValueError('Please provide a valid email address.')
            user = database.update_user_profile(
                user_id,
                full_name=full_name,
                email=email,
                age=age,
                gender=gender
            )
            flash('Profile updated successfully!', 'success')
        except ValueError as e:
            flash(str(e), 'danger')

    assessments = database.get_assessments(user_id=user_id)
    assessments_count = len(assessments)
    tracked_symptoms = database.get_tracked_symptoms(user_id=user_id)
    tracked_count = len(tracked_symptoms)

    return render_template(
        'profile.html',
        user=user,
        assessments_count=assessments_count,
        tracked_count=tracked_count
    )


# ==========================================
# CORE DISEASE DIAGNOSIS (LOGIN MANDATORY)
# ==========================================

@app.route('/')
@login_required
def get_symp():
    """Renders the main disease diagnosis interface for authenticated users."""
    return render_template('index.html')


@app.route("/diagnose", methods=["POST", "GET"])
@login_required
def diagnose():  
    """
    Assigns a 132-binary symptom vector to its nearest k-means cluster (unsupervised).
    Enforces minimum 3 symptoms, reports the cluster, the top 3 conditions recorded among
    the nearest clusters, and flags critical symptoms (chest pain, breathlessness).
    """
    if request.method == 'POST':
        test_list = None
        req_symps = None

        # Check for symptom names in JSON or form data
        if request.is_json:
            json_data = request.get_json(silent=True) or {}
            req_symps = json_data.get('symptoms')
        if not req_symps and 'symptoms' in request.form:
            raw_s = request.form.get('symptoms')
            if isinstance(raw_s, str):
                try:
                    req_symps = json.loads(raw_s)
                except Exception:
                    req_symps = [s.strip() for s in raw_s.split(',') if s.strip()]

        if req_symps and isinstance(req_symps, (list, tuple)):
            all_raw = symptoms_data.get_symptoms_list()
            # Canonical lookup dictionary for order-independent mapping
            lookup = {r.replace('_', ' ').lower(): idx for idx, r in enumerate(all_raw)}
            lookup.update({r.lower(): idx for idx, r in enumerate(all_raw)})
            test_list = [0] * 132
            seen = set()
            for s in req_symps:
                clean_s = str(s).strip().lower().replace('_', ' ')
                if clean_s in lookup and clean_s not in seen:
                    seen.add(clean_s)
                    test_list[lookup[clean_s]] = 1
        

        # ADD THIS NEW BLOCK
        if test_list is None and 'checkboxvalue' in request.form:
            insert = request.form.get('checkboxvalue', '')
            if insert:
                try:
                    test_list = [int(i.strip()) for i in insert.split(',') if i.strip()]
                except ValueError:
                    test_list = []

        # This is your existing code below it
        if test_list is None:
            insert = request.form.get('checkboxvalue', '')
            if not insert and request.is_json:
                json_data = request.get_json(silent=True) or {}
                insert = json_data.get('checkboxvalue') or json_data.get('symptom_vector') or ''
                if isinstance(insert, (list, tuple)):
                    test_list = [int(x) for x in insert]
                    insert = None

            if insert is not None:
                list_vals = insert.split(',') if isinstance(insert, str) else list(insert)
                try:
                    test_list = [int(i.strip()) if isinstance(i, str) else int(i) for i in list_vals if str(i).strip()]
                except ValueError:
                    test_list = []

        z_list = [0] * 132
        if not test_list or len(test_list) != 132 or test_list == z_list:
            return jsonify({'status': 'error', 'message': "Invalid Data: A valid symptom selection or 132-symptom vector is required"}), 400

        active_count = sum(test_list)
        allow_sparse = request.form.get('allow_sparse') in ('1', 'true', 'True')
        if not allow_sparse and request.is_json:
            json_data = request.get_json(silent=True) or {}
            allow_sparse = json_data.get('allow_sparse') in ('1', 'true', True)

        if active_count < 3 and not allow_sparse:
            return jsonify({
                'status': 'error',
                'error': 'minimum_symptoms_required',
                'symptom_count': active_count,
                'message': 'Please select at least 3 symptoms before requesting a diagnosis. Sparse inputs (1 or 2 symptoms) produce unreliable or alarming results.'
            }), 400

        # ---- K-Means cluster assignment ----
        try:
            cluster = kmeans_clustering.assign_cluster(test_list)
        except FileNotFoundError as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500
        except Exception as e:
            app.logger.exception('K-means assignment failed')
            return jsonify({'status': 'error', 'message': f'Diagnosis engine error: {e}'}), 500

        top_3 = [
            {'disease': c['condition'], 'probability': c['probability']}
            for c in cluster['associated_conditions']
        ]
        if not top_3:
            return jsonify({'status': 'error', 'message': 'No cluster could be assigned'}), 500
        primary_disease = top_3[0]['disease']
        alternative_conditions = [t['disease'] for t in top_3[1:] if t['disease'] != primary_disease]
        selected_model_name = 'K-Means Clustering'

        # Red-flag symptoms check (e.g. chest pain, breathlessness)
        all_symptoms_raw = symptoms_data.get_symptoms_list()
        detected_red_flags = []
        selected_symptoms_display = []
        for idx, val in enumerate(test_list):
            if val == 1 and idx < len(all_symptoms_raw):
                raw_symp = all_symptoms_raw[idx]
                disp = raw_symp.replace('_', ' ').strip().title()
                if disp == 'Fluid Overload.1':
                    disp = 'Fluid Overload (Systemic)'
                elif disp == 'Dischromic  Patches':
                    disp = 'Dischromic Patches'
                elif disp == 'Spotting  Urination':
                    disp = 'Spotting Urination'
                selected_symptoms_display.append(disp)
                if raw_symp in CRITICAL_RED_FLAGS:
                    detected_red_flags.append(CRITICAL_RED_FLAGS[raw_symp])

        has_red_flag = len(detected_red_flags) > 0
        red_flag_note = (
            f"RED-FLAG EMERGENCY ADVISORY: You reported critical symptom(s): {', '.join(detected_red_flags)}. "
            "If you or the patient are experiencing severe chest tightness/pain, acute shortness of breath, or loss of balance, "
            "please contact emergency medical services (such as 911 / 112 / 108) or visit the nearest emergency facility immediately."
        ) if has_red_flag else None

        return jsonify({
            'status': 'success',
            'model_id': 'kmeans',
            'model_name': selected_model_name,
            'predicted_disease': primary_disease,
            'alternative_conditions': alternative_conditions,
            'selected_symptoms': selected_symptoms_display,
            'top_3_diseases': top_3,
            'has_red_flag': has_red_flag,
            'red_flag_symptoms': detected_red_flags,
            'red_flag_note': red_flag_note,
            'cluster': cluster,
            'preventive_plan': preventive_measures.get_preventive_plan(primary_disease)
        })
    else:
        return jsonify({'status': 'error', 'message': 'Invalid request method'}), 405


# ==========================================
# PUBLIC SURVEY FUNCTIONALITY (PRESERVED)
# ==========================================

@app.route('/survey')
def survey():
    """Renders the public health experience survey."""
    return render_template('survey.html')


@app.route('/thanks')
def thanks():
    return render_template('thanks.html')


@app.route('/surveydata', methods=['GET', 'POST'])
def surveydata():
    """Preserved survey submission writing to datasets/survey.csv."""
    if request.method == 'POST':
        data = request.form.get('checkboxvalue', '')
        row_data = [item.strip() for item in data.split(',') if item.strip()]
       
        with open(SURVEY_CSV_PATH, 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row_data)
        return jsonify("Successful")
    return redirect(url_for('survey'))


# ==========================================
# SMART FOLLOW-UP, HISTORY & TIMELINE
# ==========================================

@app.route('/history')
@login_required
def history_page():
    """Renders the authenticated user's private Assessment History dashboard."""
    return render_template('history.html')


@app.route('/timeline')
@login_required
def timeline_page():
    """Renders the authenticated user's private Symptom Progression Timeline."""
    return render_template('timeline.html')


@app.route('/api/symptoms', methods=['GET'])
def get_symptoms():
    """Returns canonical list of all 132 symptoms for search and dropdowns."""
    symptoms = symptoms_data.get_symptoms_formatted()
    return jsonify({'status': 'success', 'symptoms': symptoms})


@app.route('/api/followup/questions', methods=['POST'])
@login_required
def get_followup():
    """
    Returns dynamically tailored follow-up questions based on user's selected symptoms.
    Answers are kept separate from the 132-symptom ML feature vector.
    """
    req_data = request.get_json(silent=True) or {}
    symptoms = req_data.get('symptoms', [])
    if not symptoms and 'symptoms' in request.form:
        raw_val = request.form.get('symptoms', '')
        if raw_val:
            symptoms = [s.strip() for s in raw_val.split(',') if s.strip()]

    questions_data = followup.get_follow_up_questions(symptoms)
    return jsonify({'status': 'success', 'data': questions_data})


@app.route('/api/assessment/save', methods=['POST'])
@login_required
def save_assessment_record():
    """
    Saves an assessment strictly bound to the authenticated user's ID.
    Also computes Explainable AI breakdown for transparent diagnostics.
    """
    try:
        req_data = request.get_json(silent=True) or {}
        if not req_data:
            req_data = request.form.to_dict()

        current_user_id = session['user_id']
        current_user = database.get_user_by_id(current_user_id) if current_user_id else None

        default_name = (current_user.get('full_name') or current_user.get('username')) if current_user else (session.get('username') or 'Anonymous')
        default_age = current_user.get('age') if (current_user and current_user.get('age') is not None) else ''
        default_gender = current_user.get('gender') if (current_user and current_user.get('gender')) else 'Not Specified'

        patient_name = req_data.get('name') or req_data.get('patient_name') or default_name
        patient_age = req_data.get('age') or req_data.get('patient_age') or default_age
        patient_gender = req_data.get('gender') or req_data.get('patient_gender') or default_gender
        
        symptoms = req_data.get('symptoms', [])
        if isinstance(symptoms, str):
            try:
                symptoms = json.loads(symptoms)
            except Exception:
                symptoms = [s.strip() for s in symptoms.split(',') if s.strip()]

        symptom_vector = req_data.get('symptom_vector', '')
        predicted_disease = req_data.get('predicted_disease', '').strip()
        if not predicted_disease:
            return jsonify({'status': 'error', 'message': 'Missing predicted disease'}), 400

        follow_up = req_data.get('follow_up') or {}
        if isinstance(follow_up, str):
            try:
                follow_up = json.loads(follow_up)
            except Exception:
                follow_up = {}

        # Preserve top 3 predictions, low confidence alerts, and red-flag notes in assessment record
        if req_data.get('top_3_diseases') and 'top_3_diseases' not in follow_up:
            follow_up['top_3_diseases'] = req_data['top_3_diseases']
        if req_data.get('low_confidence_warning') and 'low_confidence_warning' not in follow_up:
            follow_up['low_confidence_warning'] = req_data['low_confidence_warning']
        if req_data.get('red_flag_note') and 'red_flag_note' not in follow_up:
            follow_up['red_flag_note'] = req_data['red_flag_note']
        if req_data.get('symptom_severities') and 'symptom_severities' not in follow_up:
            follow_up['symptom_severities'] = req_data['symptom_severities']

        current_user_id = session['user_id']

        result = database.save_assessment(
            patient_name=patient_name,
            patient_age=patient_age,
            patient_gender=patient_gender,
            symptoms=symptoms,
            symptom_vector=symptom_vector,
            predicted_disease=predicted_disease,
            follow_up=follow_up,
            user_id=current_user_id
        )

        # Generate Explainable AI breakdown
        xai_breakdown = None
        if symptom_vector:
            try:
                xai_breakdown = explainability.explain_prediction(symptom_vector, predicted_disease=predicted_disease)
            except Exception:
                xai_breakdown = None

        return jsonify({
            'status': 'success',
            'assessment': result,
            'explainability': xai_breakdown
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/assessments', methods=['GET'])
@login_required
def get_assessments_list():
    """
    Retrieves saved assessment history strictly for the authenticated user.
    """
    try:
        search = request.args.get('search')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        current_user_id = session['user_id']

        assessments = database.get_assessments(
            user_id=current_user_id,
            search=search,
            start_date=start_date,
            end_date=end_date
        )
        return jsonify({'status': 'success', 'count': len(assessments), 'assessments': assessments})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/assessment/<assessment_id>', methods=['GET'])
@login_required
def get_assessment_detail(assessment_id):
    """Retrieves full details of a specific assessment, verifying ownership."""
    try:
        current_user_id = session['user_id']
        assessment = database.get_assessment_by_id(assessment_id, user_id=current_user_id)
        if not assessment:
            return jsonify({'status': 'error', 'message': 'Assessment not found or access denied'}), 404

        # Include XAI breakdown
        if assessment.get('symptom_vector'):
            try:
                assessment['explainability'] = explainability.explain_prediction(
                    assessment['symptom_vector'],
                    predicted_disease=assessment['predicted_disease']
                )
            except Exception:
                assessment['explainability'] = None

        return jsonify({'status': 'success', 'assessment': assessment})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/assessment/<assessment_id>/delete', methods=['POST', 'DELETE'])
@login_required
def delete_assessment_record(assessment_id):
    """Deletes an assessment, verifying ownership."""
    try:
        current_user_id = session['user_id']
        deleted = database.delete_assessment(assessment_id, user_id=current_user_id)
        if deleted:
            return jsonify({'status': 'success', 'message': 'Assessment deleted successfully'})
        return jsonify({'status': 'error', 'message': 'Assessment not found or unauthorized'}), 404
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/assessments/clear', methods=['POST'])
@login_required
def clear_all_history():
    """Clears saved assessment history strictly for current authenticated user."""
    try:
        current_user_id = session['user_id']
        database.clear_all_assessments(user_id=current_user_id)
        return jsonify({'status': 'success', 'message': 'Assessment history cleared successfully'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ==========================================
# DAILY HEALTH ASSESSMENT ("HOW ARE YOU FEELING TODAY?")
# ==========================================

@app.route('/api/daily-assessment/questions', methods=['GET'])
@login_required
def get_daily_assessment_questions():
    """
    Returns dynamically tailored questions based on that user's own recorded symptoms.
    Does not show a generic static list; asks specifically about user's prior symptoms.
    """
    try:
        current_user_id = session['user_id']
        recent_symptoms = database.get_user_recent_symptoms(current_user_id)
        all_symptoms = symptoms_data.get_symptoms_formatted()

        return jsonify({
            'status': 'success',
            'previous_symptoms': recent_symptoms,
            'all_supported_symptoms': all_symptoms
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/daily-assessment/save', methods=['POST'])
@login_required
def submit_daily_health_assessment():
    """
    Saves a dated daily health assessment comparing changes against prior check-ins.
    Categorizes into improving, worsening, unchanged, resolved, and new symptoms.
    """
    try:
        current_user_id = session['user_id']
        req_data = request.get_json(silent=True) or {}

        patient_name = req_data.get('patient_name') or session.get('username') or 'Anonymous'
        patient_age = req_data.get('patient_age', '')
        patient_gender = req_data.get('patient_gender', 'Not Specified')
        symptom_updates = req_data.get('symptom_updates', [])
        new_symptoms = req_data.get('new_symptoms', [])
        notes = req_data.get('notes', '').strip()

        result = database.save_daily_health_assessment(
            user_id=current_user_id,
            patient_name=patient_name,
            patient_age=patient_age,
            patient_gender=patient_gender,
            symptom_updates=symptom_updates,
            new_symptoms=new_symptoms,
            notes=notes
        )

        return jsonify({'status': 'success', 'data': result})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ==========================================
# EXPLAINABLE AI & PDF REPORT ENDPOINTS
# ==========================================

@app.route('/api/prevention/<path:disease>', methods=['GET'])
@login_required
def prevention_plan(disease):
    """Returns preventive measures and routine for a disease."""
    return jsonify({'status': 'success', 'data': preventive_measures.get_preventive_plan(disease)})


@app.route('/api/explain', methods=['POST'])
@login_required
def explain_diagnosis():
    """
    Computes Explainable AI breakdown for a symptom vector or assessment.
    Explains the k-means cluster assignment for a symptom vector.
    """
    try:
        req_data = request.get_json(silent=True) or {}
        symptom_vector = req_data.get('symptom_vector')
        predicted_disease = req_data.get('predicted_disease')

        # Can also pass an assessment_id
        assessment_id = req_data.get('assessment_id')
        if assessment_id:
            asmt = database.get_assessment_by_id(assessment_id, user_id=session['user_id'])
            if asmt:
                symptom_vector = asmt.get('symptom_vector')
                predicted_disease = asmt.get('predicted_disease')

        if not symptom_vector:
            return jsonify({'status': 'error', 'message': 'Symptom vector or valid assessment_id required'}), 400

        explanation = explainability.explain_prediction(symptom_vector, predicted_disease=predicted_disease)
        return jsonify({'status': 'success', 'data': explanation})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/assessment/<assessment_id>/pdf', methods=['GET'])
@login_required
def download_assessment_pdf(assessment_id):
    """
    Generates and downloads a clinical assessment summary PDF for the authenticated owner.
    Uses saved assessment results rather than triggering a new prediction.
    """
    try:
        current_user_id = session['user_id']
        assessment = database.get_assessment_by_id(assessment_id, user_id=current_user_id)
        if not assessment:
            return jsonify({'status': 'error', 'message': 'Assessment not found or access denied'}), 404

        pdf_bytes = pdf_generator.generate_assessment_pdf(assessment)
        return send_file(
            io.BytesIO(pdf_bytes),
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f"Clinical_Assessment_{assessment_id}.pdf"
        )
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Failed to generate PDF: {str(e)}'}), 500


# ==========================================
# SYMPTOM PROGRESSION TIMELINE ROUTES
# ==========================================

@app.route('/api/timeline', methods=['GET', 'POST'])
@login_required
def handle_timeline():
    """
    GET: Returns timeline progression entries strictly for the logged-in user.
         Supports timeframe filters: 'today', 'week', 'month', or custom date range.
    POST: Records a new timeline progression entry for the logged-in user.
    """
    current_user_id = session['user_id']

    if request.method == 'POST':
        try:
            req_data = request.get_json(silent=True) or {}
            if not req_data:
                req_data = request.form.to_dict()

            symptom_name = req_data.get('symptom_name', '').strip()
            if not symptom_name:
                return jsonify({'status': 'error', 'message': 'Symptom name is required'}), 400

            raw_severity = req_data.get('severity', 3)
            try:
                severity = int(raw_severity)
            except (ValueError, TypeError):
                severity = 3

            timestamp = req_data.get('timestamp') or datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            duration = req_data.get('duration', '')
            progress_status = req_data.get('progress_status', 'unchanged')
            notes = req_data.get('notes', '')
            assessment_id = req_data.get('assessment_id')

            entry_id = database.add_timeline_entry(
                symptom_name=symptom_name,
                severity=severity,
                timestamp=timestamp,
                duration=duration,
                progress_status=progress_status,
                notes=notes,
                assessment_id=assessment_id,
                user_id=current_user_id
            )

            return jsonify({'status': 'success', 'id': entry_id, 'message': 'Timeline entry added successfully'})
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500

    else:
        # GET request
        try:
            symptom = request.args.get('symptom')
            assessment_id = request.args.get('assessment_id') or request.args.get('assessment')
            time_filter = request.args.get('time_filter') # 'today', 'week', 'month'
            start_date = request.args.get('start_date')
            end_date = request.args.get('end_date')

            entries = database.get_timeline_entries(
                user_id=current_user_id,
                symptom_name=symptom,
                assessment_id=assessment_id,
                time_filter=time_filter,
                start_date=start_date,
                end_date=end_date,
                order_asc=True
            )
            return jsonify({'status': 'success', 'count': len(entries), 'entries': entries})
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/timeline/<int:entry_id>/delete', methods=['POST', 'DELETE'])
@login_required
def delete_timeline(entry_id):
    """Deletes a specific timeline progression entry, strictly verifying user ownership."""
    try:
        current_user_id = session['user_id']
        deleted = database.delete_timeline_entry(entry_id, user_id=current_user_id)
        if deleted:
            return jsonify({'status': 'success', 'message': 'Timeline entry deleted successfully'})
        return jsonify({'status': 'error', 'message': 'Entry not found or unauthorized'}), 404
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/timeline/tracked-symptoms', methods=['GET'])
@login_required
def get_tracked_symptoms_list():
    """Returns list of symptoms currently tracked for the authenticated user."""
    try:
        current_user_id = session['user_id']
        symptoms = database.get_tracked_symptoms(user_id=current_user_id)
        return jsonify({'status': 'success', 'tracked_symptoms': symptoms})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/timeline/symptom/delete', methods=['POST', 'DELETE'])
@login_required
def delete_tracked_symptom_timeline():
    """Removes all chronological timeline records for one tracked symptom."""
    try:
        current_user_id = session['user_id']
        req_data = request.get_json(silent=True) or {}
        symptom_name = (req_data.get('symptom_name') or request.form.get('symptom_name') or '').strip()
        if not symptom_name:
            return jsonify({'status': 'error', 'message': 'Symptom name is required'}), 400

        deleted_count = database.delete_timeline_entries_for_symptom(current_user_id, symptom_name)
        if deleted_count > 0:
            return jsonify({
                'status': 'success',
                'message': f'Removed {deleted_count} timeline record(s) for {symptom_name}',
                'deleted_count': deleted_count
            })
        return jsonify({'status': 'error', 'message': 'No timeline records found for that symptom'}), 404
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/static/css/<path:filename>')
def static_css_fallback(filename):
    """Fallback route for static/css/style.css to resolve static/stylesheets/style.css cleanly."""
    return send_from_directory(os.path.join(app.root_path, 'static', 'stylesheets'), filename)


@app.route('/reports/<path:filename>')
def serve_report_file(filename):
    """Safely serves generated report artifacts (clustering charts, CSV tables)."""
    abs_dir = os.path.join(BASE_DIR, 'reports')
    target = os.path.join(abs_dir, filename)
    if os.path.exists(target):
        return send_from_directory(abs_dir, filename)
    return jsonify({'status': 'error', 'message': f'Report file {filename} not found'}), 404


@app.route('/api/evaluation/metrics', methods=['GET'])
def get_evaluation_metrics():
    """Returns k-means clustering evaluation metrics JSON."""
    try:
        metrics = dashboard_charts.load_metrics()
        return jsonify(metrics)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/evaluation')
@app.route('/dashboard')
def evaluation_dashboard():
    """Renders the K-Means clustering evaluation dashboard."""
    try:
        metrics = dashboard_charts.load_metrics()
        cluster_report = dashboard_charts.load_cluster_report()
        metrics_table = dashboard_charts.load_metrics_table()
        chart_assets = dashboard_charts.verify_dashboard_assets().get('assets', {})
    except Exception as e:
        metrics = {'status': 'error', 'error': str(e)}
        cluster_report = []
        metrics_table = []
        chart_assets = {}

    return render_template(
        'evaluation.html',
        metrics=metrics,
        cluster_report=cluster_report,
        metrics_table=metrics_table,
        chart_assets=chart_assets
    )


if __name__ == "__main__":
    # Local development only. On Render the app is started by gunicorn (see Procfile).
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=os.environ.get("FLASK_DEBUG", "0") == "1",
    )
