import sqlite3
import json
import os
import uuid
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash, check_password_hash
import symptoms_data
import explainability

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'disease_history.db')

def get_db_connection():
    """Returns a SQLite connection with row factory configured."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the SQLite database with users, assessments, and timeline tables."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Table for user authentication
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    ''')

    # Table for assessments history
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS assessments (
            id TEXT PRIMARY KEY,
            user_id INTEGER,
            timestamp TEXT NOT NULL,
            patient_name TEXT,
            patient_age TEXT,
            patient_gender TEXT,
            symptoms_json TEXT NOT NULL,
            symptom_vector TEXT,
            symptom_count INTEGER DEFAULT 0,
            predicted_disease TEXT NOT NULL,
            follow_up_json TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        )
    ''')

    # Table for symptom progression timeline
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS timeline_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            assessment_id TEXT,
            symptom_name TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            severity INTEGER NOT NULL,
            duration TEXT,
            progress_status TEXT,
            notes TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
            FOREIGN KEY (assessment_id) REFERENCES assessments (id) ON DELETE SET NULL
        )
    ''')

    # Safe column migrations if database was initialized previously without user_id
    try:
        cursor.execute("ALTER TABLE assessments ADD COLUMN user_id INTEGER")
    except sqlite3.OperationalError:
        pass

    try:
        cursor.execute("ALTER TABLE timeline_entries ADD COLUMN user_id INTEGER")
    except sqlite3.OperationalError:
        pass

    try:
        cursor.execute("ALTER TABLE users ADD COLUMN full_name TEXT")
    except sqlite3.OperationalError:
        pass

    try:
        cursor.execute("ALTER TABLE users ADD COLUMN age INTEGER")
    except sqlite3.OperationalError:
        pass

    try:
        cursor.execute("ALTER TABLE users ADD COLUMN gender TEXT")
    except sqlite3.OperationalError:
        pass

    # Create indexes for fast querying and strict user separation
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_assessments_timestamp ON assessments (timestamp DESC)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_assessments_user ON assessments (user_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_timeline_symptom ON timeline_entries (symptom_name, timestamp)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_timeline_assessment ON timeline_entries (assessment_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_timeline_user ON timeline_entries (user_id)')

    conn.commit()
    conn.close()

# ========================================================
# USER AUTHENTICATION & ACCESS CONTROL
# ========================================================

def create_user(username, email, password, full_name=None, age=None, gender=None):
    """Registers a new user with securely hashed password and optional profile details."""
    username = username.strip() if username else ''
    email = email.strip().lower() if email and email.strip() else None
    if not username or len(username) < 3:
        raise ValueError("Username must be at least 3 characters long")
    if not password or len(password) < 6:
        raise ValueError("Password must be at least 6 characters long")

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT id FROM users WHERE LOWER(username) = LOWER(?)", (username,))
    if cursor.fetchone():
        conn.close()
        raise ValueError("Username is already taken. Please choose another.")

    if email:
        cursor.execute("SELECT id FROM users WHERE LOWER(email) = LOWER(?)", (email,))
        if cursor.fetchone():
            conn.close()
            raise ValueError("Email is already registered. Please sign in.")

    # Validate age if provided
    age_val = None
    if age is not None and str(age).strip() != '':
        try:
            age_val = int(age)
            if age_val < 1 or age_val > 120:
                raise ValueError("Age must be between 1 and 120.")
        except ValueError as ex:
            if "between 1 and 120" in str(ex):
                raise
            raise ValueError("Age must be a valid number.")

    gender_val = gender.strip() if gender and gender.strip() else None
    cleaned_name = full_name.strip() if full_name and full_name.strip() else None

    pwd_hash = generate_password_hash(password)
    created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    cursor.execute("PRAGMA table_info(users)")
    cols = [r['name'] for r in cursor.fetchall()]
    if 'full_name' not in cols:
        cursor.execute("ALTER TABLE users ADD COLUMN full_name TEXT")
    if 'age' not in cols:
        cursor.execute("ALTER TABLE users ADD COLUMN age INTEGER")
    if 'gender' not in cols:
        cursor.execute("ALTER TABLE users ADD COLUMN gender TEXT")

    cursor.execute(
        "INSERT INTO users (username, email, password_hash, created_at, full_name, age, gender) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (username, email, pwd_hash, created_at, cleaned_name, age_val, gender_val)
    )
    user_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return {
        'id': user_id,
        'username': username,
        'email': email,
        'full_name': cleaned_name,
        'age': age_val,
        'gender': gender_val
    }

def authenticate_user(username_or_email, password):
    """Authenticates a user against secure password hash."""
    if not username_or_email or not password:
        return None

    cleaned_input = username_or_email.strip()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM users WHERE LOWER(username) = LOWER(?) OR LOWER(email) = LOWER(?)",
        (cleaned_input, cleaned_input)
    )
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    if check_password_hash(row['password_hash'], password):
        row_keys = row.keys()
        user_dict = {
            'id': row['id'],
            'username': row['username'],
            'email': row['email'],
            'created_at': row['created_at'],
            'full_name': row['full_name'] if 'full_name' in row_keys else None,
            'age': row['age'] if 'age' in row_keys else None,
            'gender': row['gender'] if 'gender' in row_keys else None
        }
        return user_dict
    return None

def get_user_by_id(user_id):
    """Retrieves safe user information by ID."""
    if not user_id:
        return None
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(users)")
    cols = [r['name'] for r in cursor.fetchall()]
    select_cols = ['id', 'username', 'email', 'created_at']
    if 'full_name' in cols:
        select_cols.append('full_name')
    if 'age' in cols:
        select_cols.append('age')
    if 'gender' in cols:
        select_cols.append('gender')
    query = f"SELECT {', '.join(select_cols)} FROM users WHERE id = ?"
    cursor.execute(query, (user_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    res = dict(row)
    if 'full_name' not in res:
        res['full_name'] = None
    if 'age' not in res:
        res['age'] = None
    if 'gender' not in res:
        res['gender'] = None
    return res

def update_user_profile(user_id, full_name=None, email=None, age=None, gender=None):
    """Updates user full_name, email, age, and gender safely with database validation."""
    if not user_id:
        raise ValueError("User ID is required.")

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT id, email, full_name FROM users WHERE id = ?", (user_id,))
    user_row = cursor.fetchone()
    if not user_row:
        conn.close()
        raise ValueError("User not found.")

    cleaned_email = email.strip().lower() if email and email.strip() else None
    if cleaned_email:
        cursor.execute("SELECT id FROM users WHERE LOWER(email) = LOWER(?) AND id != ?", (cleaned_email, user_id))
        if cursor.fetchone():
            conn.close()
            raise ValueError("This email address is already registered to another account.")

    cursor.execute("PRAGMA table_info(users)")
    cols = [r['name'] for r in cursor.fetchall()]
    if 'full_name' not in cols:
        cursor.execute("ALTER TABLE users ADD COLUMN full_name TEXT")
    if 'age' not in cols:
        cursor.execute("ALTER TABLE users ADD COLUMN age INTEGER")
    if 'gender' not in cols:
        cursor.execute("ALTER TABLE users ADD COLUMN gender TEXT")

    cleaned_name = full_name.strip() if full_name is not None and full_name.strip() != '' else None
    gender_val = gender.strip() if gender is not None and gender.strip() != '' else None

    age_val = None
    if age is not None and str(age).strip() != '':
        try:
            age_val = int(age)
            if age_val < 1 or age_val > 120:
                raise ValueError("Age must be between 1 and 120.")
        except ValueError as ex:
            if "between 1 and 120" in str(ex):
                raise
            raise ValueError("Age must be a valid number.")

    cursor.execute(
        "UPDATE users SET full_name = ?, email = ?, age = ?, gender = ? WHERE id = ?",
        (cleaned_name, cleaned_email, age_val, gender_val, user_id)
    )
    conn.commit()
    conn.close()
    return get_user_by_id(user_id)

# ========================================================
# ASSESSMENTS HISTORY (STRICT USER ISOLATION)
# ========================================================

def save_assessment(patient_name, patient_age, patient_gender, symptoms, symptom_vector, predicted_disease, follow_up=None, timestamp=None, user_id=None):
    """
    Saves a new diagnosis assessment.
    user_id is strictly required to guarantee data ownership.
    """
    if user_id is None:
        raise ValueError("User ID is required to save an assessment. Users must be authenticated.")

    if timestamp is None:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    assessment_id = f"ASMT-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
    symptoms_list = symptoms if isinstance(symptoms, list) else [s.strip() for s in str(symptoms).split(',') if s.strip()]
    # Preserve order while removing duplicate symptom labels from one assessment
    seen_symptoms = set()
    deduped_symptoms = []
    for s in symptoms_list:
        key = s.strip().lower()
        if key and key not in seen_symptoms:
            seen_symptoms.add(key)
            deduped_symptoms.append(s.strip())
    symptoms_list = deduped_symptoms
    symptoms_json = json.dumps(symptoms_list)
    vector_str = json.dumps(symptom_vector) if isinstance(symptom_vector, (list, tuple)) else str(symptom_vector or '')
    follow_up_json = json.dumps(follow_up) if follow_up else "{}"
    created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO assessments (
            id, user_id, timestamp, patient_name, patient_age, patient_gender,
            symptoms_json, symptom_vector, symptom_count, predicted_disease,
            follow_up_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        assessment_id, user_id, timestamp, patient_name or 'Anonymous',
        str(patient_age or ''), patient_gender or 'Not Specified',
        symptoms_json, vector_str, len(symptoms_list), predicted_disease,
        follow_up_json, created_at
    ))

    # Always record ALL selected symptoms to the timeline on every diagnosis.
    # Use individual severity per symptom from follow-up answers if provided; fall back to neutral defaults.
    symp_severities = {}
    if follow_up and isinstance(follow_up, dict):
        if isinstance(follow_up.get('symptom_severities'), dict):
            symp_severities = follow_up.get('symptom_severities')
        raw_sev = follow_up.get('severity', 3)
        try:
            default_severity = int(str(raw_sev).split('-')[0].strip()) if raw_sev else 3
            if default_severity < 1 or default_severity > 5:
                default_severity = 3
        except (ValueError, TypeError):
            default_severity = 3
        duration = follow_up.get('duration', '')
        progression = follow_up.get('progression', 'unchanged')
    else:
        default_severity = 3
        duration = ''
        progression = 'unchanged'

    # Mapping helper for text or numeric severity values
    SEV_MAP = {'very mild': 1, 'mild': 2, 'moderate': 3, 'severe': 4, 'very severe': 5, 'incapacitating': 5}
    def parse_symptom_severity(val, fallback):
        if val is None:
            return fallback
        try:
            if isinstance(val, (int, float)):
                v = int(val)
                return v if 1 <= v <= 5 else fallback
            s = str(val).strip().lower()
            if s in SEV_MAP:
                return SEV_MAP[s]
            if s and s[0].isdigit():
                v = int(s[0])
                return v if 1 <= v <= 5 else fallback
            return fallback
        except Exception:
            return fallback

    notes = f"Auto-logged from Assessment {assessment_id} — Predicted: {predicted_disease}"

    # Save ALL symptoms (no cap), each as a separate timeline checkpoint with its own individual severity
    for symp in symptoms_list:
        clean_key = symp.strip().lower().replace(' ', '_')
        s_sev_val = (
            symp_severities.get(symp) or
            symp_severities.get(clean_key) or
            symp_severities.get(symp.strip().title()) or
            (follow_up.get(f'severity_{clean_key}') if isinstance(follow_up, dict) else None) or
            (follow_up.get(f'severity_{symp}') if isinstance(follow_up, dict) else None)
        )
        s_sev = parse_symptom_severity(s_sev_val, default_severity)

        # Read per-symptom duration and progression if available
        s_dur = duration
        s_prog = progression
        if isinstance(follow_up, dict):
            dur_dict = follow_up.get('symptom_duration') or follow_up.get('symptom_durations') or {}
            if isinstance(dur_dict, dict):
                s_dur = dur_dict.get(symp) or dur_dict.get(clean_key) or dur_dict.get(symp.strip().title()) or s_dur
            prog_dict = follow_up.get('symptom_progression') or follow_up.get('symptom_progressions') or {}
            if isinstance(prog_dict, dict):
                s_prog = prog_dict.get(symp) or prog_dict.get(clean_key) or prog_dict.get(symp.strip().title()) or s_prog

        cursor.execute('''\
            INSERT INTO timeline_entries (
                user_id, assessment_id, symptom_name, timestamp, severity,
                duration, progress_status, notes, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id, assessment_id, symp, timestamp, s_sev,
            s_dur, s_prog, notes, created_at
        ))

    conn.commit()
    conn.close()

    return {
        'id': assessment_id,
        'timestamp': timestamp,
        'predicted_disease': predicted_disease,
        'user_id': user_id
    }

def get_assessments(user_id, search=None, start_date=None, end_date=None):
    """
    Retrieves assessments strictly for the given user_id.
    If user_id is None, returns empty list to prevent data leaks.
    """
    if user_id is None:
        return []

    conn = get_db_connection()
    cursor = conn.cursor()

    query = "SELECT * FROM assessments WHERE user_id = ?"
    params = [user_id]

    if search:
        search_pattern = f"%{search.strip().lower()}%"
        query += " AND (LOWER(predicted_disease) LIKE ? OR LOWER(symptoms_json) LIKE ? OR LOWER(patient_name) LIKE ?)"
        params.extend([search_pattern, search_pattern, search_pattern])

    if start_date:
        query += " AND timestamp >= ?"
        params.append(f"{start_date} 00:00:00")

    if end_date:
        query += " AND timestamp <= ?"
        params.append(f"{end_date} 23:59:59")

    query += " ORDER BY timestamp DESC"

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    results = []
    for row in rows:
        item = dict(row)
        try:
            item['symptoms'] = json.loads(item['symptoms_json'])
        except Exception:
            item['symptoms'] = []
        try:
            item['follow_up'] = json.loads(item['follow_up_json'])
        except Exception:
            item['follow_up'] = {}
        results.append(item)

    return results

def get_assessment_by_id(assessment_id, user_id):
    """
    Retrieves full details of a single assessment, strictly validating user ownership.
    Returns None if the record does not belong to the user.
    """
    if user_id is None or not assessment_id:
        return None

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM assessments WHERE id = ? AND user_id = ?", (assessment_id, user_id))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    item = dict(row)
    try:
        item['symptoms'] = json.loads(item['symptoms_json'])
    except Exception:
        item['symptoms'] = []
    try:
        item['follow_up'] = json.loads(item['follow_up_json'])
    except Exception:
        item['follow_up'] = {}
    return item

def delete_assessment(assessment_id, user_id):
    """Deletes an assessment and its linked timeline entries, scoped strictly to user."""
    if user_id is None or not assessment_id:
        return False

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM timeline_entries WHERE assessment_id = ? AND user_id = ?", (assessment_id, user_id))
    cursor.execute("DELETE FROM assessments WHERE id = ? AND user_id = ?", (assessment_id, user_id))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted

def clear_all_assessments(user_id):
    """Clears assessments and timeline history strictly for the authenticated user."""
    if user_id is None:
        return False

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM timeline_entries WHERE user_id = ?", (user_id,))
    cursor.execute("DELETE FROM assessments WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()
    return True

# ========================================================
# SYMPTOM PROGRESSION TIMELINE (STRICT USER ISOLATION)
# ========================================================

def add_timeline_entry(symptom_name, severity, timestamp=None, duration=None, progress_status=None, notes=None, assessment_id=None, user_id=None):
    """Adds a symptom progression timeline entry strictly associated with user_id."""
    if user_id is None:
        raise ValueError("User ID is required to log a timeline entry.")

    if timestamp is None or not str(timestamp).strip():
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    else:
        timestamp = str(timestamp).replace('T', ' ')
        if len(timestamp) == 16:
            timestamp += ":00"

    try:
        severity = int(severity)
        severity = max(0, min(5, severity))
    except (ValueError, TypeError):
        severity = 3

    created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO timeline_entries (
            user_id, assessment_id, symptom_name, timestamp, severity,
            duration, progress_status, notes, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        user_id, assessment_id, symptom_name.strip(), timestamp, severity,
        duration or '', progress_status or 'unchanged', notes or '', created_at
    ))
    entry_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return entry_id

def get_timeline_entries(user_id, symptom_name=None, assessment_id=None, time_filter=None, start_date=None, end_date=None, order_asc=True):
    """
    Retrieves timeline entries strictly for user_id.
    Supports timeframe filters: 'today', 'week', 'month', or custom date range.
    Optional assessment_id returns all symptoms logged for one diagnosis checkpoint.
    """
    if user_id is None:
        return []

    conn = get_db_connection()
    cursor = conn.cursor()

    order = "ASC" if order_asc else "DESC"
    query = "SELECT * FROM timeline_entries WHERE user_id = ?"
    params = [user_id]

    # Timeframe filtering
    now = datetime.now()
    if time_filter == 'today':
        query += " AND timestamp >= ?"
        params.append(now.strftime('%Y-%m-%d 00:00:00'))
    elif time_filter == 'week':
        week_ago = now - timedelta(days=7)
        query += " AND timestamp >= ?"
        params.append(week_ago.strftime('%Y-%m-%d 00:00:00'))
    elif time_filter == 'month':
        month_ago = now - timedelta(days=30)
        query += " AND timestamp >= ?"
        params.append(month_ago.strftime('%Y-%m-%d 00:00:00'))

    if start_date:
        query += " AND timestamp >= ?"
        params.append(f"{start_date} 00:00:00")

    if end_date:
        query += " AND timestamp <= ?"
        params.append(f"{end_date} 23:59:59")

    if symptom_name and symptom_name.strip():
        query += " AND LOWER(symptom_name) = LOWER(?)"
        params.append(symptom_name.strip())

    if assessment_id and str(assessment_id).strip():
        query += " AND assessment_id = ?"
        params.append(str(assessment_id).strip())

    query += f" ORDER BY timestamp {order}"

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_tracked_symptoms(user_id):
    """Returns a list of distinct symptoms currently tracked for the user."""
    if user_id is None:
        return []

    conn = get_db_connection()
    cursor = conn.cursor()

    query = '''
        SELECT symptom_name, COUNT(*) as count, MAX(timestamp) as last_updated, AVG(severity) as avg_severity
        FROM timeline_entries
        WHERE user_id = ?
        GROUP BY symptom_name
        ORDER BY count DESC, last_updated DESC
    '''
    cursor.execute(query, (user_id,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def delete_timeline_entry(entry_id, user_id):
    """Deletes a single timeline entry by ID, strictly verifying user ownership."""
    if user_id is None or not entry_id:
        return False

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM timeline_entries WHERE id = ? AND user_id = ?", (entry_id, user_id))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted

def delete_timeline_entries_for_symptom(user_id, symptom_name):
    """Deletes all timeline entries for a given symptom name, scoped to the user."""
    if user_id is None or not symptom_name or not str(symptom_name).strip():
        return 0

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "DELETE FROM timeline_entries WHERE user_id = ? AND LOWER(symptom_name) = LOWER(?)",
        (user_id, str(symptom_name).strip())
    )
    deleted_count = cursor.rowcount
    conn.commit()
    conn.close()
    return deleted_count

# ========================================================
# DAILY HEALTH ASSESSMENT ("HOW ARE YOU FEELING TODAY?")
# ========================================================

def get_user_recent_symptoms(user_id):
    """
    Retrieves symptoms recorded in that user's previous assessment or symptom history.
    Provides previous severity, duration, and latest status for dynamic question generation.
    """
    if user_id is None:
        return []

    conn = get_db_connection()
    cursor = conn.cursor()

    # Find the user's latest assessment
    cursor.execute('''
        SELECT id, symptoms_json, follow_up_json, timestamp, patient_name, patient_age, patient_gender
        FROM assessments
        WHERE user_id = ?
        ORDER BY timestamp DESC
        LIMIT 1
    ''', (user_id,))
    latest_asmt = cursor.fetchone()

    recent_symptoms = []
    seen = set()

    if latest_asmt:
        try:
            symptoms_list = json.loads(latest_asmt['symptoms_json'])
        except Exception:
            symptoms_list = []

        # Find latest timeline record for each symptom to obtain most recent severity
        for symp in symptoms_list:
            symp_clean = symp.strip()
            if symp_clean.lower() in seen:
                continue
            seen.add(symp_clean.lower())

            cursor.execute('''
                SELECT severity, progress_status, duration, timestamp
                FROM timeline_entries
                WHERE user_id = ? AND LOWER(symptom_name) = LOWER(?)
                ORDER BY timestamp DESC
                LIMIT 1
            ''', (user_id, symp_clean))
            t_row = cursor.fetchone()

            recent_symptoms.append({
                'symptom_name': symp_clean,
                'previous_severity': int(t_row['severity']) if t_row else 3,
                'previous_status': t_row['progress_status'] if t_row else 'unchanged',
                'previous_duration': t_row['duration'] if t_row else '',
                'last_recorded_at': t_row['timestamp'] if t_row else latest_asmt['timestamp']
            })

    # If user has timeline entries that were not in latest assessment
    cursor.execute('''
        SELECT symptom_name, severity, progress_status, duration, MAX(timestamp) as last_time
        FROM timeline_entries
        WHERE user_id = ?
        GROUP BY symptom_name
        ORDER BY last_time DESC
        LIMIT 10
    ''', (user_id,))
    for row in cursor.fetchall():
        s_name = row['symptom_name'].strip()
        if s_name.lower() not in seen:
            seen.add(s_name.lower())
            recent_symptoms.append({
                'symptom_name': s_name,
                'previous_severity': int(row['severity']),
                'previous_status': row['progress_status'] or 'unchanged',
                'previous_duration': row['duration'] or '',
                'last_recorded_at': row['last_time']
            })

    conn.close()
    return recent_symptoms

def save_daily_health_assessment(user_id, patient_name, patient_age, patient_gender, symptom_updates, new_symptoms, notes='', timestamp=None):
    """
    Saves a dated daily health assessment:
    - Compares current symptom severity against previous assessments
    - Categorizes into: improving, worsening, unchanged, resolved, and new symptoms
    - Retains resolved symptoms in history without deleting
    - Generates 132-binary vector for supported active symptoms and obtains ML prediction
    - Creates corresponding timeline entries with user_id
    """
    if user_id is None:
        raise ValueError("User must be authenticated to submit a daily health assessment.")

    if timestamp is None:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    previous_symptoms_map = {
        s['symptom_name'].lower(): s for s in get_user_recent_symptoms(user_id)
    }

    improving_symptoms = []
    worsening_symptoms = []
    unchanged_symptoms = []
    resolved_symptoms = []
    active_symptoms = []

    # 1. Process Updates for Existing Symptoms
    for upd in symptom_updates:
        s_name = upd.get('symptom_name', '').strip()
        if not s_name:
            continue

        status = upd.get('status', 'unchanged').lower() # improving, worsening, unchanged, resolved
        raw_sev = upd.get('severity', 3)
        try:
            curr_sev = int(raw_sev)
            curr_sev = max(0, min(5, curr_sev))
        except (ValueError, TypeError):
            curr_sev = 3

        prev_info = previous_symptoms_map.get(s_name.lower(), {})
        prev_sev = prev_info.get('previous_severity', curr_sev)

        duration = upd.get('duration', prev_info.get('previous_duration', ''))
        still_present = upd.get('still_present', True)

        if not still_present or status == 'resolved' or curr_sev == 0:
            resolved_symptoms.append({
                'symptom_name': s_name,
                'previous_severity': prev_sev,
                'status': 'resolved',
                'duration': duration
            })
        else:
            active_symptoms.append(s_name)
            # Compare severity change
            if curr_sev < prev_sev or status == 'improving':
                improving_symptoms.append({
                    'symptom_name': s_name,
                    'previous_severity': prev_sev,
                    'current_severity': curr_sev,
                    'severity_change': curr_sev - prev_sev,
                    'status': 'improving',
                    'duration': duration
                })
            elif curr_sev > prev_sev or status == 'worsening':
                worsening_symptoms.append({
                    'symptom_name': s_name,
                    'previous_severity': prev_sev,
                    'current_severity': curr_sev,
                    'severity_change': curr_sev - prev_sev,
                    'status': 'worsening',
                    'duration': duration
                })
            else:
                unchanged_symptoms.append({
                    'symptom_name': s_name,
                    'previous_severity': prev_sev,
                    'current_severity': curr_sev,
                    'severity_change': 0,
                    'status': 'unchanged',
                    'duration': duration
                })

    # 2. Process Newly Added Symptoms
    added_new_list = []
    for new_s in new_symptoms:
        s_name = new_s.get('symptom_name', '').strip()
        if not s_name:
            continue

        raw_sev = new_s.get('severity', 3)
        try:
            sev = int(raw_sev)
            sev = max(1, min(5, sev))
        except (ValueError, TypeError):
            sev = 3

        duration = new_s.get('duration', '')
        custom_note = new_s.get('note', '')
        is_custom = bool(new_s.get('is_custom', False))

        active_symptoms.append(s_name)
        added_new_list.append({
            'symptom_name': s_name,
            'current_severity': sev,
            'duration': duration,
            'note': custom_note,
            'is_custom': is_custom,
            'status': 'new'
        })

    # Determine Overall Symptom Progression Trend
    if len(active_symptoms) == 0 and len(resolved_symptoms) > 0:
        overall_trend = "All Previous Symptoms Resolved"
    elif len(improving_symptoms) > len(worsening_symptoms) and len(improving_symptoms) > len(added_new_list):
        overall_trend = "Overall Improving"
    elif len(worsening_symptoms) > len(improving_symptoms) or len(added_new_list) > len(improving_symptoms):
        overall_trend = "Escalating / Worsening"
    else:
        overall_trend = "Stable / Unchanged"

    # 3. Build 132-Binary Symptom Vector for Supported Symptoms
    all_canonical_raw = symptoms_data.get_symptoms_list()
    canonical_lookup = {raw.replace('_', ' ').lower(): idx for idx, raw in enumerate(all_canonical_raw)}

    vector = [0] * 132
    supported_active_count = 0
    for s in active_symptoms:
        norm = s.lower().strip()
        if norm in canonical_lookup:
            idx = canonical_lookup[norm]
            vector[idx] = 1
            supported_active_count += 1

    # 4. Predict Disease using Existing Saved Model
    predicted_disease = "Health Status Update"
    if supported_active_count > 0:
        try:
            model = explainability.get_model()
            pred = model.predict([vector])
            if len(pred) > 0:
                predicted_disease = pred[0]
        except Exception:
            predicted_disease = "Evaluation Pending"
    elif len(active_symptoms) == 0:
        predicted_disease = "All Symptoms Resolved"

    # 5. Build Follow-Up Context Package
    follow_up_payload = {
        'is_daily_assessment': True,
        'overall_trend': overall_trend,
        'improving_symptoms': improving_symptoms,
        'worsening_symptoms': worsening_symptoms,
        'unchanged_symptoms': unchanged_symptoms,
        'resolved_symptoms': resolved_symptoms,
        'new_symptoms': added_new_list,
        'total_active_today': len(active_symptoms),
        'notes': notes or ''
    }

    # Save to assessments table
    asmt_res = save_assessment(
        patient_name=patient_name or 'Anonymous',
        patient_age=patient_age,
        patient_gender=patient_gender,
        symptoms=active_symptoms if active_symptoms else ["All Resolved"],
        symptom_vector=vector,
        predicted_disease=predicted_disease,
        follow_up=follow_up_payload,
        timestamp=timestamp,
        user_id=user_id
    )

    assessment_id = asmt_res['id']

    # 6. Record detailed timeline entries for each tracked symptom
    for item in improving_symptoms + worsening_symptoms + unchanged_symptoms:
        add_timeline_entry(
            symptom_name=item['symptom_name'],
            severity=item['current_severity'],
            timestamp=timestamp,
            duration=item.get('duration', ''),
            progress_status=item['status'],
            notes=f"Daily Check-in: {item['status'].capitalize()} (Prev: {item['previous_severity']}/5 -> Now: {item['current_severity']}/5)",
            assessment_id=assessment_id,
            user_id=user_id
        )

    for item in resolved_symptoms:
        add_timeline_entry(
            symptom_name=item['symptom_name'],
            severity=0,
            timestamp=timestamp,
            duration=item.get('duration', ''),
            progress_status='resolved',
            notes="Daily Check-in: Completely Resolved",
            assessment_id=assessment_id,
            user_id=user_id
        )

    for item in added_new_list:
        add_timeline_entry(
            symptom_name=item['symptom_name'],
            severity=item['current_severity'],
            timestamp=timestamp,
            duration=item.get('duration', ''),
            progress_status='new',
            notes=f"Daily Check-in: Newly Reported. {item.get('note', '')}".strip(),
            assessment_id=assessment_id,
            user_id=user_id
        )

    return {
        'status': 'success',
        'assessment_id': assessment_id,
        'timestamp': timestamp,
        'predicted_disease': predicted_disease,
        'overall_trend': overall_trend,
        'improving_count': len(improving_symptoms),
        'worsening_count': len(worsening_symptoms),
        'unchanged_count': len(unchanged_symptoms),
        'resolved_count': len(resolved_symptoms),
        'new_count': len(added_new_list),
        'active_count': len(active_symptoms),
        'report_summary': follow_up_payload
    }

# Auto-initialize database on import
init_db()
