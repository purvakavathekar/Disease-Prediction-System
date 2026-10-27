# Deploying to Render

1. Push the **contents of this folder** (the folder that holds `app.py`) to the root of a GitHub repo.
2. Render -> New -> Web Service -> pick the repo (or use "Blueprint" so `render.yaml` is read).
3. Settings if you fill them in by hand:
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn app:app --workers 1 --threads 4 --timeout 180 --bind 0.0.0.0:$PORT`
   - Environment: `PYTHON_VERSION = 3.12.3`, `SECRET_KEY = <any long random string>`
4. Open the service URL: you land on the **Sign in** page; after signing in you reach the symptom assessment.

Notes
- The free tier sleeps; the first request after a pause can take ~50 s.
- SQLite is stored on Render's temporary disk, so accounts/history reset on every redeploy.
  For permanent data add a Render Disk and set `DATABASE_PATH=/var/data/disease_history.db`.
- Folder names are case-sensitive on Linux: the dataset folder must stay `Datasets/`.
