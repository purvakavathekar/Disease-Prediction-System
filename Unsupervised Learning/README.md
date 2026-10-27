# System Based Disease Prediction System

A symptom-checker web app (Flask) that groups a user's symptoms with the most similar
records in a 4,000-row dataset using **k-means clustering (unsupervised learning)**.
The app reports which cluster your symptoms fall into, what characterises that cluster,
and which conditions were recorded most often inside it.

> Educational prototype only. It is not clinically validated and is not a medical diagnosis.

## Features

- 132 binary symptom inputs, 41 recorded conditions in the dataset
- K-means (k-means++) clustering of symptom vectors, no labels used for fitting
- `k` chosen automatically with a silhouette sweep (elbow + silhouette charts)
- Cluster card per assessment: cluster id, size, characteristic symptoms, nearest clusters
- Red-flag symptom detection and emergency advisory
- Explainability view, PDF report, history, timeline and follow-up questions
- Evaluation dashboard (`/evaluation`) with cluster-quality metrics and charts

## How it works

1. `src/train_and_evaluate.py` loads `datasets/Training.csv`, drops the `prognosis` column
   and fits k-means on the 132 symptom columns.
2. For each cluster it stores the size, the most prevalent symptoms, and the conditions
   recorded among its member rows (descriptive metadata only, never used to build clusters).
3. At prediction time `src/kmeans_clustering.py` assigns the user's symptom vector to the
   nearest centroid. Conditions from the 3 nearest clusters are blended with a softmax over
   squared distance to produce the "associated conditions" percentages.

## Metrics reported

| Metric | Meaning |
|---|---|
| Silhouette score | Cohesion vs separation of clusters (higher is better) |
| Davies-Bouldin index | Cluster overlap (lower is better) |
| Inertia | Within-cluster sum of squared distances |
| Assignment stability | Share of held-out rows that stay in the same cluster after simulated symptom omission |
| Purity / ARI / NMI | Descriptive agreement of clusters with the recorded conditions |

## Project layout

```
app.py                      Flask app (routes, /diagnose, dashboard)
src/kmeans_clustering.py    k-means fit, k selection, cluster profiles, assignment
src/train_and_evaluate.py   training + evaluation pipeline (run this to retrain)
src/predict.py              cluster-based prediction / triage interface
src/dashboard_charts.py     chart generation and metric loaders
src/config.py               paths and k-means hyper-parameters
explainability.py           explanation of a cluster assignment
algo.py                     helpers for exploring k and cluster quality
models/                     kmeans_model.joblib, kmeans_clusters.json, metrics
reports/                    generated charts and CSV tables
```

## How to run

```bash
pip install -r requirements.txt
python src/train_and_evaluate.py     # optional: retrain, regenerates models/ and reports/
python app.py
```

Tests:

```bash
python -m unittest test_system test_triage_evaluation
python test.py                       # quick check of the saved model on Testing.csv
```

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | HTML, CSS, Bootstrap, jQuery |
| Backend | Python, Flask, SQLite |
| ML | scikit-learn KMeans, NumPy, pandas, matplotlib |
