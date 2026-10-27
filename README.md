# 🏥 Symptom-Driven Disease Prediction System

An educational research prototype that predicts potential diseases from user-selected symptoms and raises emergency triage alerts. The repository holds two versions built on the same dataset (132 binary symptoms, 41 conditions) so that supervised and unsupervised approaches can be compared.

> **🔗 Live Deployments:**
> * **Supervised Learning:** [https://disease-prediction-system-2-ix9f.onrender.com](https://disease-prediction-system-2-ix9f.onrender.com)
> * **Unsupervised Learning:** [https://disease-prediction-system-y0ur.onrender.com](https://disease-prediction-system-y0ur.onrender.com)
> 

> **Disclaimer:** This project is for educational and research purposes only. It is **not** a substitute for professional medical diagnosis, advice, or treatment.

## 📂 Project Structure

```text
.
├── Supervised Learning/     Random Forest, KNN, Bernoulli Naive Bayes
├── Unsupervised Learning/   K-Means clustering (+ preventive measures and routines)
└── README.md
```

| Feature | [Supervised Learning](./Supervised%20Learning) | [Unsupervised Learning](./Unsupervised%20Learning) |
|---|---|---|
| Algorithms | Random Forest (main), KNN, Bernoulli Naive Bayes | K-Means (k-means++) |
| Uses disease labels to train | Yes | No (labels only describe clusters afterwards) |
| Output | Predicted disease, top 3 with probabilities, confidence | Cluster assignment, top 3 conditions recorded in nearby clusters |
| Model choice in UI | Compare 3 models | Single K-Means model |
| Preventive measures & routine | No | Yes, all 41 diseases |
| Evaluation dashboard | Confusion matrix, feature importance, model comparison | Silhouette score, cluster sizes, PCA scatter, k selection |

## 📊 Performance Comparison

| Version | Metric | Value |
|---|---|---|
| Supervised | Test accuracy, Random Forest (noisy-symptom benchmark) | **88.11%** |
| Supervised | Test accuracy, KNN / Naive Bayes | 84.04% / 75.20% |
| Supervised | Macro precision / recall / F1 (Random Forest) | 0.887 / 0.881 / 0.881 |
| Supervised | Emergency recall / under-triage rate | 87.5% / 12.5% |
| Unsupervised | Silhouette score / Davies-Bouldin index | 0.758 / 0.370 |
| Unsupervised | Held-out disease agreement, clean input | 100% |
| Unsupervised | Held-out agreement with ~32% of symptoms dropped | 90.45% |

**How to read this:**
Supervised accuracy is a standard accuracy score on a 20% test split with simulated symptom dropout. K-Means has no accuracy in the usual sense, so held-out agreement with the recorded disease is the closest equivalent. The dataset is very clean, so the 100% clean-input figure is optimistic. These are not clinical accuracy figures.

## ✨ Common Features

- 132 binary symptom inputs covering 41 conditions
- Red-flag symptom detection (e.g., chest pain, breathlessness) and emergency advisory
- User accounts, assessment history, timeline, and follow-up questions
- Explainability view and PDF assessment report
- Evaluation dashboard at `/evaluation`

## 🛠️ Tech Stack

- **Backend:** Python, Flask
- **Machine Learning:** scikit-learn, Pandas, NumPy, Matplotlib
- **Database:** SQLite (user accounts and assessment history)
- **Frontend:** HTML, CSS, JavaScript (Jinja2 templates)
- **Reporting:** PDF generation for assessment history

## 🚀 How to Run Locally

Each version has its own dependencies and setup steps. See the `README.md` inside each folder for details.

**Supervised version:**

```bash
cd "Supervised Learning"
pip install -r requirements.txt
python src/train_and_evaluate.py   # optional: retrain
python app.py
```

**Unsupervised version:**

```bash
cd "Unsupervised Learning"
pip install -r requirements.txt
python src/train_and_evaluate.py   # optional: retrain
python app.py
```

