# 🏥 Symptom-Driven Disease Prediction System

An educational research prototype that predicts potential diseases based on user-selected symptoms and provides emergency triage alerts. 

> **Disclaimer:** This project is for educational and research purposes only. It is **not** a substitute for professional medical diagnosis, advice, or treatment.

## 📂 Project Structure
This repository contains two versions of the project developed to compare different machine learning approaches:

- **[Supervised Learning](./Supervised%20Learning)**: The original version. Uses Random Forest, K-Nearest Neighbors (KNN), and Bernoulli Naive Bayes to predict diseases from 132 binary symptom features.
- **[Unsupervised Learning](./Unsupervised%20Learning)**: The upgraded version. Uses K-Means clustering on symptom patterns (excluding the disease label) to discover natural groupings, evaluated using ARI and NMI.

## 🛠️ Tech Stack
- **Backend:** Python, Flask
- **Machine Learning:** scikit-learn, Pandas, NumPy
- **Database:** SQLite (for user accounts and assessment history)
- **Frontend:** HTML, CSS, JavaScript (Jinja2 templates)
- **Reporting:** PDF generation for assessment history

## 🚀 How to Run Locally
Each version has its own specific dependencies and setup steps. Please refer to the `README.md` inside the respective folder (`Supervised Learning` or `Unsupervised Learning`) for detailed instructions on how to run them.

## ⚠️ Dataset Note
The dataset used contains approximately 4,920 records, 132 binary symptom features, and 41 disease classes. The data was cleaned and analyzed to handle duplicate records and ensure reproducible training.