## quick sanity check of the saved k-means model (unsupervised)
import os
import sys

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))
import kmeans_clustering as kc  # noqa: E402

model, profiles, features = kc.load_artifacts()
file = pd.read_csv(os.path.join(BASE_DIR, 'datasets', 'Testing.csv'))

x_test = file[features].values
y_test = file['prognosis'].astype(str).str.strip().values

clusters = model.predict(x_test)
print("Assigned clusters (first 20):", clusters[:20])

# Descriptive check: does the dominant recorded condition of the assigned cluster match?
dominant = {cid: p['dominant_condition'] for cid, p in profiles.items()}
matches = np.array([dominant[int(c)] == y for c, y in zip(clusters, y_test)])
print("Dominant-condition agreement:", round(matches.mean(), 4))
print("Clusters used:", len(np.unique(clusters)), "of", model.n_clusters)
