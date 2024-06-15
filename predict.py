import joblib
import numpy as np
import pandas as pd

model = joblib.load('kmeans_model.pkl')
csv_file = 'hodnotenie predmetov.csv'
data = pd.read_csv(csv_file, sep=';')
focus_categories = data.iloc[:, 1:].values

# chosen_rows = focus_categories[:3]
chosen_rows = np.array([[0, 0, 0, 3, 5, 2, 1, 10, 8, 7, 1, 0]])

centroid = np.mean(chosen_rows, axis=0).reshape(1, -1)

predicted_cluster = model.predict(centroid)[0]
cluster_indices = np.where(model.labels_ == predicted_cluster)[0]
cluster_subjects = data.iloc[cluster_indices]

print("Cluster subjects:\n", cluster_subjects)