import joblib
from sklearn.cluster import KMeans
import pandas as pd

csv_file = 'hodnotenie predmetov.csv'
data = pd.read_csv(csv_file, sep=';')

focus_categories = data.iloc[:, 1:].values
plus = KMeans(n_clusters=6, init='k-means++', n_init=10, max_iter=1000, random_state=1)
plus.fit(focus_categories)
joblib.dump(plus, 'kmeans_model.pkl')