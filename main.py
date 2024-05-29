import pandas as pd
import numpy as np


def kmeans(X, k=4, iterations=10000, tolerance=1e-4):

    n_samples, n_features = X.shape

    # inicializacia centroidov nahodne
    centroids = X[np.random.choice(n_samples, k, replace=False)]

    for iteration in range(iterations):

        # priradenie centroidu kazdemu predmetu
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        # vypocet novych centroidov
        new_centroids = []
        for centroid in range(k):
            cluster_points = X[labels == centroid]
            if len(cluster_points) > 0:
                new_centroids.append(cluster_points.mean(axis=0))
            else:
                # prazdny cluster, pridanie noveho clustra na nahodnej pozicii
                new_centroids.append(X[np.random.choice(n_samples, 1, replace=False)][0])

        new_centroids = np.array(new_centroids)

        # ukoncenie algoritmu ak sa centroidy vyrazne nepresuvaju
        if np.all(np.abs(new_centroids - centroids) < tolerance):
            print("Finished at iteration: " + str(iteration))
            break

        centroids = new_centroids

    return centroids, labels


if __name__ == "__main__":

    csv_file = 'hodnotenie predmetov.csv'
    data = pd.read_csv(csv_file, sep=';')

    focus_categories = data.iloc[:, 1:].values

    k = 5
    centroids, labels = kmeans(focus_categories, k)

    data['cluster'] = labels

    output_file = 'clustered_subjects.csv'
    data.to_csv(output_file, index=False)
