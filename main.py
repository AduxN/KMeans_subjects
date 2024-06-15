import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, OPTICS, SpectralClustering, kmeans_plusplus
from sklearn.metrics import silhouette_score, davies_bouldin_score


def compute_inertia(X, labels, centroids):
    inertia = 0
    for i in range(len(X)):
        centroid = centroids[labels[i]]
        inertia += np.sum((X[i] - centroid) ** 2)
    return inertia


if __name__ == "__main__":

    csv_file = 'hodnotenie predmetov.csv'
    data = pd.read_csv(csv_file, sep=';')

    focus_categories = data.iloc[:, 1:].values

    seed = 2
    ks = range(2, 12)

    silhouette_kmeans = []
    silhouette_plus = []
    silhouette_spectral = []
    db_kmeans = []
    db_plus = []
    db_spectral = []
    inertia_kmeans = []
    inertia_plus = []

    for k in ks:
        plus = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=1000, random_state=seed)
        plus.fit(focus_categories)

        kmeans = KMeans(n_clusters=k, random_state=seed, max_iter=1000)
        kmeans.fit(focus_categories)

        spectral = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', random_state=seed)
        spectral.fit(focus_categories)

        print()
        print(f"K: {k}")
        # kmeans testy

        # od -1 po 1, cim vacsie tym lepsie
        silhouette_avg = silhouette_score(focus_categories, kmeans.labels_)
        print(f"Silhouette Score kmeans: {silhouette_avg}")
        silhouette_kmeans.append(silhouette_avg)

        # cim mensie tym lepsie
        inertia = compute_inertia(focus_categories, kmeans.labels_, kmeans.cluster_centers_)
        print(f"Inertia kmeans: {inertia}")
        inertia_kmeans.append(inertia)

        # cim mensie tym lepsie
        db_index = davies_bouldin_score(focus_categories, kmeans.labels_)
        print(f"DB kmeans: {db_index}")
        db_kmeans.append(db_index)



        # od -1 po 1, cim vacsie tym lepsie
        silhouette_avg3 = silhouette_score(focus_categories, plus.labels_)
        print(f"Silhouette Score kmeans: {silhouette_avg3}")
        silhouette_plus.append(silhouette_avg3)

        # cim mensie tym lepsie
        inertia2 = compute_inertia(focus_categories, plus.labels_, plus.cluster_centers_)
        print(f"Inertia kmeans: {inertia2}")
        inertia_plus.append(inertia2)

        # cim mensie tym lepsie
        db_index3 = davies_bouldin_score(focus_categories, plus.labels_)
        print(f"DB kmeans: {db_index3}")
        db_plus.append(db_index3)


        # spektral testy
        silhouette_avg2 = silhouette_score(focus_categories, spectral.labels_)
        print(f"Silhouette Score spectral: {silhouette_avg2}")
        silhouette_spectral.append(silhouette_avg2)

        db_index2 = davies_bouldin_score(focus_categories, spectral.labels_)
        print(f"DB spectral: {db_index2}")
        db_spectral.append(db_index2)

        if k == 6:
            data['kmeans cluster'] = kmeans.labels_
            data['spectral cluster'] = spectral.labels_

            output_file = 'clustered_subjects2.csv'
            data.to_csv(output_file, index=False)



    # grafy
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.plot(ks, silhouette_kmeans, marker='o', label='K-means', color='blue')
    plt.plot(ks, silhouette_spectral, marker='o', label='Spectral', color='orange')
    plt.plot(ks, silhouette_plus, marker='o', label='Plus', color='green')
    plt.title('Silhouette Score - čím väčšie, tým lepšie')
    plt.xlabel('Počet zhlukov (k)')
    plt.ylabel('Silhouette Score')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(ks, db_kmeans, marker='o', label='K-means', color='blue')
    plt.plot(ks, db_spectral, marker='o', label='Spectral', color='orange')
    plt.plot(ks, db_plus, marker='o', label='Plus', color='green')
    plt.title('Davies-Bouldin Index - čím menšie, tým lepšie')
    plt.xlabel('Počet zhlukov (k)')
    plt.ylabel('Davies-Bouldin Index')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(ks, inertia_kmeans, marker='o', label='K-means', color='blue')
    plt.plot(ks, inertia_plus, marker='o', label='Plus', color='green')
    plt.title('Inertia - čím menšie, tým lepšie')
    plt.xlabel('Počet zhlukov (k)')
    plt.ylabel('Inertia')
    plt.legend()

    plt.tight_layout()
    plt.show()
