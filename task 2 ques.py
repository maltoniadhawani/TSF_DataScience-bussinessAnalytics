```python
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)

def calculate_silhouette(data, range_k):
  silhouette_scores = []
  for k in range_k:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)
    silhouette_scores.append(silhouette_score(data, kmeans.labels_))
  return silhouette_scores

k_range = range(2, 11)
silhouette_scores = calculate_silhouette(data, k_range)

best_k = k_range[silhouette_scores.index(max(silhouette_scores))]

import matplotlib.pyplot as plt

plt.plot(k_range, silhouette_scores)
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Elbow Method for Iris Dataset")
plt.axvline(x=best_k, color='red', linestyle='--', label='Optimal k')
plt.legend()
plt.show()

kmeans = KMeans(n_clusters=best_k)
kmeans.fit(data)

data['cluster'] = kmeans.labels_

plt.scatter(data['sepal length (cm)'], data['sepal width (cm)'], c=data['cluster'])
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.title("Iris Dataset with Clusters")
plt.show()

print("Optimal number of clusters based on silhouette score:", best_k)