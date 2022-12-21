from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

############# generate random sample
X = np.random.rand(300,2)
plt.scatter(X[:, 0], X[:, 1])
plt.show()

############# k-means
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
labels = kmeans.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=labels)
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.show()