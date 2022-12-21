import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing
import matplotlib.pyplot as plt
import math

data = pd.read_csv('samples.csv')
data = data.to_numpy()
timesteps = data.shape[0]
t_steps = np.arange(timesteps)
data = data[:, :5]

if False:
    for i in range(0, len(data)):
        plt.plot(np.arange(data[i][:].shape[0]), data[i][:])
        plt.yscale('log')
        plt.yscale('log')
        plt.pause(0.001)
    plt.show()

kmeans = KMeans(n_clusters=2)
scaler = preprocessing.StandardScaler().fit(data) 
scaled_data=scaler.transform(data)
kmeans.fit(scaled_data)
labels = kmeans.predict(data)

plt.plot(t_steps, labels, label='labels')
plt.show()

#plt.scatter(X1, X2, c=labels)
#centers = kmeans.cluster_centers_
#plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
#plt.yscale('log')
#plt.show()
