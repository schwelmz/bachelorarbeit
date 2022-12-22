import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing
import matplotlib.pyplot as plt

def clustering(samples, ncluster, scaling=False):
    #KMeans Clustering
    kmeans = KMeans(n_clusters=ncluster)
    if scaling == True:
        scaler = preprocessing.StandardScaler().fit(samples) 
        samples = scaler.transform(samples)
    kmeans.fit(samples)
    labels = kmeans.predict(samples)

    #calculate cuts
    k = 0
    select_label = labels[0]
    cut_indices = [0]
    cut_index = -1
    cut_index_old = -1
    while k < ncluster:
        select_label = labels[cut_index+1]
        clustersize = labels[(labels==select_label)].shape[0]
        cut_index = cut_index_old + clustersize
        cut_indices.append(cut_index)
        cut_index_old = cut_index
        k += 1
    print(cut_indices)
    return cut_indices

def main():
    #read csv files
    samples = pd.read_csv('samples.csv')
    samples = samples.to_numpy()
    data = pd.read_csv('data.csv')
    data = data.to_numpy()

    #extract values
    t_steps = data[:,0]
    timesteps = t_steps[-1]
    cgiters = data[:,1]
    samples = samples[:, :4]

    #calculate index where the iterations switch from 6 to 5
    cut_index_true = cgiters[(cgiters == 6)].shape[0]-1

    #clustering
    ncluster = 3
    cut_indices = clustering(samples, ncluster, scaling=False)

    #color clusters of cg iterations
    colors = ['red','orange','blue','green','purple','brown','pink','gray','olive','cyan','magenta']
    for c in range(0,ncluster):
        plt.plot(t_steps[cut_indices[c]+1:cut_indices[c+1]+1], cgiters[cut_indices[c]+1:cut_indices[c+1]+1],color=colors[c],label=f'cluster{c}')
    plt.legend()
    plt.show()

    #plot cg residual trajectories
    for c in range(0,ncluster):
        for i in range(cut_indices[c]+1, cut_indices[c+1]):
            plt.plot(np.arange(samples[i][:].shape[0]), samples[i][:], color=colors[c])
    plt.yscale('log')
    plt.show()

if __name__ == '__main__':
    main()