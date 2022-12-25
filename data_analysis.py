import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.legend_handler import HandlerTuple

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
    print('Cluster-Grenzen: ',cut_indices)
    return cut_indices

def make_plots():
    #color clusters of cg iterations
    if True:
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        colors = ['red','orange','blue','green','purple','brown','pink','gray','olive','cyan','magenta']
        for c in range(0,ncluster):
            start = cut_indices[c]+1
            stop = cut_indices[c+1]+1
            ax1.plot(t_steps[start:stop], cgiters[start:stop],color=colors[c],label=f'cluster{c}')
        ax1.set_xlabel('timesteps')
        ax1.set_ylabel('CG Iterations when stopping criterion reached')
        ax1.legend()
        plt.show()

    #plot cg resiudal trajectories
    if True:
        step_stride = 1
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        ax2.set_yscale('log')
        ax2.set_xlabel('CG Iterations')
        ax2.set_ylabel('residual norm')
        l, b, h, w = .6, .5, .35, .35
        ax2in = ax2.inset_axes([l,b,w,h])
        ax2in.set_ylim(2.04e-3,2.05e-3)
        ax2in.set_xlim(1.53, 1.5314)
        ax2in.set_yscale('log')
        ax2.indicate_inset_zoom(ax2in)
        ax2.plot(iters, samples[::-step_stride,:].T)
        ax2in.plot(iters, samples[::-step_stride,:].T)
        cs = np.linspace(0,1, timesteps // step_stride+1)
        for idx,line in enumerate(ax2.lines): line.set_color((cs[idx], 0.5, 0.5))
        for idx,line in enumerate(ax2in.lines): line.set_color((cs[idx], 0.5, 0.5))
        plt.show()

    #animation of the plot above
    if True:
        step_stride = 10
        cs = np.linspace(0,1, timesteps // step_stride+1)
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111)
        for idx in range(0, timesteps+1, step_stride):
            ax3.plot(iters,samples[idx,:].T,color=(cs[idx//step_stride], 0.5, 0.5))
            ax3.set_ylim(2.04e-3,2.05e-3)
            ax3.set_xlim(1.53, 1.5314)
            ax3.set_yscale('log')
            ax3.set_xlabel('CG Iterations')
            ax3.set_ylabel('residual norm')
            plt.pause(0.01)
        plt.show()

    #plot cg residual trajectories with colored clusters
    if True:
        step_stride = 1
        fig4 = plt.figure()
        ax4 = fig4.add_subplot(111)
        l, b, h, w = .6, .5, .35, .35
        ax4in = ax4.inset_axes([l,b,w,h])
        ax4in.set_ylim(2.04e-3,2.05e-3)
        ax4in.set_xlim(1.53, 1.5314)
        ax4in.set_yscale('log')
        ax4.indicate_inset_zoom(ax4in)
        ax4.plot(iters, samples[::-step_stride,:].T, color = 'black')
        ax4in.plot(iters, samples[::-step_stride,:].T, color = 'black')
        for c in range(0,ncluster):
            start = cut_indices[c]//step_stride + 1
            stop = cut_indices[c+1]//step_stride
            for line in ax4.lines[start:stop]: line.set_color(colors[c])
            for line in ax4in.lines[start:stop]: line.set_color(colors[c])
        ax4.set_yscale('log')
        ax4.set_xlabel('CG Iterations')
        ax4.set_ylabel('residual norm')
        plt.show()

    #plot inner plot separately
    if True:
        step_stride = 10
        fig5 = plt.figure()
        ax5 = fig5.add_subplot(111)
        ax5.set_ylim(2.04e-3,2.05e-3)
        ax5.set_xlim(1.53, 1.5314)
        ax5.set_yscale('log')
        ax5.plot(iters, samples[::-step_stride,:].T, color = 'black')
        for c in range(0,ncluster):
            start = cut_indices[c]//step_stride + 1
            stop = cut_indices[c+1]//step_stride
            for line in ax5.lines[start:stop]: line.set_color(colors[c])
        ax5.set_xlabel('CG Iterations')
        ax5.set_ylabel('residual norm')
        plt.show()

    #plot this for the true clusters
    if True:
        step_stride = 10
        fig6 = plt.figure()
        ax6 = fig6.add_subplot(111)
        ax6.set_ylim(2.04e-3,2.05e-3)
        ax6.set_xlim(1.53, 1.5314)
        ax6.set_yscale('log')
        ax6.plot(iters, samples[::-step_stride,:].T, color = 'black')
        cut_index_true_stride = cut_index_true//step_stride
        for line in ax6.lines[:cut_index_true_stride]: line.set_color('red')
        for line in ax6.lines[cut_index_true_stride+1:]: line.set_color('blue')
        ax6.set_xlabel('CG Iterations')
        ax6.set_ylabel('residual norm')
        plt.show()

    #color the difference
    if True and ncluster == 2:
        step_stride = 10
        fig7 = plt.figure()
        ax7 = fig7.add_subplot(111)
        ax7.set_ylim(2.04e-3,2.05e-3)
        ax7.set_xlim(1.53, 1.5314)
        ax7.set_yscale('log')
        ax7.plot(iters, samples[::-step_stride,:].T, color = 'black')
        cut_index_true_stride = cut_index_true//step_stride
        start = cut_indices[0]//step_stride + 1
        stop = cut_indices[1]//step_stride
        for line in ax7.lines[start:stop]: line.set_color('blue')
        start = cut_indices[1]//step_stride + 1
        stop = cut_index_true_stride
        for line in ax7.lines[start:stop]: line.set_color('lime')
        stop = cut_indices[-1]//step_stride
        start = cut_index_true_stride
        for line in ax7.lines[start:stop]: line.set_color('red')
        ax7.set_xlabel('CG Iterations')
        ax7.set_ylabel('residual norm')
        plt.show()

if __name__ == '__main__':
    #read csv files
    samples = pd.read_csv('samples.csv')
    samples = samples.to_numpy()
    data = pd.read_csv('data.csv')
    data = data.to_numpy()

    #global
    t_steps = data[:,0]
    timesteps = int(t_steps[-1])
    cgiters = data[:,1]
    samples = samples[:, :5]
    iters = np.arange(samples.shape[1])

    #calculate index where the iterations switch from 6 to 5
    cut_index_true = cgiters[(cgiters == 6)].shape[0]-1
    print('Sprung von 6 auf 5 Iterationen: ', cut_index_true)

    #clustering
    ncluster = 2
    cut_indices = clustering(samples, ncluster, scaling=False)

    #plot results
    make_plots()