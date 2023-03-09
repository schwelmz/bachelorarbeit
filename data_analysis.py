import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.legend_handler import HandlerTuple
import time
import imageio

def clustering(samples, ncluster, scaling=False):
    #KMeans Clustering
    kmeans = KMeans(n_clusters=ncluster, init='k-means++', random_state=170)
    if scaling == True:
        scaler = preprocessing.StandardScaler().fit(samples) 
        samples = scaler.transform(samples)
    kmeans.fit(samples)
    labels = kmeans.predict(samples)

    #calculate cuts
    cut_idxs = [-1]
    label_old = labels[0]
    for idx in range(0,timesteps):
        label_select = labels[idx]
        if label_select == label_old:
            continue
        else:
            cut_idxs.append(idx)
            label_old = label_select
    cut_idxs.append(timesteps)
    print('Cluster-Grenzen:',cut_idxs)


    if False:
        k = 0
        cut_indices = [-1]
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
    return labels, cut_idxs

def make_plots(labels):
    colors = ['red','orange','blue','green','purple','brown','pink','gray','olive','cyan','magenta', 'yellow', 'black', 'darkgreen', 'darkblue', 'deeppink']
    #color clusters of cg iterations
    if False:
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        for c in range(0,ncluster):
            start = cut_idxs[c]+1
            stop = cut_idxs[c+1]+1
            ax1.plot(t_steps[start:stop], cgiters[start:stop],color=colors[c],label=f'cluster{c}')
        ax1.set_xlabel('timesteps')
        ax1.set_ylabel('CG Iterations when stopping criterion reached')
        ax1.legend()
        plt.show()

    ######################################################### residual trajectories ####################################################################################
    #plot cg resiudal trajectories
    if False:
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
    if False:
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
    if False:
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
        #color lines
        labels_stride = labels[::step_stride]
        for idx in range(0, len(ax4.lines)):
            ax4.lines[timesteps//step_stride-idx].set_color(colors[labels_stride[idx]])
            ax4in.lines[timesteps//step_stride-idx].set_color(colors[labels_stride[idx]])
        #label
        for label in range(0, ncluster):
            for idx in range(0,len(ax4.lines)):
                if labels_stride[idx] == label:
                    ax4.lines[timesteps//step_stride-idx].set_label(f'cluster{label}')
                    break
        plt.xlabel('CG Iterations')
        plt.ylabel('residual norm')
        ax4.set_yscale('log')
        plt.legend()
        plt.show()

    #plot inner plot separately
    if False:
        step_stride = 10
        fig5 = plt.figure()
        ax5 = fig5.add_subplot(111)
        ax5.plot(iters, samples[::step_stride,:].T, color = 'black')
        #color lines
        labels_stride = labels[::step_stride]
        for idx in range(0, len(ax5.lines)):
            ax5.lines[idx].set_color(colors[labels_stride[idx]])
        #label
        for label in range(0, ncluster):
            for idx in range(0,len(ax5.lines)):
                if labels_stride[idx] == label:
                    ax5.lines[idx].set_label(f'cluster{label}')
                    break
        plt.xlabel('CG Iterations')
        plt.ylabel('residual norm')
        ax5.set_ylim(2.04e-3,2.05e-3)
        ax5.set_xlim(1.53, 1.5314)
        ax5.set_yscale('log')
        plt.legend()
        plt.show()

    #plot this for the true clusters
    if False:
        step_stride = 10
        fig6 = plt.figure()
        ax6 = fig6.add_subplot(111)
        ax6.set_ylim(2.04e-3,2.05e-3)
        ax6.set_xlim(1.53, 1.5314)
        ax6.set_yscale('log')
        ax6.plot(iters, samples[::step_stride,:].T, color = 'black')
        cut_index_true_stride = cut_index_true//step_stride
        for line in ax6.lines[:cut_index_true_stride]: line.set_color('blue')
        for line in ax6.lines[cut_index_true_stride+1:]: line.set_color('red')
        ax6.set_xlabel('CG Iterations')
        ax6.set_ylabel('residual norm')
        plt.show()

    #color the difference
    if False and ncluster == 2:
        step_stride = 10
        fig7 = plt.figure()
        ax7 = fig7.add_subplot(111)
        ax7.set_ylim(2.04e-3,2.05e-3)
        ax7.set_xlim(1.53, 1.5314)
        ax7.set_yscale('log')
        ax7.plot(iters, samples[::step_stride,:].T, color = 'black')
        cut_index_true_stride = cut_index_true//step_stride
        start = cut_idxs[0]//step_stride + 1
        stop = cut_idxs[1]//step_stride
        print(start,stop)
        for line in ax7.lines[start:stop]: line.set_color('blue')
        start = cut_idxs[1]//step_stride + 1
        stop = cut_index_true_stride
        print(start,stop)
        for line in ax7.lines[start:stop]: line.set_color('lime')
        stop = cut_idxs[-1]//step_stride
        start = cut_index_true_stride
        print(start,stop)
        for line in ax7.lines[start:stop]: line.set_color('red')
        ax7.set_xlabel('CG Iterations')
        ax7.set_ylabel('residual norm')
        plt.show()

    ######################################################### transmembrane voltage ####################################################################################
    ###### plot the transmembarane voltage colored by cg iterations needed
    xs = np.linspace(0,11.9,1191)
    if False:
        step_stride = 10
        fig8 = plt.figure()
        ax8 = fig8.add_subplot(111)
        l, b, h, w = .7, .7, .25, .25
        ax8in = ax8.inset_axes([l,b,w,h])
        ax8.plot(xs, trajectory[0, :, 0], '--', color='black',label='V_h(t=0)')
        ax8in.plot(xs, trajectory[0, :, 0], '--', color='black',label='V_h(t=0)')
        plt.pause(0.001)
        print('\rStarting animation in... 2', end='', flush=True)
        time.sleep(1)
        print('\rStarting animation in... 1', end='', flush=True)
        print('\n')
        time.sleep(1)
        for idx in range(1, timesteps, step_stride):
            if idx <= cut_index_true:
                ax8.plot(xs, trajectory[idx,:,0].T, color='cyan')
                ax8in.plot(xs, trajectory[idx,:,0].T, color='cyan')
            else:
                ax8.plot(xs, trajectory[idx,:,0].T, color='magenta')
                ax8in.plot(xs, trajectory[idx,:,0].T, color='magenta')
            ax8.set_xlim(5.5,6.5)
            ax8.set_xlabel('fiber length [cm]')
            ax8.set_ylabel('transmembrane Potental V_m [mV]')
            ax8.indicate_inset_zoom(ax8in)
            ax8in.set_xlim(5.8,6.1)
            ax8in.set_ylim(24,33)
            ax8.legend()
            plt.pause(0.0005)
        ax8.plot(xs, trajectory[-1, :, 0], color='black', label='V_h(t=5)')
        ax8in.plot(xs, trajectory[-1, :, 0], color='black', label='V_h(t=5)')
        plt.show()

    ###### plot the transmembrane voltage turning from green to red over time
    if False:
        step_stride = 10
        cs = np.linspace(0,1, timesteps // step_stride + 1)
        fig9 = plt.figure()
        ax9 = fig9.add_subplot(111)
        ax9.plot(xs, trajectory[::step_stride, :, 0].T)
        ax9.set_xlim(5.5,6.5)
        for idx,line in enumerate(ax9.lines): line.set_color((cs[idx-1], 0.5, 0.5))
        ax9.plot(xs, trajectory[0, :, 0], '--', color='black',label='V_h(t=0)')
        ax9.plot(xs, trajectory[-1, :, 0], color='black', label='V_h(t=5)')
        plt.xlabel('fiber length [cm]')
        plt.ylabel('transmembrane Potental V_m [mV]')
        plt.legend()
        plt.show()

    ###### plot the transmembrane voltage colored by cluster membership
    if False:
        step_stride = 10
        fig10 = plt.figure()
        ax10 = fig10.add_subplot(111)
        ax10.set_xlim(5.5,6.5)
        ax10.plot(xs, trajectory[::step_stride, :, 0].T)
        #color lines
        labels_stride = labels[::step_stride]
        for idx in range(0, len(ax10.lines)-1):
            ax10.lines[idx].set_color(colors[labels_stride[idx]])
        for label in range(0, ncluster):
            for idx in range(0,len(ax10.lines)-1):
                if labels_stride[idx] == label:
                    ax10.lines[idx+1].set_label(f'cluster{label}')
                    break
        ax10.lines[0].set_linestyle('--')
        ax10.lines[0].set_color('black')
        ax10.lines[0].set_label('V_h(t=0)')
        ax10.lines[-1].set_color('black')
        ax10.lines[-1].set_label('V_h(t=5)')
        #label
        plt.xlabel('fiber length [cm]')
        plt.ylabel('transmembrane Potental V_m [mV]')
        plt.legend()
        plt.show()

    ######################################################### error estimates trajectories ####################################################################################
    ###### plot the iteration error trajectories
    if False:
        step_stride = 10
        fig11 = plt.figure()
        ax11 = fig11.add_subplot()
        l, b, h, w = .35, .7, .30, .30
        ax11in = ax11.inset_axes([l,b,w,h])
        ax11in.set_ylim(1.4e-3,2.0e-3)
        ax11in.set_xlim(0.925, 1.025)
        ax11in.set_yscale('log')
        ax11.indicate_inset_zoom(ax11in)
        ax11in2 = ax11.inset_axes([.7,.4,w,h])
        ax11in2.set_ylim(8e-08, 2e-07)
        ax11in2.set_xlim(3.9, 4)
        ax11in2.set_yscale('log')
        ax11.indicate_inset_zoom(ax11in2)

        ax11.set_yscale('log')
        cs = np.linspace(0,1, timesteps//step_stride+1)
        for k in range(0, timesteps, step_stride): 
            ax11.plot(np.arange(5), abs(e_iters[k,:]), color=(cs[k//step_stride], 0.5, 0.5))
            ax11in.plot(np.arange(5), abs(e_iters[k,:]), color=(cs[k//step_stride], 0.5, 0.5))
            ax11in2.plot(np.arange(5), abs(e_iters[k,:]), color=(cs[k//step_stride], 0.5, 0.5))
            fig11.set_size_inches(14,10)
            fig11.savefig('plots/error_trajectories/error_trajectory'+str(k)+'.png', dpi=100)
            print('\rplotting '+str(k)+' of '+str(timesteps), end='', flush=True)
        plt.show()

        # Build GIF
        print('\n building .gif file')
        with imageio.get_writer('plots/error_trajectories.gif', mode='I') as writer:
            for filename in range(0,timesteps, step_stride):
                image = imageio.v2.imread('plots/error_trajectories/error_trajectory' + str(filename)+ '.png')
                writer.append_data(image)
    
    ###### plot the iteration error trajectories colored by cluster membership
    if False:
        step_stride = 10
        fig12 = plt.figure()
        ax12 = fig12.add_subplot()
        l, b, h, w = .6, .5, .35, .35
        ax12in = ax12.inset_axes([l,b,w,h])
        ax12in.set_ylim(1.4e-3,2.0e-3)
        ax12in.set_xlim(0.925, 1.025)
        ax12in.set_yscale('log')
        ax12.indicate_inset_zoom(ax12in)
        ax12.set_yscale('log')
        cs = np.linspace(0,1, timesteps//step_stride+1)
        labels_stride = labels[::step_stride]
        for k in range(0, timesteps, step_stride): 
            ax12.plot(np.arange(5), abs(e_iters[k,:]), color=colors[labels_stride[k//step_stride]])
            ax12in.plot(np.arange(5), abs(e_iters[k,:]), color=colors[labels_stride[k//step_stride]])
            fig12.set_size_inches(14,10)
            fig12.savefig('plots/error_trajectories/error_trajectory'+str(k)+'.png', dpi=100)
            print('\rplotting '+str(k)+' of '+str(timesteps), end='', flush=True)
        plt.show()

        # Build GIF
        print('\nbuilding .gif file')
        with imageio.get_writer('plots/error_trajectories.gif', mode='I') as writer:
            for filename in range(0,timesteps, step_stride):
                image = imageio.v2.imread('plots/error_trajectories/error_trajectory' + str(filename)+ '.png')
                writer.append_data(image)

    ###### plot the iteration error trajectories colored by cg iterations needed
    if True:
        step_stride = 10
        fig13 = plt.figure()
        ax13 = fig13.add_subplot()
        l, b, h, w = .35, .7, .30, .30
        ax13in = ax13.inset_axes([l,b,w,h])
        ax13in.set_ylim(1.4e-3,2.0e-3)
        ax13in.set_xlim(0.925, 1.025)
        ax13in.set_yscale('log')
        ax13.indicate_inset_zoom(ax13in)
        ax13in2 = ax13.inset_axes([.7,.4,w,h])
        ax13in2.set_ylim(8e-08, 5e-07)
        ax13in2.set_xlim(3.9, 4)
        ax13in2.set_yscale('log')
        ax13.indicate_inset_zoom(ax13in2)
        ax13.set_yscale('log')
        cs = np.linspace(0,1, timesteps//step_stride+1)
        labels_stride = labels[::step_stride]
        for k in range(0, timesteps, step_stride): 
            if k <= cut_index_true:
                ax13.plot(np.arange(5), abs(e_iters[k,:]), color='cyan')
                ax13.plot(np.arange(5), abs(e_discs[k,:]), color='cyan')
                ax13in.plot(np.arange(5), abs(e_iters[k,:]), color='cyan')
                ax13in2.plot(np.arange(5), abs(e_iters[k,:]), color='cyan')
                ax13in2.plot(np.arange(5), abs(e_discs[k,:]), color='cyan')
            else:
                ax13.plot(np.arange(5), abs(e_iters[k,:]), color='magenta')
                ax13.plot(np.arange(5), abs(e_discs[k,:]), color='magenta')
                ax13in.plot(np.arange(5), abs(e_iters[k,:]), color='magenta')
                ax13in2.plot(np.arange(5), abs(e_iters[k,:]), color='magenta')
                ax13in2.plot(np.arange(5), abs(e_discs[k,:]), color='magenta')
            #ax13.plot(np.arange(5), abs(e_discs[k,:]), color='black', linestyle='--')
            fig13.set_size_inches(14,10)
            #plt.pause(0.0001)
            ax13.set_xlabel('# CG Iterations')
            ax13.set_ylabel('error estimates (discretization & iteration)')
            fig13.savefig('plots/error_trajectories/error_trajectory'+str(k)+'.png', dpi=100)
            print('\rplotting '+str(k)+' of '+str(timesteps), end='', flush=True)
        ax13.lines[-1].set_label('5 CG Iterations needed')
        ax13.lines[-1].set_label('6 CG Iterations needed')

        plt.show()

        # Build GIF
        print('\nbuilding .gif file')
        with imageio.get_writer('plots/error_trajectories.gif', mode='I') as writer:
            for filename in range(0,timesteps, step_stride):
                image = imageio.v2.imread('plots/error_trajectories/error_trajectory' + str(filename)+ '.png')
                writer.append_data(image)
 
    ###### plot the first error estimates in a 2d plot
    if False:
        fig14 = plt.figure()
        ax14 = fig14.add_subplot()
        ax14.scatter(e_discs[:cut_index_true,0], e_iters[:cut_index_true,0], color='cyan')
        ax14.scatter(e_discs[cut_index_true:,0], e_iters[cut_index_true:,0], color='magenta')
        ax14.set_xlabel('e_discs')
        ax14.set_ylabel('e_iters')
        plt.show()

    ###### plot the first two iteration error estimates in a 2d plot
    if False:
        fig14 = plt.figure()
        ax14 = fig14.add_subplot()
        ax14.scatter(e_iters[:cut_index_true,0], e_iters[:cut_index_true,1], color='cyan')
        ax14.scatter(e_iters[cut_index_true:,0], e_iters[cut_index_true:,1], color='magenta')
        ax14.set_xlabel('e_iters[0]')
        ax14.set_ylabel('e_iters[1]')
        plt.show()

    ###### plot the first two discretization error estimates in a 2d plot
    if False:
        fig14 = plt.figure()
        ax14 = fig14.add_subplot()
        ax14.scatter(e_discs[:cut_index_true,0], e_discs[:cut_index_true,1], color='cyan')
        ax14.scatter(e_discs[cut_index_true:,0], e_discs[cut_index_true:,1], color='magenta')
        ax14.set_xlabel('e_discs[0]')
        ax14.set_ylabel('e_discs[1]')
        plt.show()
 
if __name__ == '__main__':
    #read out.npy file
    trajectory = np.load('out/out_trajectories.npy')
    e_iters = np.load('out/out_itererrors.npy')
    e_discs = np.load('out/out_discerrors.npy')
    #e_iters = e_iters[:,:2]
    #e_discs = e_discs[:,:2]

    #read csv files
    samples = pd.read_csv('out/samples.csv')
    samples = samples.to_numpy()
    data = pd.read_csv('out/data.csv')
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

    #take max of potential trajectories
    trajectories_max = np.max(trajectory[:-1,:,0], axis=1)

    #clustering
    #samples = trajectories_max.reshape(-1,1)
    samples = e_iters
    #samples = np.hstack((e_discs,e_iters))
    ncluster = 2
    labels, cut_idxs = clustering(samples, ncluster, scaling=False)

    #plot results
    make_plots(labels)
