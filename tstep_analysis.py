import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

trajectories = np.load('out/out_trajectories_1e-3.npy')
print(trajectories[:,:,0].shape)
Nx = trajectories.shape[1]
xs = np.linspace(0,11.9,Nx)

#calculate the index where the cgiters jump from 6 to 5
data = pd.read_csv('out/data_1e-3.csv')
data = data.to_numpy()
t_steps = data[:,0]
timesteps = int(t_steps[-1])
cgiters = data[:,1]

#calculate cuts
cut_idxs = [0]
label_old = cgiters[0]
for idx in range(0,timesteps):
    label_select = cgiters[idx]
    if label_select == label_old:
        continue
    else:
        cut_idxs.append(idx)
        label_old = label_select
cut_idxs.append(timesteps)
print('Cluster-Grenzen:',cut_idxs)

step_stride = 100
x_left = 500
x_right = 700
colors = ['red','orange','blue','green','purple','brown','pink','gray','olive','cyan','magenta', 'yellow', 'black', 'darkgreen', 'darkblue', 'deeppink']
fig1 = plt.figure()
ax1 = fig1.add_subplot()
for label in range(0, len(cut_idxs)):
    ax1.plot(xs[x_left:x_right], trajectories[cut_idxs[label-1]:cut_idxs[label]:step_stride, x_left:x_right, 0].T, color=colors[label])
plt.show()

#################################################################################################################################################################
trajectories = np.load('out/out_trajectories_1e-2.npy')
print(trajectories[:,:,0].shape)
Nx = trajectories.shape[1]
xs = np.linspace(0,11.9,Nx)

#calculate the index where the cgiters jump from 6 to 5
data = pd.read_csv('out/data_1e-2.csv')
data = data.to_numpy()
t_steps = data[:,0]
timesteps = int(t_steps[-1])
cgiters = data[:,1]

#calculate cuts
cut_idxs = [0]
label_old = cgiters[0]
for idx in range(0,timesteps):
    label_select = cgiters[idx]
    if label_select == label_old:
        continue
    else:
        cut_idxs.append(idx)
        label_old = label_select
cut_idxs.append(timesteps)
print('Cluster-Grenzen:',cut_idxs)

step_stride = 10
x_left = 500
x_right = 700
colors = ['red','orange','blue','green','purple','brown','pink','gray','olive','cyan','magenta', 'yellow', 'black', 'darkgreen', 'darkblue', 'deeppink']
fig2 = plt.figure()
ax2 = fig2.add_subplot()
for label in range(0, len(cut_idxs)):
    ax2.plot(xs[x_left:x_right], trajectories[cut_idxs[label-1]:cut_idxs[label]:step_stride, x_left:x_right, 0].T, color=colors[label])
plt.show()

#################################################################################################################################################################
trajectories = np.load('out/out_trajectories_1e-1.npy')
print(trajectories[:,:,0].shape)
Nx = trajectories.shape[1]
xs = np.linspace(0,11.9,Nx)

#calculate the index where the cgiters jump from 6 to 5
data = pd.read_csv('out/data_1e-1.csv')
data = data.to_numpy()
t_steps = data[:,0]
timesteps = int(t_steps[-1])
cgiters = data[:,1]

#calculate cuts
cut_idxs = [0]
label_old = cgiters[0]
for idx in range(0,timesteps):
    label_select = cgiters[idx]
    if label_select == label_old:
        continue
    else:
        cut_idxs.append(idx)
        label_old = label_select
cut_idxs.append(timesteps)
print('Cluster-Grenzen:',cut_idxs)

step_stride = 1
x_left = 500
x_right = 700
colors = ['red','orange','blue','green','purple','brown','pink','gray','olive','cyan','magenta', 'yellow', 'black', 'darkgreen', 'darkblue', 'deeppink']
fig2 = plt.figure()
ax2 = fig2.add_subplot()
for label in range(0, len(cut_idxs)):
    ax2.plot(xs[x_left:x_right], trajectories[cut_idxs[label-1]:cut_idxs[label]:step_stride, x_left:x_right, 0].T, color=colors[label])
plt.show()