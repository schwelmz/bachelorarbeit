import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import lagrange
from scipy.interpolate import interp1d
from scipy.interpolate import barycentric_interpolate
import scipy as sp
import imageio
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
import numpy.polynomial.chebyshev as chebyshev

##################################################################################################
### auxiliary methods
##################################################################################################
def exponential_interpolation(x,y):
    [x1, x2] = x
    [y1, y2] = y
    b = (y2/y1)**(1/(x2-x1))
    a = y1/(b**x1)
    return [a,b]    #a*b**x

def divided_diff(x,f):
    '''
    compute the divided differences table
    '''
    n = f.shape[0]
    coef = np.zeros([n,n])
    coef[:,0] = f
    for j in range(1,n):
        for i in range(n-j):
            coef[i, j] = (coef[i+1, j-1] - coef[i, j-1]) / (x[j+i] - x[i])
    return coef;

def newton_poly(coef, x_data, x):
    '''
    evaluate the newton polynomial at x using the horner scheme
    '''
    n = len(x_data)-1
    p = coef[n]
    for k in range(1,n+1):
        p = coef[n-k] + (x-x_data[n-k])*p
    return p

def lagrange_interpolant(x, f):
    N = len(x)
    a = np.ones(N)
    for i in range(N):
        for j in range(N):
            if i != j:
                a[i] *= 1/(x[i] - x[j])
    def p(x_):
        result = 0.
        for i in range(N):
            term_i = a[i] * f[i]
            for j in range(N):
                if i != j:
                    term_i *= x_ - x[j]
            result += term_i
        return result
    
    return p

def chebyshevspace(x_left, x_right, N):
    radius = (x_right - x_left) / 2.
    center = (x_right + x_left) / 2.    
    return center + radius * np.cos(np.pi - np.arange(N+1)*np.pi/N)
##################################################################################################
### functions
##################################################################################################
def intersection():
    intersect_list = []
    x_end = e_iters.shape[1]
    fig1 = plt.figure()
    fig1.set_size_inches(14,10)
    ax1 = fig1.add_subplot(211)
    ax2 = fig1.add_subplot(212)
    for tdx in range(0,timesteps):
        print('\rplotting '+str(tdx)+' of '+str(timesteps), end='', flush=True)
        ### points
        y1 = e_iters[tdx,:]
        y1 = abs(y1[y1!=0])
        maxiter = y1.shape[0]
        y1 = y1[-2:]
        x = np.array([maxiter-1, maxiter])
        xfine = np.linspace(maxiter-1, maxiter, 10)
        ###
        y2 = e_discs[tdx,:]
        y2 = abs(y2[y2!=0])
        y2 = y2[-2:]
        ### second order intepolation
        [a1,b1] = exponential_interpolation(x,y1)   #a*b**x
        poly2 = lagrange(x,y2)
        [a2,b2] = poly2.coef[::-1]     #a + b*x
        y1_interp = a1*b1**xfine
        y2_interp = a2 + b2*xfine
        ### calculate intersection
        func = lambda x: a1*b1**x - b2*x -a2
        x_intersect = sp.optimize.fsolve(func,0)
        intersect_list.append(x_intersect)
        ### plot
        if False:
            ### ax1
            ax1.set_title(f'tdx={tdx}')
            ax1.plot(x,y1, color='blue', label='abs. iteration error (estimate)')
            ax1.plot(x,y2, color='orange', label='abs. discretisation error (estimate)')
            #ax1.plot(xfine, y1_interp, linestyle='--', color='black')
            #ax1.plot(xfine, y2_interp, linestyle='--', color='black')
            ax1.axvline(x=x_intersect, color='red', linestyle='--')
            ax1.set_yscale('log')
            ax1.set_xlim(4,6)
            ax1.set_ylim(1e-9,2e-6)
            ax1.set_xlabel('#CG Iterations')
            ax1.set_ylabel('error')
            ax1.legend()
            ### ax2
            ax2.plot(intersect_list,t_steps[:tdx+1], color='red', label='x-coordinate of ntersecn disc. & iter. error')
            ax2.axvline(x=x_intersect, color='red', linestyle='--')
            ax2.set_xlim(4,6)
            ax2.set_xlabel('#CG Iterations')
            ax2.set_ylabel('#timesteps')
            ax2.legend()
            plt.ylim(0,timesteps)
            plt.pause(0.001)
            #fig1.savefig('plots/intersection/intersection'+str(tdx)+'.png', dpi=100)
            ax1.cla()
            ax2.cla()
    if False:
        plt.show()
        print('\nbuild GIF file')
        #create .gif file
        print('\n building .gif file')
        with imageio.get_writer('plots/intersection.gif', mode='I') as writer:
            for filename in range(0,timesteps):
                image = imageio.v2.imread('plots/intersection/intersection' + str(filename)+ '.png')
                writer.append_data(image)

    #plot the behaviour of the intersection over time
    fig2 = plt.figure()
    ax21 = fig2.add_subplot(211)
    ax22 = fig2.add_subplot(212)
    ax21.plot(t_steps, intersect_list, label='x-coordinate of intersection disc. & iter. error')
    ax21.axhline(y=5, color='black', linestyle='--')
    ax21.legend()
    ax21.set_xlabel('#timesteps')
    ax21.set_ylabel('#CG iterations')
    e_discs_nonzero = np.ones(timesteps)
    e_iters_nonzero = np.ones(timesteps)
    for j in range(0,timesteps):
        e_discs_j = abs(e_discs[j,:])
        e_iters_j = abs(e_iters[j,:])
        e_discs_j = e_discs_j[e_discs_j!=0]
        e_iters_j = e_iters_j[e_iters_j!=0]
        e_discs_nonzero[j] = e_discs_j[-1]
        e_iters_nonzero[j] = e_iters_j[-1]
    ax22.plot(t_steps, e_discs_nonzero, label='abs. discretisation error (estimate)')
    ax22.plot(t_steps, e_iters_nonzero, label='abs. iteration error (estimate)')
    ax22.set_yscale('log')
    ax22.legend()
    ax22.set_xlabel('#timesteps')
    ax22.set_ylabel('error')
    plt.show()

def error_difference():
    e_discs_4 = abs(e_discs[:,4])
    e_iters_4 = abs(e_iters[:,4])
    e_diff = e_discs_4 - e_iters_4
    fig3 = plt.figure()
    ax3 = fig3.add_subplot()
    #calculate the lagrange interpolate
    x = t_steps[:10]
    y = e_discs_4[:10]
    c = divided_diff(x,y)
    x_extrap = np.arange(x[0],x[-1], 0.1)
    y_extrap = newton_poly(c, x, x_extrap)
    #plot
    #ax3.plot(t_steps, e_iters_4, label='e_iter')
    #ax3.plot(t_steps, e_discs_4, label='e_disc')
    #ax3.plot(t_steps, e_diff, label='e_disc - e_iter')
    plt.scatter(x,y)
    ax3.plot(x_extrap,y_extrap, label='extrapolation', color = 'black')
    ax3.axhline(y=0, linestyle='--', color='black')
    ax3.legend()
    ax3.set_xlabel('#timesteps')
    ax3.set_ylabel('error')
    ax3.set_title('Differenz der Fehler nach der 5ten Iteration')
    #ax3.set_ylim((-2e-7,2e-7))
    plt.show()

def newton_interpolation():
    e_diff = abs(e_discs[:,4]) - abs(e_iters[:,4])
    x = t_steps[100:400:20]
    y = e_diff[100:400:20]
    # get the divided difference coef
    a_s = divided_diff(x, y)[0, :]

    # evaluate on new data points
    x_new = np.arange(x[0], t_steps[-1], .1)
    print(a_s.shape, x.shape, x_new.shape)
    y_new = newton_poly(a_s, x, x_new)

    plt.figure(figsize = (12, 8))
    plt.plot(x, y, 'bo')
    plt.axhline(y=0, color='black', linestyle = '--')
    plt.plot(t_steps, e_diff, label='|e_disc|-|e_iter|')
    plt.plot(x_new, y_new, label='extrapolation')
    plt.ylim(-1.5e-7, 1.5e-7)
    plt.legend()
    plt.show()

def polynomial_regression():
    plt.figure(figsize = (12, 8))
    for tdx in range(300, 16200, 100):
       e_diff = abs(e_discs[:,4]) - abs(e_iters[:,4])
       x = t_steps[tdx-300:tdx]
       y = e_diff[tdx-300:tdx]
       curve_x = t_steps

       #curve_x = np.linspace(0,11,333)

       poly = PolynomialFeatures(degree=2, include_bias=False)
       poly_features = poly.fit_transform(x.reshape(-1, 1))

       ridge = Ridge(alpha=0.001, fit_intercept=True, normalize=True, copy_X=True, max_iter=None, tol=0.001, random_state=1)

       ridge.fit(poly_features, y)

       poly_features_curve = poly.fit_transform(curve_x.reshape(-1, 1))
       y_pred = ridge.predict(poly_features_curve)

       plt.scatter(x, y, label='Data')
       plt.plot(t_steps, e_diff[:], color='blue', label='|e_disc|-|e_iter|')
       plt.plot(curve_x, y_pred, label='extrapolation', linestyle='--', color='red')
       plt.axhline(y=0, color='black', linestyle = '--')
       plt.legend()
       plt.ylim(-1.5e-7, 1.5e-7)
       plt.show()
       plt.cla()

def chebyshev_interpolation():
    e_diff = abs(e_discs[:,4]) - abs(e_iters[:,4])

    plt.figure(figsize = (12, 8))
    for tdx in range(200, timesteps, 200):
        #chebyshev nodes
        x = chebyshevspace(tdx-200, tdx, 2)
        x = np.rint(x).astype('int')
        f_cheb = np.take(e_diff, x)
        
        #newton interpolation
        xx = np.linspace(tdx-200, timesteps,1000)
        coef = divided_diff(x,f_cheb)[0, :]
        f_inter = newton_poly(coef, x, xx)

        plt.plot(x, f_cheb, 'o', label='chebyshev nodes')
        plt.axhline(y=0, linestyle='--', color='black')
        plt.plot(t_steps, e_diff[:], color='blue', label='|e_disc|-|e_iter|')
        plt.plot(xx, f_inter, label='extrapolation')
        plt.legend()
        plt.ylim(-1.5e-7, 1.5e-7)
        plt.pause(0.5)
        plt.cla()

##################################################################################################
### main
##################################################################################################
if __name__ == "__main__":
    e_discs = np.load('out/out_discerrors.npy')
    e_iters = np.load('out/out_itererrors.npy')
    timesteps = e_discs.shape[0]
    t_steps = np.arange(timesteps)

    #last nonzero entries
    e_discs_nonzero = np.ones(timesteps)
    e_iters_nonzero = np.ones(timesteps)
    for j in range(0,timesteps):
        e_discs_j = abs(e_discs[j,:])
        e_iters_j = abs(e_iters[j,:])
        e_discs_j = e_discs_j[e_discs_j!=0]
        e_iters_j = e_iters_j[e_iters_j!=0]
        e_discs_nonzero[j] = e_discs_j[-1]
        e_iters_nonzero[j] = e_iters_j[-1]
    
    #test methods 
    #intersection()
    #error_difference()
    #newton_interpolation()
    #polynomial_regression()
    chebyshev_interpolation()

