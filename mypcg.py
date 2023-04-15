import numpy as np
import scipy as sp
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from math import sqrt
import pprint
import time
import scipy
import time

def make_laplace(N, hx = 1, bounds=None):
    if hasattr(hx, "__len__"):
        """
        0    1    2
        |----|----|----|--
        h0   h1   h2
        """
        assert(len(hx) == N - 1), f"len(hx) = {len(hx)}, N = {N}"
        h = lambda i: hx[i]
    else:
        h = lambda i: hx


    rows = []
    cols = []
    vals = []
    for i in range(N):
        if bounds=='skip' and i in [0, N-1]:
            continue
        if bounds=='dirichlet' and i in [0, N-1]:
            rows.append(i)
            cols.append(i)
            vals.append(1)
            continue
        if bounds=='neumann' and i in [0, N-1]:
            if i == 0:
                rows.append(i)
                cols.append(i)
                vals.append(-1/h(0))
                rows.append(i)
                cols.append(i+1)
                vals.append(1/h(0))
            else:
                rows.append(i)
                cols.append(i-1)
                vals.append(1/h(N-2))
                rows.append(i)
                cols.append(i)
                vals.append(-1/h(N-2))
            continue

        if i != 0:
            rows.append(i)
            cols.append(i-1)
            vals.append(1/h(i-1)) # ∇φ_i ∇φ_i-1
        rows.append(i)
        cols.append(i)
        vals.append(-1/h(i-1) - 1/h(i)) # ∇φ_i ∇φ_i
        if i != N-1:
            rows.append(i)
            cols.append(i+1)
            vals.append(1/h(i)) # ∇φ_i ∇φ_i+1
    # return as Δ =  <∇φ, ∇φ>
    return -sparse.csr_matrix((vals, (rows, cols)), shape=(N,N))

def random_symmetric_matrix(n):
    np.random.seed(1234)
    A = np.random.normal(size=(n,n))
    A = A + np.eye(n)*5
    A = A @ A.T
    return scipy.sparse.csc_matrix(A)

def cg(A, b, x0, ilu=False, tol=1e-6, max_iter=None):
    # Initialization x = x0.copy()
    x = x0.copy()
    r = b - A @ x
    if ilu:
        M = sparse.linalg.spilu(A)
    else:
        M = None
    if M is None:
        z = r
    else:
        z = M.solve(r)
    p = z.copy()
    rsold = r.T @ z

    # Iterations
    for i in range(max_iter) if max_iter else range(1, 10000):
        Ap = A @ p
        alpha = rsold / (p.T @ Ap)
        x += alpha * p
        r -= alpha * Ap
        if np.linalg.norm(r) < tol:
            break
        if M is None:
            z = r
        else:
            z = M.solve(r)
        rsnew = r.T @ z
        beta = rsnew / rsold
        p = z + beta * p
        rsold = rsnew

    return x

def ichol(A):
    #incomplete cholesky: L_kj = 0 if A_kj = 0
    n = A.shape[0]
    L = np.zeros([n,n])

    for k in range(0,n-1):
        L[k,k] = sqrt(A[k,k])
        for j in range(k+1,n):
            L[k,j] = A[k,j]/L[k,k]
        for i in range(k+1,n):
            for j in range(i,n):
                if A[i,j] != 0:
                    A[i,j] = A[i,j] - L[k,i]*L[k,j]
    L[-1,-1] = sqrt(A[-1,-1])
    return L.T

def ichol_cg(A,b,x0=1,tol=1e-5,maxiter=None,convcontrol=None):
    b = b*x0
    n = A.shape[0]
    if maxiter==None:
        maxiter = n*10
    x = np.zeros(A.shape[0])

    #compute incomplete cholesky decomposition
    L = ichol(A)
    L_inv = np.linalg.inv(L)

    #initialize r and p
    r = b-A @ x
    r_ = L_inv @ r
    p = L_inv.T @ r
    alpha = np.inner(r_,r_)

    e_iter = 0
    e_disc = 0
    k=0
    exit_code = -1
    while (exit_code < 0):
        v = A @ p
        lam = alpha/np.inner(v,p)
        x = x + lam * p
        r_ = r_ - lam * L_inv @ v
        alphanew = np.inner(r_,r_)
        p = L_inv.T @ r + alphanew/alpha*p
        alpha = alphanew
        k+=1
        if  convcontrol is not None:
            e_disc, e_iter = convcontrol(x,x0)
            if abs(e_iter) < abs(e_disc):
                exit_code = 2
        elif(np.sqrt(alpha)<=tol):
            exit_code = 0
        if(k>=maxiter):
            exit_code = k
    return L_inv.T@x, exit_code


#build system
n = 10000
hx = 1/n
#A = random_symmetric_matrix(n)
#b = np.random.rand(n)
A = make_laplace(n, hx)
A = sparse.csc_matrix(A)
b = np.ones(n)*hx

#no precondtioner
start = time.time()
x_true = scipy.sparse.linalg.cg(A,b)[0]
stop = time.time()
print('no preconditioner, time taken:', stop-start)

#ILU preconditioner
start = time.time()
x1 = cg(A,b,np.zeros(n),ilu=True)
stop = time.time()
print('ILU preconditioner, time taken:', stop-start)

xs = np.linspace(0,1,n)
plt.plot(xs,x1, label = 'ILU')
plt.plot(xs,x_true,color='black',linestyle='--', label='no precond.')
plt.legend()
plt.show()
