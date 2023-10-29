import numpy as np
import time
import os
from measure import *
from dataload import *
from sklearn import model_selection, preprocessing
from sklearn.decomposition import PCA
from scipy.linalg import solve_sylvester
from numpy.linalg import inv, norm


# quick sort
##--------------------------------------------##
def findKth(s, k):
    return findKth_c(s, 0, len(s) - 1, k)


def findKth_c(s, low, high, k):
    m = partition(s, low, high)
    if m == len(s) - k:
        return s[m]
    elif m < len(s) - k:
        return findKth_c(s, m + 1, high, k)
    else:
        return findKth_c(s, low, m - 1, k)


def partition(s, low, high):
    pivot, j = s[low], low
    for i in range(low + 1, high + 1):
        if s[i] <= pivot:
            j += 1
            s[i], s[j] = s[j], s[i]
    s[j], s[low] = s[low], s[j]
    return j
##--------------------------------------------##


# Construct Distance Matrix
def distance_L(X, k, tau=10):
    n = X.shape[0]
    S = np.zeros((n, n))
    for i in range(n):
        S[i] = np.array([np.exp(-norm(X[i]-X[j])**2/tau) for j in range(n)])
        S[i, i] = 0
        temp = S[i]
        t = findKth(temp, k+1)
        S[i][S[i] <= t] = 0
    return S


def Weight_Matrix(X, Y, beta=2, lam=1, gamma=1, s=10, iteration=100):
    # feature dimension
    nf = X.shape[1]
    # num of label
    nl = Y.shape[1]
    # num of sample
    nt = X.shape[0]
    
    # initialize
    W = np.random.random((nf, nl))
    t = np.random.random((nl, 1))
    z = np.random.random((nl, 1))
    B = np.random.random((nl, nl))
    
    nv = np.ones((nt, 1))
    
    q_last = 0
    q_current = 0

    # construct L
    S = np.corrcoef(X)
    row, col = np.diag_indices_from(S)
    S[row, col] = np.zeros(nt)
    for i in range(nt):
        tem = S[i].copy()
        tkk = findKth(tem, s)
        S[i][S[i] < tkk] = 0
    S = (S+S.T) / 2
    D = np.zeros((nt, nt))
    for i in range(nt):
        D[i][i] = sum(S[i])
    L = D - S

    te = gamma/2*nv.T @ (L+L.T) @ nv

    for it in range(iteration):
        Tar = inv(gamma / 2 * X.T @ (L + L.T) @ X) @ (X.T @ Y - X.T @ nv @ z.T -  \
                                                    gamma / 2 * X.T @ (L + L.T) @ (nv @ z.T @ B + nv @ t.T) @ B.T)
        Q = inv(gamma / 2 * X.T @ (L + L.T) @ X) @ (X.T @ X + lam * np.eye(nf))
        W = solve_sylvester(Q, B @ B.T, Tar)
        
        z = inv((nt + lam) * np.eye(nl) + te * B @ B.T) @ ((Y.T - W.T @ X.T) @ nv - \
                                                    gamma / 2 * B @ (B.T @ W.T @ X.T + t @ nv.T) @ (L + L.T) @ nv)
        
        B = inv(beta * Y.T @ Y + lam * np.eye(nl) + gamma / 2 * (W.T @ X.T + z @ nv.T) @ (L + L.T) @ (X @ W + nv @ z.T)) @ \
            (beta * Y.T @ Y - beta * (Y.T @ nv @ t.T) - gamma / 2 * (W.T @ X.T + z @ nv.T) @ (L + L.T) @ nv @ t.T)
            
        t = (beta * (Y.T - B.T @ Y.T) @ nv - gamma / 2 * (B.T @ W.T @ X.T + B.T @ z @ nv.T) @ (L + L.T) @ nv) / (beta * nt + lam + te)
        
        q_current = (norm(Y - X @ W - nv @ z.T) + beta * norm(Y - Y @ B - nv @ t.T) +
                 gamma * np.trace(((X @ W + nv @ z.T) @ B + nv @ t.T).T @ L @ ((X @ W + nv @ z.T) @ B + nv @ t.T)) +
                 lam * norm(W) + lam * norm(B) + lam * norm(t) + lam * norm(z))

        if abs(q_current - q_last) <= 0.000005:
            break
        
        q_last = q_current
        
    return W, B, z, t


def run(data_X, data_Y):
    X0, Y0 = data_X, data_Y
    X0 = np.array(X0, dtype=float)
    p = PCA(n_components=min(X0.shape[0], X0.shape[1]))
    X0 = p.fit_transform(X0)
    X0 = preprocessing.scale(X0)
    Y0 = np.array(Y0 > 0, dtype=int)

    cross = model_selection.KFold(n_splits=5, shuffle=True)
    for train, test in cross.split(X0, Y0):
        X_train = X0[train]
        X_test = X0[test]
        Y_train = Y0[train]
        Y_test = Y0[test]

        W, B, z, t = Weight_Matrix(X_train, Y_train)
        O = (X_test @ W + np.ones((X_test.shape[0], 1)) @ z.T) @ B + np.ones((X_test.shape[0], 1)) @ t.T
        P = np.array(O > 0.5, dtype=int) * 1
        
        print(str((hamming_loss(Y_test, P, O))))

    return 0

run(X, Y)