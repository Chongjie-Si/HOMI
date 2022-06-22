
# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np


import os
from sklearn.metrics.pairwise import rbf_kernel

import seaborn as sns
from dataload import *
from sklearn import model_selection, preprocessing
from sklearn.decomposition import PCA
from scipy.linalg import solve_sylvester
from numpy.linalg import inv, norm
import matplotlib.pyplot as plt
import scipy.io as scio


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


# exp(- ||xi-xj||^2 / tau)
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


def Weight_Matrix(X, Y, Xttt, Yttt, nfeature, nlabel, ntrain, beta=2, lam=1, gamma=0.01, iteration=100):
    s = 2
    print(s)
    lam1 = lam
    lam2 = lam
    lam3 = lam
    lam4 = lam
    W = np.random.random((nfeature, nlabel))
    t = np.random.random((nlabel, 1))
    z = np.random.random((nlabel, 1))
    B = np.random.random((nlabel, nlabel))
    nv = np.ones((ntrain, 1))
    q = [0]
    ham = [0, 0, 0, 0]
    if os.path.isfile('W.npy') and os.path.isfile('B0.npy') and os.path.isfile('bp.npy'):
        W = np.load('W.npy')
        B = np.load('B0.npy')
    else:
        if os.path.isfile('S.npy'):
            S = np.load('S.npy')
        else:
            #S = distance_L(X, s)
            #'''
            S = np.corrcoef(X)
            row, col = np.diag_indices_from(S)
            S[row, col] = np.zeros(ntrain)
            for i in range(ntrain):
                tem = S[i].copy()
                tkk = findKth(tem, s)
                S[i][S[i] < tkk] = 0
            #'''
        S = (S+S.T) / 2
        D = np.zeros((ntrain, ntrain))
        for i in range(ntrain):
            D[i][i] = sum(S[i])
        L = D - S
        '''
        for i in range(nlabel):
            Z[i][i] = sum(Q[i])
        L1 = Z - Q
        '''
        te = gamma/2*nv.T @ (L+L.T) @ nv

        for it in range(iteration):
            Tar = inv(gamma/2 * X.T @ (L+L.T) @ X) @ (X.T @ Y - X.T @ nv @ z.T - gamma/2 * X.T @ (L+L.T) @ (nv@z.T@B+nv @ t.T)@ B.T)
            Q = inv(gamma/2*X.T @ (L+L.T) @ X) @ (X.T @ X + lam1 * np.eye(nfeature))
            W = solve_sylvester(Q, B @ B.T, Tar)
            z = inv((ntrain + lam3)*np.eye(nlabel) + te*B@B.T) @ ((Y.T-W.T@X.T)@nv - gamma/2*B@(B.T@W.T@X.T + t@nv.T) @ (L+L.T) @ nv)
            B = inv(beta * Y.T @ Y + lam2 * np.eye(nlabel) + gamma/2 * (W.T@X.T+z@nv.T) @ (L+L.T) @ (X@W + nv @ z.T)) @ \
                (beta * Y.T @ Y - beta * (Y.T @ nv @ t.T) - gamma/2 * (W.T@X.T+z@nv.T) @ (L+L.T) @ nv @ t.T)
            t = (beta*(Y.T-B.T@Y.T)@nv - gamma/2*(B.T@W.T@X.T+B.T@z@nv.T) @ (L+L.T) @ nv) / (beta*ntrain + lam4 + te)
            q.append(norm(Y - X @ W - nv @ z.T) + beta * norm(Y - Y @ B - nv @ t.T) +
                     gamma * np.trace(((X @ W + nv @ z.T) @ B + nv @ t.T).T @ L @ ((X @ W + nv @ z.T) @ B + nv @ t.T)) +
                     lam1 * norm(W) + lam2 * norm(B) + lam3 * norm(t) + lam4 * norm(z))

            O1 = (Xttt @ W + np.ones((Xttt.shape[0], 1)) @ z.T) @ B + np.ones((Xttt.shape[0], 1)) @ t.T
            P1 = np.array(O1 > 0.5, dtype=int) * 1
            ham.append(hamming_loss(Yttt, P1, O1))
            if abs(q[-1] - abs(q[-2])) <= 0.000005:
                print(it)
                break
            '''
            if abs(ham[-1] - ham[-5]) <= 0.0000001:
                print(it)
                break
            '''

        q = np.array(q)
    return W, B, z, t, q


# '''
# 阈值确定 确定分类标签的阈值
def run(Xt, Yt):
    X0, Y0 = Xt, Yt
    X0 = np.array(X0, dtype=float)
    p = PCA(n_components=min(X0.shape[0], X0.shape[1]))
    X0 = p.fit_transform(X0)
    X0 = preprocessing.scale(X0)
    Y0 = np.array(Y0 > 0, dtype=int)
    hamming = list()
    pre = list()
    acc = list()
    avf = list()
    oneeoor = list()
    rkloss = list()
    MAP = list()
    MAAUC = list()
    q0 = []
    cross = model_selection.KFold(n_splits=5, shuffle=True)
    for train, test in cross.split(X0, Y0):
        #X_train = rbf_kernel(X0[train], X0[train], 0.01)
        #X_test = rbf_kernel(X0[test], X0[train], 0.01)
        X_train = X0[train]
        X_test = X0[test]
        W, B, z, t, q = Weight_Matrix(X_train, Y0[train], X_test, Y0[test], X_train.shape[1], Y0[train].shape[1], X_train.shape[0])
        O = (X_test @ W + np.ones((X_test.shape[0], 1)) @ z.T) @ B + np.ones((X_test.shape[0], 1)) @ t.T
        P = np.array(O > 0.5, dtype=int) * 1
        hamming.append(hamming_loss(Y0[test], P, O))
        pre.append(precision(Y0[test], P, O))
        acc.append(accuracy(Y0[test], P, O))
        avf.append(average_precision(Y0[test], P, O))
        oneeoor.append(one_error(Y0[test], P, O))
        rkloss.append(ranking_loss(Y0[test], P, O))
        MAP.append(macro_averaging_precision(Y0[test], P, O))
        MAAUC.append(macro_averaging_accuracy(Y0[test], P, O))
        q0.append(q)
        print(str((hamming[-1])))
        print(str(numpy.array(pre)[-1]))
        print(str((numpy.array(acc)[-1])))
        print(str((numpy.array(avf)[-1])))
        print(str((numpy.array(oneeoor)[-1])))
        print(str((numpy.array(rkloss)[-1])))
        print(str((numpy.array(MAP)[-1])))
        print(str((numpy.array(MAAUC)[-1])))
        print('\n')
        #np.save('data/B.npy', B)
        #return
        #break
    print(str(numpy.mean(numpy.array(hamming)).item()) + " +- " + str(numpy.std(numpy.array(hamming), ddof=1).item()))
    print(str(numpy.mean(numpy.array(pre)).item()) + " +- " + str(numpy.std(numpy.array(pre), ddof=1).item()))
    print(str(numpy.mean(numpy.array(acc)).item()) + " +- " + str(numpy.std(numpy.array(acc), ddof=1).item()))
    print(str(numpy.mean(numpy.array(avf)).item()) + " +- " + str(numpy.std(numpy.array(avf), ddof=1).item()))
    print(str(numpy.mean(numpy.array(oneeoor)).item()) + " +- " + str(numpy.std(numpy.array(oneeoor), ddof=1).item()))
    print(str(numpy.mean(numpy.array(rkloss)).item()) + " +- " + str(numpy.std(numpy.array(rkloss), ddof=1).item()))
    print(str(numpy.mean(numpy.array(MAP)).item()) + " +- " + str(numpy.std(numpy.array(MAP), ddof=1).item()))
    print(str(numpy.mean(numpy.array(MAAUC)).item()) + " +- " + str(numpy.std(numpy.array(MAAUC), ddof=1).item()))
    print('\n')
    print(str((hamming)))
    print(str(numpy.array(pre)))
    print(str((numpy.array(acc))))
    print(str((numpy.array(avf))))
    print(str((numpy.array(oneeoor))))
    print(str((numpy.array(rkloss))))
    print(str((numpy.array(MAP))))
    print(str((numpy.array(MAAUC))))
    # + " +- ", str(numpy.std(numpy.array(MAAUC), ddof=1).item())

    return q0[0]


#X = rbf_kernel(X, X, 0.01)
q = run(X, Y)
#np.save('scene.npy', q)
