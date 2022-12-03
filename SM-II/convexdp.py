# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2022 cmla-psu/Yingtai Xiao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
# import cupy as np
# from softmax import WRange, WRelated


def is_pos_def(A):
    """
    Check positive definiteness.

    Return true if psd.
    """
    zero = np.zeros_like(A)
    # first check symmetry
    if np.allclose(A, A.T, 1e-5, 1e-8):
        # whether cholesky decomposition is successful
        try:
            L = np.linalg.cholesky(A)
            return L, True
        except np.linalg.LinAlgError:
            return zero, False
    else:
        return zero, False


def Hx(iX, S, G):
    """Hess tiems x."""
    mat = -iX @ S @ G - G @ S @ iX
    return mat


def ConvexDP(W):
    """Newton Approximation."""
    _, n = W.shape
    maxiter = 30
    maxiterls = 50
    maxitercg = 5
    theta = 1e-3
    accuracy = 1e-5
    beta = 0.5
    sigma = 1e-4

    X = np.eye(n)
    iD = np.eye(n)
    V = W.T @ W
    V = V + theta*np.mean(np.diag(V)) * iD
    A = np.linalg.cholesky(X)
    iX = np.linalg.solve(A.T, np.linalg.solve(A, iD))
    G = -iX @ V @ iX
    fcurr = np.sum(V * iX)
    history = []

    for iteration in range(maxiter):
        # find search direction
        if iteration == 0:
            D = -G
            np.fill_diagonal(D, 0)
            i = -1
        else:
            D = np.zeros([n, n])
            R = -G - Hx(iX, D, G)
            np.fill_diagonal(R, 0)
            P = R
            rsold = np.sum(R * R)
            for i in range(maxitercg):
                Hp = Hx(iX, P, G)
                alpha = rsold/np.sum(P * Hp)
                D = D + alpha*P
                np.fill_diagonal(D, 0)
                R = R - alpha*Hp
                np.fill_diagonal(R, 0)
                rsnew = np.sum(R * R)
                if rsnew < 1e-10:
                    break
                P = R + rsnew/rsold * P
                rsold = rsnew

        # find step size
        delta = np.sum(D * G)
        Xold = X
        flast = fcurr
        history.append(fcurr)
        for j in range(maxiterls):
            alpha = beta**j
            X = Xold + alpha*D
            A, flag = is_pos_def(X)
            if flag:
                iX = np.linalg.solve(A.T, np.linalg.solve(A, iD))
                G = -iX @ V @ iX
                fcurr = np.sum(V * iX)
            if fcurr <= flast + alpha * sigma * delta:
                break
        # print(iteration, fcurr, np.linalg.norm(D), i, j)

        if i == maxiterls:
            X = Xold
            fcurr = flast
            break
        if np.abs((flast-fcurr)/flast) <= accuracy:
            break

    return np.linalg.cholesky(X)


def ca_variance(W, A, s):
    """
    Calculate the maximum variance for a single query in workload W.

    using HB method with privacy cost at most s.

    Parameters
    ----------
    A is the hb tree structure
    s is the privacy cost
    """
    m, n = A.shape
    mI = np.eye(m)
    # inverse of matrix A
    pA = np.linalg.inv(A)
    sigma = np.sqrt(1/s)
    Var = W @ pA.T
    BXB = Var @ (sigma**2*mI) @ Var.T
    a = np.diag(BXB)
    return a


def wCA(work, bound, pcost):
    """Re-weighted workload for CA, w[i] = c[i]."""
    m, n = work.shape
    work_s = np.copy(work)
    work_s = work_s + 0.0
    for i in range(m):
        work_s[i, :] = work[i, :]/bound[i]
    strat = ConvexDP(work_s)
    var_s = ca_variance(work_s, strat, pcost)
    var_s = var_s * bound * bound
    return strat, var_s


def wCA2(work, bound, pcost):
    """Re-weighted workload for CA, w[i] = sqrt(c[i])."""
    m, n = work.shape
    work_s = np.copy(work)
    work_s = work_s + 0.0
    for i in range(m):
        work_s[i, :] = work[i, :]/np.sqrt(bound[i])
    strat = ConvexDP(work_s)
    var_s = ca_variance(work_s, strat, pcost)
    var_s = var_s * bound
    return strat, var_s


