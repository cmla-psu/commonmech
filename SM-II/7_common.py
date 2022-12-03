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
import time
from softmax import configuration, workload, matrix_query, func_var


def pcostmat(mat, cov):
    return mat.T @ np.linalg.inv(cov) @ mat


def pdiag(mat, cov):
    return np.diag(pcostmat(mat, cov))


def plb(mat, cov):
    return np.max(pdiag(mat, cov)) / 2.0


if __name__ == '__main__':
    start = time.time()
    np.random.seed(0)
    # obj = np.load("..//signal_select//cm-b1b2.npy", allow_pickle=True)
    obj = np.load("..//brazil//cm-brazil.npy", allow_pickle=True)
    # obj = np.load("..//finegrain_signal//cm-age-234.npy", allow_pickle=True)

    track = obj.item()
    work = track['mat']
    covar = track['cov']
    bound = np.diag(covar)
    param_m, param_n = work.shape

    # configuration parameters
    args = configuration()
    args.init_mat = 'id_index'
    args.basis = 'id'
    args.maxitercg = 5

    index = np.eye(param_m)
    basis = work

    mat_opt = matrix_query(args, basis, index, bound)
    mat_opt.optimize()
    mat_cov = mat_opt.cov/np.max(mat_opt.f_var)
    acc = func_var(mat_cov, index)

    plb_total = plb(work, covar)
    plb_opt = plb(work, mat_cov)
    plb_ratio = plb_opt / plb_total * 100
    print("The percentage of plb saved is, ", plb_ratio)

    end = time.time()
    print("time: ", end-start)

