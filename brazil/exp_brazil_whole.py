import numpy as np
import matplotlib.pyplot as plt
from CM_rand import *
from utils import *
import time


def twoway_marignal(sizes):
    res = []
    n = len(sizes)
    for i in range(n):
        for j in range(i + 1, n):
            mat = np.eye(sizes[i])
            mat2 = np.eye(sizes[j])
            for k in range(i-1, -1, -1):
                eye = np.ones([1, sizes[k]])
                mat = np.kron(eye, mat)
            for k in range(i+1, j):
                eye = np.ones([1, sizes[k]])
                mat = np.kron(mat, eye)
            mat = np.kron(mat, mat2)
            for k in range(j+1, n):
                eye = np.ones([1, sizes[k]])
                mat = np.kron(mat, eye)
            res.append(mat)
    result = np.concatenate(res, axis=0)
    return result


def oneway_marignal(sizes):
    res = []
    for i, size in enumerate(sizes):
        mat = np.eye(size)
        for j in range(i-1, -1, -1):
            eye = np.ones([1, sizes[j]])
            mat = np.kron(eye, mat)
        for j in range(i+1, len(sizes)):
            eye = np.ones([1, sizes[j]])
            mat = np.kron(mat, eye)
        res.append(mat)
    result = np.concatenate(res, axis=0)
    return result


if __name__ == '__main__':
    np.random.seed(43)
    dataset = np.load("brazil.npy")
    size, _ = np.shape(dataset)
    print("get data successfully")

    sizes = [101, 2]

    sigma_list = [0.1, 0.2, 0.3, 0.4, 0.5, 1/np.sqrt(2), 1, 2, 4]
    y_list = [1, 2, 3, 4, 5, 6, 8, 10, 12]
    x_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    for sigma in sigma_list:
        B1 = oneway_marignal(sizes)
        m, n = np.shape(B1)
        S1 = np.eye(m)*sigma**2*2
        B2 = np.eye(n)
        S2 = np.eye(np.shape(B2)[0])*sigma**2

        tic = time.time()
        args = config()

        args.sigma = sigma
        args.sigma_noise = sigma
        args.param_x = 0.3
        args.param_y = 3
        k_list = np.sum(B1, axis=1)
        args.k_list = k_list

        print("-------------- sigma: ", sigma, "------------------")
        com_mech = run_experiment(args, dataset, B1, S1, B2, S2)
        toc = time.time()
        print("time is: ", toc-tic)

    for y in y_list:
        sigma = 0.5
        B1 = oneway_marignal(sizes)
        m, n = np.shape(B1)
        S1 = np.eye(m) * sigma ** 2 * 2
        B2 = np.eye(n)
        S2 = np.eye(np.shape(B2)[0]) * sigma ** 2

        tic = time.time()
        args = config()

        args.sigma = sigma
        args.sigma_noise = sigma
        args.param_x = 0.3
        args.param_y = y
        k_list = np.sum(B1, axis=1)
        args.k_list = k_list

        print("-------------- y: ", y, "------------------")
        com_mech = run_experiment(args, dataset, B1, S1, B2, S2)
        toc = time.time()
        print("time is: ", toc - tic)


    for x in x_list:
        sigma = 0.5
        B1 = oneway_marignal(sizes)
        m, n = np.shape(B1)
        S1 = np.eye(m) * sigma ** 2 * 2
        B2 = np.eye(n)
        S2 = np.eye(np.shape(B2)[0]) * sigma ** 2

        tic = time.time()
        args = config()

        args.sigma = sigma
        args.sigma_noise = sigma
        args.param_x = x
        args.param_y = 3
        k_list = np.sum(B1, axis=1)
        args.k_list = k_list

        print("-------------- x: ", x, "------------------")
        com_mech = run_experiment(args, dataset, B1, S1, B2, S2)
        toc = time.time()
        print("time is: ", toc - tic)
