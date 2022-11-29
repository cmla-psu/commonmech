import numpy as np
import os
import multiprocessing
import matplotlib.pyplot as plt
from multiprocessing.pool import ThreadPool as Pool
from CM_rand import *
from utils import *
import time
from dask.distributed import Client


def test_mechanism(choice=1, sigma=1, dims=7):
    """Create Mechanims (B1, S1)."""
    B1, S1 = 0, 0
    if choice == 1:
        B1 = marginal(dims, 2, "one")
        m = np.shape(B1)[0]
        S1 = np.eye(m) * dims * sigma ** 2
    if choice == 2:
        B1 = marginal(dims, 2, "two")
        m = np.shape(B1)[0]
        S1 = np.eye(m) * (3*dims) * sigma ** 2
    if choice == 3:
        n = 2**dims
        B1 = np.eye(n)
        S1 = np.eye(n) * sigma ** 2
    return B1, S1


def custom_sum(ls):
    ls = np.array(ls)
    return np.sum(ls, axis=0)


if __name__ == '__main__':
    np.random.seed(43)

    tic = time.time()
    args = config()
    args.repeat = 1
    args.cell_k = 32
    args.batch_size = 500
    args.n_workers = multiprocessing.cpu_count()

    # Params for experiment
    args.sigma = 2
    args.sigma_noise = args.sigma
    args.param_x = 0.5
    args.param_y = 5

    sigma = args.sigma
    dims = 7
    B1, S1 = test_mechanism(choice=2, sigma=sigma, dims=dims)
    B2, S2 = test_mechanism(choice=3, sigma=sigma, dims=dims)
    print(args)

    algo1_ls = []
    algo2_ls = []
    com_ls = []
    base_ls = []
    total_ls = []
    # iterating over all files
    dirname = "..\\datafile\\HispRace"
    ext = '.npy'
    for files in os.listdir(dirname):
        if files.endswith(ext):
            print(files, end='   ')  # printing file name of desired extension
            tic1 = time.time()
            dataset = np.load(dirname + "\\" + files, allow_pickle=True)
            size = np.shape(dataset)[0]
            print("Number of blocks: ", size)
            batch_size = args.batch_size
            idx = np.arange(0, size, batch_size)
            idx = np.append(idx, size)
            databatch = [dataset[idx[i]: idx[i + 1], :] for i in range(len(idx) - 1)]
            with Client(n_workers=args.n_workers) as c:
                sims = [c.submit(run_experiment, args, data, B1, S1, B2, S2) for data in databatch]
                tot = c.submit(custom_sum, sims)
                count_ls = c.gather(tot)

            algo1, algo2, base, com, total = count_ls
            print("Algo1: ", algo1/total * 100, "Algo2: ", algo2/total * 100)
            print("Base: ", base/total * 100, "Com: ", com/total * 100)
            toc1 = time.time()
            print("Time: ", toc1-tic1)
            print("########################################################")

            algo1_ls.append(algo1)
            algo2_ls.append(algo2)
            base_ls.append(base)
            com_ls.append(com)
            total_ls.append(total)

    algo1 = sum(algo1_ls)
    algo2 = sum(algo2_ls)
    base = sum(base_ls)
    com = sum(com_ls)
    total = sum(total_ls)
    print("Algo1: ", algo1 / total * 100, "Algo2: ", algo2 / total * 100)
    print("Base: ", base / total * 100, "Com: ", com / total * 100)
    toc = time.time()
    print("Total time is: ", toc - tic, "Total: ", total * 100)

