import numpy as np
import os
import multiprocessing
import matplotlib.pyplot as plt
from multiprocessing.pool import ThreadPool as Pool
from CM_rand import *
from utils import *
import time
from dask.distributed import Client
from dask import config as cfg


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

    cfg.set({'distributed.scheduler.worker-ttl': None})
    args = config()
    args.repeat = 1
    args.cell_k = 2
    args.batch_size = 500
    args.n_workers = multiprocessing.cpu_count()
    print(args)

    data_ls = []
    # iterating over all files
    dirname = "..\\datafile\\HispRace"
    ext = '.npy'
    tic1 = time.time()
    for files in os.listdir(dirname):
        if files.endswith(ext):
            data_state = np.load(dirname + "\\" + files, allow_pickle=True)
            data_ls.append(data_state)
    toc1 = time.time()
    print("Time: ", toc1-tic1, "Finished data loading.")
    print("##########################################################")
    dataset = np.concatenate(data_ls)

    # dataset = dataset[:10_000, :]
    size = np.shape(dataset)[0]
    print("Number of blocks: ", size)
    batch_size = args.batch_size
    idx = np.arange(0, size, batch_size)
    idx = np.append(idx, size)
    databatch = [dataset[idx[i]: idx[i + 1], :] for i in range(len(idx) - 1)]

    tic1 = time.time()
    sigma_list = [0.1, 0.2, 0.5, 1/np.sqrt(2), 1, 2, 4, 5, 6, 8]
    y_list = [8, 10, 12]
    x_list = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    # for sigma in sigma_list:
    #     tic = time.time()
    #     args.sigma = sigma
    #     args.sigma_noise = np.sqrt(21)*args.sigma
    #     args.param_x = 0.5
    #     args.param_y = 5
    #     dims = 7
    #     B1, S1 = test_mechanism(choice=1, sigma=args.sigma, dims=dims)
    #     B2, S2 = test_mechanism(choice=2, sigma=args.sigma, dims=dims)
    #     with Client(n_workers=args.n_workers) as c:
    #         sims = [c.submit(run_experiment, args, data, B1, S1, B2, S2) for data in databatch]
    #         tot = c.submit(custom_sum, sims)
    #         count_ls = c.gather(tot)
    #     algo1, algo2, base, com, total = count_ls
    #     print("sigma: ", sigma)
    #     print("Algo1: ", algo1 / total, "Algo2: ", algo2 / total)
    #     print("Base: ", base / total, "Com: ", com / total)
    #     toc = time.time()
    #     print("Total time is: ", toc - tic, "Total: ", total)
    #     print("############################################################################")

    for y in y_list:
        tic = time.time()
        args.sigma = 2
        args.sigma_noise = np.sqrt(21)*args.sigma
        args.param_x = 0.5
        args.param_y = y
        dims = 7
        B1, S1 = test_mechanism(choice=1, sigma=args.sigma, dims=dims)
        B2, S2 = test_mechanism(choice=2, sigma=args.sigma, dims=dims)
        with Client(n_workers=args.n_workers) as c:
            sims = [c.submit(run_experiment, args, data, B1, S1, B2, S2) for data in databatch]
            tot = c.submit(custom_sum, sims)
            count_ls = c.gather(tot)
            del sims
            del tot
        algo1, algo2, base, com, total = count_ls
        print("y: ", y)
        print("Algo1: ", algo1 / total, "Algo2: ", algo2 / total)
        print("Base: ", base / total, "Com: ", com / total)
        toc = time.time()
        print("Total time is: ", toc - tic, "Total: ", total)
        print("############################################################################")

    for x in x_list:
        tic = time.time()
        args.sigma = 2
        args.sigma_noise = np.sqrt(21)*args.sigma
        args.param_x = x
        args.param_y = 5
        dims = 7
        B1, S1 = test_mechanism(choice=1, sigma=args.sigma, dims=dims)
        B2, S2 = test_mechanism(choice=2, sigma=args.sigma, dims=dims)
        with Client(n_workers=args.n_workers) as c:
            sims = [c.submit(run_experiment, args, data, B1, S1, B2, S2) for data in databatch]
            tot = c.submit(custom_sum, sims)
            count_ls = c.gather(tot)
            del sims
            del tot

        algo1, algo2, base, com, total = count_ls
        print("x: ", x)
        print("Algo1: ", algo1 / total, "Algo2: ", algo2 / total)
        print("Base: ", base / total, "Com: ", com / total)
        toc = time.time()
        print("Total time is: ", toc - tic, "Total: ", total)
        print("############################################################################")

    toc1 = time.time()
    print("Total time is: ", toc1 - tic1)
