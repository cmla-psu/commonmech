import numpy as np
from utils import *
from class_finegrain import*
import time
from scipy.linalg import block_diag


def range_query(k, male=True, size_n=206):
    """Range query for k continuous age.

    male = True: count the male age
    male = False: count the female age
    """
    age_size = size_n // 2
    size_k = age_size // k
    query = np.zeros([k, size_n])
    if male:
        start = 0
    else:
        start = age_size
    for i in range(k):
        query[i, start+i*size_k: start+i*size_k+size_k] = 1
    # the remaining elements are included in the last query
    query[-1, start+k*size_k: start+age_size] = 1

    return query


def gender_age(k, sigma=1, size=206):
    """Gender age k queries."""
    q1 = range_query(k, True, size)
    q2 = range_query(k, False, size)
    query = np.concatenate((q1, q2))
    m = np.shape(query)[0]
    cov = np.eye(m) * sigma ** 2
    return query, cov


def age_group(group, sigma=1):
    """Sex by Age 4 categories."""
    size_n = group[-1]
    age_size = group[-1]
    group_size = len(group) - 1
    query = np.zeros([group_size*2, size_n*2])
    for i in range(1, group_size + 1):
        left = group[i-1]
        right = group[i]
        query[i-1, left:right] = 1
        query[group_size + (i-1), age_size + left: age_size + right] = 1
        # print(group_size + (i-1), age_size + left, age_size + right)
    cov = np.eye(group_size*2)
    return query, cov*sigma**2


def check_pst(Pc, P1):
    """Check if Pc <= P1 and Pc <= P2"""
    vec1, mat1 = np.linalg.eigh(P1 - Pc)
    check1 = np.all(vec1 >= -1e-8)
    return check1


def check_common(Pc, P1, P2):
    """Check if Pc <= P1 and Pc <= P2"""
    vec1, mat1 = np.linalg.eigh(P1 - Pc)
    vec2, mat2 = np.linalg.eigh(P2 - Pc)
    check1 = np.all(vec1 >= -1e-8)
    check2 = np.all(vec2 >= -1e-8)
    return check1 and check2


if __name__ == '__main__':
    np.random.seed(43)
    dataset = np.load("..\\datafile\\AgeGender.npy")

    args = config()
    args.repeat = 1
    args.sigma = 8
    args.sigma_noise = args.sigma
    args.param_x = 0.5
    args.param_y = 20
    print(args)
    tic = time.time()

    group4 = [0, 18, 45, 65, 103]
    group9 = [0, 5, 18, 25, 35, 45, 55, 65, 75, 103]
    group23 = [0, 5, 10, 15, 18, 20, 21, 22, 25, 30, 35, 40, 45, 50, 55, 60, 62, 65, 67, 70, 75, 80, 85, 103]

    sigma_list = [2, 4, 6, 8, 10, 12]
    y_list = [2, 7, 12, 17, 22, 27]
    for y in y_list:
        size = 103
        sigma = 8
        args.sigma = sigma
        args.param_y = y
        args.param_x = 0.5
        args.sigma_noise = args.sigma
        B1, S1 = np.ones([1, size*2]), np.eye(1)*sigma**2
        B2, S2 = age_group(group4, sigma)
        B3, S3 = age_group(group9, sigma)
        B4, S4 = age_group(group23, sigma)

        mech1 = Mechanism(args, B1, S1)
        mech2 = Mechanism(args, B2, S2)
        mech3 = Mechanism(args, B3, S3)
        mech4 = Mechanism(args, B4, S4)

        mech_list = [mech1, mech2, mech3, mech4]
        CM = CommonMechanism(args, mech_list, B1)
        CM234 = CommonMechanism(args, mech_list[1:], B2)
        CM34 = CommonMechanism(args, mech_list[2:], B3)
        cm_list = [CM, CM234, CM34]

        print("&&&&&&&&&&&&&& y is: ", y, "  &&&&&&&&&&&&&&&&&&&&&&&&&&")
        com_mech = run_experiment_adaptive(args, dataset, mech_list, cm_list)

    # x_list = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    # for x in x_list:
    #     args.param_x = x
    #     print("&&&&&&&&&&&&&& x is: ", x, "  &&&&&&&&&&&&&&&&&&&&&&&&&&")
    #     com_mech = run_experiment_adaptive(args, dataset, mech_list, cm_list)
    toc = time.time()
    print("time is: ", toc-tic)
