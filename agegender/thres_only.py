import numpy as np
from utils import *
from class_thres import*
import time
from scipy.linalg import block_diag
from sklearn.linear_model import LogisticRegression


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
    cov = np.eye(group_size*2) * sigma ** 2
    return query, cov


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
    dataset = np.load("..\\datafile\\AgeGender.npy")
    size = dataset.shape[0]

    group4 = [0, 18, 45, 65, 103]
    group9 = [0, 5, 18, 25, 35, 45, 55, 65, 75, 103]
    group23 = [0, 5, 10, 15, 18, 20, 21, 22, 25, 30, 35, 40, 45, 50, 55, 60, 62, 65, 67, 70, 75, 81, 85, 103]

    vec_k1 = np.array([8])
    vec_k2 = np.array([2, 3, 2, 2, 2, 3, 2, 2])
    vec_k3 = np.array([1, 3, 4, 2, 2, 2, 3, 3, 3, 1, 3, 4, 2, 2, 2, 3, 3, 3])

    tic = time.time()
    size_q = 206
    args = config()

    sigma_list = [2, 4, 6, 8, 10, 12]
    y_list = [2, 7, 12, 17, 22, 27]
    x_list = np.arange(0.2, 0.9, 0.1)
    for sigma in [8]:
        # sigma = 8
        args.sigma = sigma
        args.sigma_noise = sigma
        args.param_x = 0.5
        args.param_y = 20
        args.gamma = 1e-3
        print("&&&&&&&&&&&&&& sigma is: ", sigma, "  &&&&&&&&&&&&&&&&&&&&&&&&&&")
        B1, S1 = np.ones([1, size_q]), np.eye(1) * sigma ** 2
        B2, S2 = age_group(group4, sigma)
        B3, S3 = age_group(group9, sigma)
        B4, S4 = age_group(group23, sigma)

        mech1 = Mechanism(args, B1, S1)
        mech2 = Mechanism(args, B2, S2)
        mech3 = Mechanism(args, B3, S3)
        mech4 = Mechanism(args, B4, S4)
        mech_list = [mech1, mech2, mech3, mech4]

        pop_list = [[] for i in range(4)]

        for i in range(size):
            data = dataset[i]
            population = np.sum(data)
            y_true1 = mech1.B0 @ data
            is_mech2 = satisfy_user_target(args, y_true1, vec_k1)
            if is_mech2 == -1:
                true_id = 0
            else:
                y_true2 = mech2.B0 @ data
                is_mech3 = satisfy_user_target(args, y_true2, vec_k2)
                if is_mech3 == -1:
                    true_id = 1
                else:
                    y_true3 = mech3.B0 @ data
                    is_mech4 = satisfy_user_target(args, y_true3, vec_k3)
                    if is_mech4 == -1:
                        true_id = 2
                    else:
                        true_id = 3

            pop_list[true_id].append(population)

        label = [[i] * len(pop_list[i]) for i in range(4)]
        X = np.concatenate(pop_list).reshape(-1, 1)
        y = np.concatenate(label)
        clf = LogisticRegression(multi_class='multinomial', max_iter=1000)
        clf.fit(X, y)
        acc = clf.score(X, y)

        print("The acc are: ", acc*100)
        # run_experiment_adaptive(args, dataset, mech_list, clf)
        toc = time.time()
        print("time is: ", toc-tic)
        print("###################################################################")


