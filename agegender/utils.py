import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import nnls, linprog
import scipy as sp
from scipy.io import savemat
from matplotlib.pyplot import MultipleLocator
import gurobipy as gp
import cvxpy as cp
from gurobipy import GRB
import datetime
# plt.switch_backend('agg')
# from mhar import walk
# import torch
# from mhar import mhar_example


def marginal(d, k, choice):
    """
    k**d domain.

    choice: "one" -> one way marginal
            "two" -> two way marginal
    """

    def kron_i(mat, vec, i):
        """Do kronecker productor i times, i >= 0."""
        res = mat
        for times in range(i):
            res = np.kron(res, vec)
        return res

    def mat_i_j(d, mat, vec, i, j):
        """Two way margnial matrix."""
        res = 1
        for p_1 in range(i):
            res = np.kron(res, vec)
        res = np.kron(res, mat)
        for p_2 in range(j - i - 1):
            res = np.kron(res, vec)
        res = np.kron(res, mat)
        for p_3 in range(d - j - 1):
            res = np.kron(res, vec)
        return res

    mat_id = np.eye(k)
    vec_one = np.ones((1, k))
    if choice == "one":
        work = kron_i(mat_id, vec_one, d - 1)
        work_all = work
        for i in range(d - 1):
            left = kron_i(vec_one, vec_one, i)
            tmp = np.kron(left, mat_id)
            work = kron_i(tmp, vec_one, d - 2 - i)
            work_all = np.concatenate((work_all, work))

    if choice == "two":
        size_m = k * k * d * (d - 1) // 2
        size_n = k ** d
        work_all = np.zeros([size_m, size_n])

        block = k * k
        s = 0
        for i in range(d - 1):
            for j in range(i + 1, d):
                work_all[block * s: block * s + block, :] = mat_i_j(
                    d, mat_id, vec_one, i, j)
                s = s + 1
    return work_all


def cov_mat(mat):
    """Return identity query of the shape of mat."""
    return np.eye(np.shape(mat)[0])


# def common_mechanism(B1, B2, is_commute=True):
#     """Calculate common mechanism of (B1, I1), (B2, I2).
#
#     Currently we assume B2 = I, S2 = sigma^2*I.
#     """
#     if is_commute:
#         P1 = B1.T @ B1
#         vec1, mat1 = np.linalg.eigh(P1)
#         P2 = B2.T @ B2
#         vec2, mat2 = np.linalg.eigh(P2)
#         vec_lower = np.minimum(vec1, vec2)
#         idx = np.where(vec_lower > 1e-5)[0]
#         # choose the eigenvectors that correspond to non-zero eigenvalues
#         Bcommon = (mat1[:, idx] * np.sqrt(vec_lower[idx])).T
#     else:
#         _, n = np.shape(B1)
#         X = cp.Variable((n, n), symmetric=True)
#         # The operator >> denotes matrix inequality.
#         constraints = [X >> 0]
#         constraints += [B1.T @ B1 - X >> 0]
#         constraints += [B2.T @ B2 - X >> 0]
#         prob = cp.Problem(cp.Maximize(cp.trace(X)),
#                           constraints)
#         prob.solve()
#         vec, mat = np.linalg.eigh(X.value)
#         idx = np.where(vec > 1e-5)[0]
#         Bcommon = (mat[:, idx] * np.sqrt(vec[idx])).T
#     size_m = np.shape(Bcommon)[0]
#     Scommon = np.eye(size_m)
#     return Bcommon, Scommon


def noise(S):
    """Create gaussian noise."""
    # np.random.seed(0)
    size = np.shape(S)[0]
    rv = np.random.multivariate_normal(np.zeros(size), S)
    return rv


def least_square(B, y, args=None):
    if args is None:
        lamb = 0.0001
        use_nnls = True
    else:
        lamb = args.lambd
        use_nnls = args.use_nnls
    n_variables = B.shape[1]
    B2 = np.concatenate([B, np.sqrt(lamb) * np.eye(n_variables)])
    y2 = np.concatenate([y, np.zeros(n_variables)])
    if use_nnls:
        return nnls(B2, y2)[0]
    else:
        B2 = B
        y2 = y
        return np.linalg.lstsq(B2, y2, rcond=None)[0]


# def least_square(B, y, args):
#     total = 0
#     env = gp.Env(empty=True)
#     env.setParam("OutputFlag", 0)
#     env.start()
#     m = gp.Model(env=env)
#     m.Params.TIME_LIMIT = args.time1
#     # m.setParam('NonConvex', 2)
#     # m.setParam(GRB.Param.OutputFlag, 0)
#
#     size_m, size_n = np.shape(B)
#     # B_pinv = np.linalg.pinv(B)
#     N = 10**6
#     x = m.addMVar(size_n, lb=0, ub=N)
#     r = m.addVar(vtype=gp.GRB.CONTINUOUS)
#     r1 = m.addVar(vtype=gp.GRB.CONTINUOUS)
#     diff = m.addMVar(size_m, lb=-N)
#
#     lbd = args.lambd
#     m.setObjective(r + lbd*r1, sense=GRB.MINIMIZE)
#     m.addConstr(diff == B @ x - y)
#     m.addConstr(r == gp.norm(diff, 2))
#     m.addConstr(r1 == gp.norm(x, 1))
#     m.optimize()
#     try:
#         obj = m.getObjective().getValue()
#         d = x.X
#     except:
#         print("an err in obj")
#         obj = -1
#         d = np.ones(size_n)
#     return d


def perform_measure(B, data, y_noisy, args):
    """Return the performance measure of the mechanism.

    Least square measure, abs error.
    choice 0: l1 error
    choice 1: l2 error
    choice 2: l1 relative error
    choice 3: l2 relative error

    reg: Add regularization or not
    """

    x_est = least_square(B, y_noisy, args)
    err = np.linalg.norm(x_est-data, ord=args.norm)
    # y_true = B @ data
    # y_est = B @ x_est
    # size = np.shape(B)[0]
    # err = np.linalg.norm(y_true-y_est, ord=args.norm) / size
    return err


def residual_mechanism(Bcommon, B):
    """Construct residual mechanism M2.

    Core: B.T @ B - Bcommon.T @ Bcommon, precomputed to speed up.
    B_pinv: Pseudo-inverse of B, precomputed to speed up.
    -------------------------
    Given the standard form common mechanism M1 = (B1, I1)
    and original mechanism M = (B, I),
    calculate residual mechanism M2 = (B2, I2).
    """
    Core_common = Bcommon.T @ Bcommon
    Core = B.T @ B - Core_common
    B_pinv = np.linalg.pinv(B)

    vec, mat = np.linalg.eigh(Core)
    idx = np.where(vec > 1e-8)[0]
    # choose the eigenvectors that correspond to non-zero eigenvalues
    Bresidual = (mat[:, idx] * np.sqrt(vec[idx])).T
    A1 = B_pinv.T @ Bcommon.T
    A2 = B_pinv.T @ Bresidual.T
    return A1, A2, Bresidual, cov_mat(Bresidual)


# def plot_error(population, err1, err2, err3, err4,
#                err5, mad_1, mad_2, scale, name):
#     """Draw the error graph"""
#     # bins = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]) * scale
#     bins = np.arange(0, 300, 10) * scale
#     diff = (bins[1] - bins[0])*2
#     num_bins = len(bins) - 1
#     mean_err1, _, _ = stats.binned_statistic(population, err1, statistic='mean', bins=bins)
#     mean_err2, _, _ = stats.binned_statistic(population, err2, statistic='mean', bins=bins)
#     # mean_err3, _, _ = stats.binned_statistic(population, err3, statistic='mean', bins=bins)
#     mean_err4, _, _ = stats.binned_statistic(population, err4, statistic='mean', bins=bins)
#     mean_err5, _, _ = stats.binned_statistic(population, err5, statistic='mean', bins=bins)
#     # mean_mad1, _, _ = stats.binned_statistic(population, mad_1, statistic='mean', bins=bins)
#     # mean_mad2, _, _ = stats.binned_statistic(population, mad_2, statistic='mean', bins=bins)
#
#     # xp = np.arange(num_bins)
#     xp = bins[0:num_bins]
#     plt.plot(xp, mean_err1, label='Algorithm 1', color='orange')
#     plt.plot(xp, mean_err2, label="Algorithm 2", color='lightskyblue')
#     # plt.plot(xp, mean_err3, label='Opt Mechanism', linestyle='dotted', color='green')
#     plt.plot(xp, mean_err4, label='Select Algorithm', linestyle='dotted', color='red')
#     plt.plot(xp, mean_err5, label='Baseline', linestyle='dotted', color='black')
#     # plt.scatter(xp, mean_mad1, label='Choose Measure 1', s=20, color='orange')
#     # plt.scatter(xp, mean_mad2, label='Choose Measure 2', s=20, color='blue')
#
#     x_major_locator = MultipleLocator(diff)
#     ax = plt.gca()
#     ax.xaxis.set_major_locator(x_major_locator)
#     plt.xlim(-0.5*bins[1], (num_bins - 0.5)*bins[1])
#     # lower = min(min(mean_err1), min(mean_err2))
#     # upper = max(max(mean_err1), max(mean_err2))
#     # plt.ylim(0, 10 + 1)
#     plt.xlabel("Population Size")
#     plt.ylabel("Measure")
#     plt.title("Measure for each mechanism.")
#     plt.legend()
#
#     now = datetime.datetime.now()
#     month = str(now.month)
#     day = str(now.day)
#     hour = str(now.hour)
#     minute = str(now.minute)
#     second = str(now.second)
#     fig_name = name + '-' + month + '-' + day + '-' + hour + '-' + minute + '-' + second + '.png'
#     if name != 'None':
#         plt.savefig(fig_name)
#     plt.show()
#     plt.clf()
#

def plot_data(data, x):
    """Plot the data vs estimation."""
    plt.scatter(data, color='b', label='True data')
    plt.scatter(x, color='r', label='Estimated data')
    plt.legend()
    plt.xlabel("Data entry: Age for male and female")
    plt.ylabel("Number of people")
    plt.title("Count for male and female with different age.")
    # plt.show()


def sqrt_mat(X):
    """Calculate the decomposition X = B^T B."""
    vec, mat = np.linalg.eigh(X)
    idx = np.where(vec > 1e-8)[0]
    B = (mat[:, idx] * np.sqrt(vec[idx])).T
    return B


def transform_to_standard(B, S):
    """Transform (B, S) to (B_star, I).

    B_star = S^{-1/2} B, I = eye(m).
    """

    Core = B.T @ np.linalg.pinv(S) @ B
    B_star = sqrt_mat(Core)
    S_star = cov_mat(B_star)
    return B_star, S_star


# def minimal_upper_bound(A1, A2, is_special=True):
#     if is_special:
#         P1 = A1 @ A1.T
#         P2 = A2 @ A2.T
#
#         vec, mat = np.linalg.eigh(P1-P2)
#         idx = np.where(vec > 1e-8)[0]
#         Core = P2 + (mat[:, idx] * (vec[idx])) @ mat[:, idx].T
#
#     else:
#         n, _ = np.shape(A1)
#         X = cp.Variable((n, n), symmetric=True)
#         # The operator >> denotes matrix inequality.
#         constraints = [X >> 0]
#         # constraints = []
#         constraints += [A1 @ A1.T - X << 0]
#         constraints += [A2 @ A2.T - X << 0]
#         prob = cp.Problem(cp.Minimize(cp.trace(X)), constraints)
#         prob.solve()
#         Core = X.value
#     return Core
#
#
# def common_mat(B1, B2):
#     """Find a basis in the row(B1) intersect row(B2)."""
#     U = B1.T
#     V = B2.T
#     size = np.shape(U)[0]
#     Core_U = np.linalg.pinv(U.T @ U)
#     Core_V = np.linalg.pinv(V.T @ V)
#
#     Pu = U @ Core_U @ U.T
#     Pv = V @ Core_V @ V.T
#
#     M = Pu @ Pv - np.eye(size)
#     vec, mat = np.linalg.eigh(M)
#     idx = np.where(np.abs(vec) < 1e-8)[0]
#
#     res = mat[:, idx]
#     return res.T
#
#
# def common_mechanism2(B1, B2, is_commute=True):
#     """Calculate common mechanism of (B1, I1), (B2, I2).
#
#     Currently we assume B2 = I, S2 = sigma^2*I.
#     """
#
#     # in this setting, B1 = A1 @ B2;
#     Bs = common_mat(B1, B2)
#     assert np.shape(Bs)[0] > 0, "B1 B2 doesn't have common matrix."
#     A1 = Bs @ np.linalg.pinv(B1)
#     A2 = Bs @ np.linalg.pinv(B2)
#     Ss = minimal_upper_bound(A1, A2, is_commute)
#     Bs, Ss = transform_to_standard(Bs, Ss)
#     return Bs, Ss


def sensitivity(B, By=None):
    """The sensitivity of f(D) = || D - D*||_1."""
    B_pinv = np.linalg.pinv(B)
    size = np.shape(B)[1]
    mat = B_pinv @ B - np.eye(size)
    sen = np.linalg.norm(mat, 1)
    return sen


def app_error(D, B, By=None):
    """The approximation error || D- D*||_1."""
    y = B @ D
    Ds = least_square(B, y)
    obj = np.linalg.norm((D-Ds), 1)
    return obj


def stand_to_origin(B, y, B0, S0):
    """M_stand: (B, I) to M_origin: (B0, S0)."""
    B_pinv = np.linalg.pinv(B)
    A = B0 @ B_pinv

    S_add = S0 - A @ A.T
    if np.max(np.abs(S_add)) < 1e-8:
        # S_add = 0
        y0 = A @ y
    else:
        # print("noise added")
        y0 = A @ y + noise(S_add)
    return y0


# def linear_program(Bc, xc, args):
#     """Random sample from polytope.
#
#     -H t <= x.
#     """
#     env = gp.Env(empty=True)
#     env.setParam("OutputFlag", 0)
#     env.start()
#     m = gp.Model(env=env)
#     m.Params.TIME_LIMIT = args.time1
#     # m.setParam('NonConvex', 2)
#     # m.setParam(GRB.Param.OutputFlag, 0)
#
#     size_c, size_n = np.shape(Bc)
#     vec = np.random.randn(1, size_n)
#
#     N = np.sum(xc)
#     x = m.addMVar(size_n, lb=0, ub=N)
#     r = m.addVar(vtype=gp.GRB.CONTINUOUS)
#
#     m.setObjective(vec @ x, sense=GRB.MINIMIZE)
#     m.addConstr(Bc @ x == Bc @ xc)
#     # m.addConstr(r == vec @ x)
#     m.optimize()
#     try:
#         obj = m.getObjective().getValue()
#         d = x.X
#     except:
#         print("an err in obj")
#         obj = -1
#         d = xc
#     return d
#
#
# def select_measure(xc, Bc, B1, B10, B2, B20, args):
#     """Calculate the maximum possible variance.
#
#     """
#
#     size_c, _ = np.shape(Bc)
#     size_m1, size_n = np.shape(B1)
#     size_m2, _ = np.shape(B2)
#     count1 = 0
#     count2 = 0
#
#     for i in range(10):
#         d = linear_program(Bc, xc, args)
#
#         rv1 = noise(np.eye(size_m1))
#         y1 = B1 @ d + rv1
#         y01 = stand_to_origin(B1, y1, B10)
#         SM1 = perform_measure(B10, d, y01, args)
#
#         rv2 = noise(np.eye(size_m2))
#         y2 = B2 @ d + rv2
#         y02 = stand_to_origin(B2, y2, B20)
#         SM2 = perform_measure(B20, d, y02, args)
#         if SM1 <= SM2:
#             count1 += 1
#         else:
#             count2 += 1
#
#     # mdic = {"H": Nu, "x": xc}
#     # savemat("sample.mat", mdic)
#
#     return count1, count2


def minimal_upper_bound(mat_list):
    length = len(mat_list)
    if length == 2:
        A1, A2 = mat_list
        P1 = A1 @ A1.T
        P2 = A2 @ A2.T

        vec, mat = np.linalg.eigh(P1-P2)
        idx = np.where(vec > 1e-8)[0]
        Core = P2 + (mat[:, idx] * (vec[idx])) @ mat[:, idx].T

    else:
        n, _ = np.shape(mat_list[0])
        X = cp.Variable((n, n), symmetric=True)
        # The operator >> denotes matrix inequality.
        constraints = [X >> 0]
        # constraints = []
        for A in mat_list:
            constraints += [A @ A.T - X << 0]
        prob = cp.Problem(cp.Minimize(cp.trace(X)), constraints)
        prob.solve()
        Core = X.value
    return Core


def common_mat(mech_list):
    """Find a basis in the row(B1) intersect row(B2)."""
    Pu_list = []
    for mech in mech_list:
        B = mech.get_standard_query()
        U = B.T
        Core_U = np.linalg.pinv(U.T @ U)
        Pu = U @ Core_U @ U.T
        Pu_list.append(Pu)

    size = np.shape(Pu_list[0])[0]
    M = np.eye(size)
    for Pu in Pu_list:
        M = M @ Pu
    M = M - np.eye(size)

    vec, mat = np.linalg.eig(M)
    idx = np.where(np.abs(vec) < 1e-8)[0]
    res = mat[:, idx]
    return np.real(res.T)


def common_mechanism(mech_list, mat=None):
    """Calculate common mechanism of (B1, I1), (B2, I2).

    Currently we assume B2 = I, S2 = sigma^2*I.
    """

    # in this setting, B1 = A1 @ B2;
    if mat is None:
        Bs = common_mat(mech_list)
    else:
        Bs = mat
    assert np.shape(Bs)[0] > 0, "Can't find common matrix."
    mat_list = []
    for mech in mech_list:
        B = mech.get_standard_query()
        A = Bs @ np.linalg.pinv(B)
        mat_list.append(A)

    Ss = minimal_upper_bound(mat_list)
    if mat is None:
        B_com, S_com = transform_to_standard(Bs, Ss)
    else:
        # Todo: Need to fix when Ss is not diagonal
        B_com = np.sqrt(np.linalg.inv(Ss)+1e-15) @ mat
        S_com = np.eye(np.shape(Bs)[0])
    return B_com, S_com, Bs, Ss


def linear_program(Bc, xc, args):
    """Random sample from polytope.

    -H t <= x.
    """
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    m = gp.Model(env=env)
    m.Params.TIME_LIMIT = args.time1
    # m.setParam('NonConvex', 2)
    # m.setParam(GRB.Param.OutputFlag, 0)

    size_c, size_n = np.shape(Bc)
    vec = np.random.randn(1, size_n)

    N = np.sum(xc)
    x = m.addMVar(size_n, lb=0, ub=N)
    r = m.addVar(vtype=gp.GRB.CONTINUOUS)

    m.setObjective(vec @ x, sense=GRB.MINIMIZE)
    m.addConstr(Bc @ x == Bc @ xc)
    # m.addConstr(r == vec @ x)
    m.optimize()
    try:
        obj = m.getObjective().getValue()
        d = x.X
    except:
        print("an err in obj")
        obj = -1
        d = xc
    return d


def select_mech(xc, Bc, mech_list, args):
    """Calculate the maximum possible variance.

    """
    size = len(mech_list)
    count_list = [0 for _ in range(size)]

    for i in range(args.sample_size):
        d = linear_program(Bc, xc, args)

        sm_list = [0 for _ in range(size)]
        for k, mech in enumerate(mech_list):
            B = mech.get_standard_query()
            S = mech.get_standard_cov()
            B0 = mech.get_origin_query()
            S0 = mech.get_origin_cov()

            rv = noise(S)
            y = B @ d + rv
            y0 = stand_to_origin(B, y, B0, S0)
            sm = perform_measure(B0, d, y0, args)
            sm_list[k] = sm

        idx = np.argmin(sm_list)
        count_list[idx] += 1

    best_id = np.argmax(count_list)
    return best_id


def plot_error(args, population, err_mech, err_base, err_com, name='None'):
    """Draw the error graph"""
    # bins = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]) * scale
    bins = np.arange(0, 300, 10) * args.scale
    diff = (bins[1] - bins[0])*2
    num_bins = len(bins) - 1

    mean_err_ls = []
    for err in err_mech:
        mean_err, _, _ = stats.binned_statistic(population, err, statistic='mean', bins=bins)
        mean_err_ls.append(mean_err)

    mean_err_base, _, _ = stats.binned_statistic(population, err_base, statistic='mean', bins=bins)
    mean_err_com, _, _ = stats.binned_statistic(population, err_com, statistic='mean', bins=bins)

    xp = bins[0:num_bins]
    colors = ['orange', 'lightskyblue', 'darkviolet', 'limegreen']
    for i, mean_err in enumerate(mean_err_ls):
        plt.plot(xp, mean_err, label='Algorithm'+str(i+1), color=colors[i])

    plt.plot(xp, mean_err_com, label='Fine grained Algorithm', linestyle='dotted', color='red')
    plt.plot(xp, mean_err_base, label='Baseline', linestyle='dotted', color='black')

    x_major_locator = MultipleLocator(diff)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.xlim(-0.5*bins[1], (num_bins - 0.5)*bins[1])
    # lower = min(min(mean_err1), min(mean_err2))
    # upper = max(max(mean_err1), max(mean_err2))
    # plt.ylim(0, 10 + 1)
    plt.xlabel("Population Size")
    plt.ylabel("Measure")
    plt.title("Measure for each mechanism.")
    plt.legend()

    now = datetime.datetime.now()
    month = str(now.month)
    day = str(now.day)
    hour = str(now.hour)
    minute = str(now.minute)
    second = str(now.second)
    fig_name =name + '-' + month + '-' + day + '-' + hour + '-' + minute + '-' + second + '.png'
    if name != 'None':
        plt.savefig(fig_name)
    plt.show()
    plt.clf()


def plot_hist(args, population, err_mech, err_base, err_com, name='None'):
    """Draw the error graph"""
    # bins = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]) * scale

    upper_bound = np.max(err_mech)
    upper_bound = 100
    bins = np.linspace(0, upper_bound, 15)
    colors = ['orange', 'lightskyblue', 'darkviolet', 'limegreen', 'black', 'red']
    labels = ['Algo1', 'Algo2', 'Algo3', 'Algo4', 'Base', 'Fine grained']
    plt.hist(err_mech + [err_base, err_com], bins, color=colors, label=labels)
    plt.xlabel("Performance Measure")
    plt.ylabel("Count")
    plt.title("Histogram for each mechanism.")
    plt.legend()

    now = datetime.datetime.now()
    month = str(now.month)
    day = str(now.day)
    hour = str(now.hour)
    minute = str(now.minute)
    second = str(now.second)
    fig_name =name + '-' + month + '-' + day + '-' + hour + '-' + minute + '-' + second + '.png'
    if name != 'None':
        plt.savefig(fig_name)
    plt.show()
    plt.clf()


def plot_compare(args, population, err_mech, err_base, err_com, name='None'):
    """Draw the error graph"""
    # bins = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]) * scale
    bins = np.arange(0, 300, 10) * args.scale
    diff = (bins[1] - bins[0])*2
    num_bins = len(bins) - 1

    err1 = err_mech[0]
    mean_err_ls = []
    for err in err_mech:
        mean_err, bin_edges, bin_num = stats.binned_statistic(err1, err, statistic='mean', bins=bins)
        mean_err_ls.append(mean_err)

    mean_err_base, _, _ = stats.binned_statistic(err1, err_base, statistic='mean', bins=bins)
    mean_err_com, _, _ = stats.binned_statistic(err1, err_com, statistic='mean', bins=bins)

    high = np.max(bin_num)
    low = np.min(bin_num)
    counts, _, _ = plt.hist(bin_num, np.arange(low, high+1))
    plt.clf()

    ax1 = plt.subplot()
    ax2 = ax1.twinx()
    xp = bins[0:num_bins]
    colors = ['orange', 'lightskyblue', 'darkviolet', 'limegreen']
    leg_list = []
    leg_name = []
    for i, mean_err in enumerate(mean_err_ls):
        l1, = ax1.plot(xp, mean_err, label='Algorithm'+str(i+1), color=colors[i])
        leg_list.append(l1)
        leg_name.append('Always Select Mech'+str(i+1))

    l_com, = ax1.plot(xp, mean_err_com, label='Adaptively Select Queries', linestyle='dotted', color='red')
    l_base, = ax1.plot(xp, mean_err_base, label='Select one of the Mechanism', linestyle='dotted', color='black')
    l_count = ax2.bar(xp, counts, width=2, label='Count in the Group', color='lime')
    leg_list = leg_list + [l_com, l_base, l_count]
    leg_name = leg_name + ['Adaptively Select Queries', 'Select one of the Mechanism', 'Count in the Bin']

    ax1.set_xlabel("Data group, grouped by the performance measure Algo 1.")
    ax1.set_ylabel("Averaged performance measure in each group")
    ax2.set_ylabel("Number of data blocks in each group")
    plt.title("Comparison between each algorithm.")
    plt.legend(leg_list, leg_name, loc='upper left')

    now = datetime.datetime.now()
    month = str(now.month)
    day = str(now.day)
    hour = str(now.hour)
    minute = str(now.minute)
    second = str(now.second)
    fig_name =name + '-' + month + '-' + day + '-' + hour + '-' + minute + '-' + second + '.png'
    if name != 'None':
        plt.savefig(fig_name)
    plt.show()
    plt.clf()


def group_by_best(args, population, err_mech, err_thres, err_com, err_back, name='None'):
    """Draw the error graph"""
    # bins = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]) * scale
    bins = np.arange(0, 300, 5)
    num_bins = len(bins) - 1
    err_best = np.minimum.reduce(err_mech)
    mean_err_ls = []
    for err in err_mech:
        mean_err, _, _ = stats.binned_statistic(err_best, err, statistic='mean', bins=bins)
        mean_err_ls.append(mean_err)

    mean_err_best, _, _ = stats.binned_statistic(err_best, err_best, statistic='mean', bins=bins)
    mean_err_thres, _, _ = stats.binned_statistic(err_best, err_thres, statistic='mean', bins=bins)
    mean_err_com, _, _ = stats.binned_statistic(err_best, err_com, statistic='mean', bins=bins)
    mean_err_back, _, _ = stats.binned_statistic(err_best, err_back, statistic='mean', bins=bins)

    xp = bins[0:num_bins]
    plt.plot(xp, mean_err_best, label='Best Oracle', color='orange')
    plt.plot(xp, mean_err_com, label='Fine-grained Algorithm', linestyle='dotted', color='red')
    plt.plot(xp, mean_err_thres, label='Hand chosen threshold', linestyle='dotted', color='blue')
    plt.plot(xp, mean_err_back, label='Select in reverse', linestyle='dotted', color='green')

    plt.xlabel("Data Group, group by the error of the best mech")
    plt.ylabel("Averaged performance measure on each group")
    plt.title("Comparison of each mechanism.")
    plt.legend()

    now = datetime.datetime.now()
    month = str(now.month)
    day = str(now.day)
    hour = str(now.hour)
    minute = str(now.minute)
    second = str(now.second)
    fig_name =name + '-' + month + '-' + day + '-' + hour + '-' + minute + '-' + second + '.png'
    if name != 'None':
        plt.savefig(fig_name)
    plt.show()
    plt.clf()


def get_mech_id(population, thres):
    """Return mech i given population size."""
    t1, t2, t3 = thres
    if population <= t1:
        return 0
    elif population <= t2:
        return 1
    elif population <= t3:
        return 2
    else:
        return 3


def satisfy_user_target(args, yc, vec_k):
    """Select based on signal-to-noise ratio.
    """
    num_cell = len(yc)
    count = 0
    # print("yc: ", yc)
    for y, k in zip(yc, vec_k):
        thres = y / (np.sqrt(k) * args.sigma_noise)
        if thres >= args.param_y:
            # print(y)
            count += 1
    ratio = count / num_cell
    # print("ratio: ", ratio)
    if ratio >= args.param_x:
        # there are enough signal, choose mech2
        choice = 1
    else:
        choice = -1

    return choice
