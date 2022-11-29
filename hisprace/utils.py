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


def common_mechanism(B1, B2, is_commute=True):
    """Calculate common mechanism of (B1, I1), (B2, I2).

    Currently we assume B2 = I, S2 = sigma^2*I.
    """
    if is_commute:
        P1 = B1.T @ B1
        vec1, mat1 = np.linalg.eigh(P1)
        P2 = B2.T @ B2
        vec2, mat2 = np.linalg.eigh(P2)
        vec_lower = np.minimum(vec1, vec2)
        idx = np.where(vec_lower > 1e-5)[0]
        # choose the eigenvectors that correspond to non-zero eigenvalues
        Bcommon = (mat1[:, idx] * np.sqrt(vec_lower[idx])).T
    else:
        _, n = np.shape(B1)
        X = cp.Variable((n, n), symmetric=True)
        # The operator >> denotes matrix inequality.
        constraints = [X >> 0]
        constraints += [B1.T @ B1 - X >> 0]
        constraints += [B2.T @ B2 - X >> 0]
        prob = cp.Problem(cp.Maximize(cp.trace(X)),
                          constraints)
        prob.solve()
        vec, mat = np.linalg.eigh(X.value)
        idx = np.where(vec > 1e-5)[0]
        Bcommon = (mat[:, idx] * np.sqrt(vec[idx])).T
    size_m = np.shape(Bcommon)[0]
    Scommon = np.eye(size_m)
    return Bcommon, Scommon


def noise(S):
    """Create gaussian noise."""
    # np.random.seed(0)
    size = np.shape(S)[0]
    rv = np.random.multivariate_normal(np.zeros(size), S)
    return rv


def cov_mat(mat):
    """Return identity query of the shape of mat."""
    return np.eye(np.shape(mat)[0])


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
    size_m = np.shape(Bresidual)[0]
    return A1, A2, Bresidual, np.eye(size_m)


def plot_error(population, err1, err2, err3, err4,
               err5, mad_1, mad_2, scale, name):
    """Draw the error graph"""
    # bins = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]) * scale
    bins = np.arange(0, 300, 10) * scale
    diff = (bins[1] - bins[0])*2
    num_bins = len(bins) - 1
    mean_err1, _, _ = stats.binned_statistic(population, err1, statistic='mean', bins=bins)
    mean_err2, _, _ = stats.binned_statistic(population, err2, statistic='mean', bins=bins)
    # mean_err3, _, _ = stats.binned_statistic(population, err3, statistic='mean', bins=bins)
    mean_err4, _, _ = stats.binned_statistic(population, err4, statistic='mean', bins=bins)
    mean_err5, _, _ = stats.binned_statistic(population, err5, statistic='mean', bins=bins)
    # mean_mad1, _, _ = stats.binned_statistic(population, mad_1, statistic='mean', bins=bins)
    # mean_mad2, _, _ = stats.binned_statistic(population, mad_2, statistic='mean', bins=bins)

    # xp = np.arange(num_bins)
    xp = bins[0:num_bins]
    plt.plot(xp, mean_err1, label='Algorithm 1', color='orange')
    plt.plot(xp, mean_err2, label="Algorithm 2", color='lightskyblue')
    # plt.plot(xp, mean_err3, label='Opt Mechanism', linestyle='dotted', color='green')
    plt.plot(xp, mean_err4, label='Select Algorithm', linestyle='dotted', color='red')
    plt.plot(xp, mean_err5, label='Baseline', linestyle='dotted', color='black')
    # plt.scatter(xp, mean_mad1, label='Choose Measure 1', s=20, color='orange')
    # plt.scatter(xp, mean_mad2, label='Choose Measure 2', s=20, color='blue')

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
    fig_name = name + '-' + month + '-' + day + '-' + hour + '-' + minute + '-' + second + '.png'
    if name != 'None':
        plt.savefig(fig_name)
    plt.show()
    plt.clf()


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
    # S_sqrt = sqrt_mat(S)
    # S_inv_sqrt = np.linalg.pinv(S_sqrt)
    # B_star = S_inv_sqrt.T @ B
    size_m = np.shape(B_star)[0]
    S_star = np.eye(size_m)
    return B_star, S_star


def minimal_upper_bound(A1, A2, is_special=True):
    if is_special:
        P1 = A1 @ A1.T
        P2 = A2 @ A2.T

        vec, mat = np.linalg.eigh(P1-P2)
        idx = np.where(vec > 1e-8)[0]
        Core = P2 + (mat[:, idx] * (vec[idx])) @ mat[:, idx].T

    else:
        n, _ = np.shape(A1)
        X = cp.Variable((n, n), symmetric=True)
        # The operator >> denotes matrix inequality.
        constraints = [X >> 0]
        # constraints = []
        constraints += [A1 @ A1.T - X << 0]
        constraints += [A2 @ A2.T - X << 0]
        prob = cp.Problem(cp.Minimize(cp.trace(X)), constraints)
        prob.solve()
        Core = X.value
    return Core


def common_mat(B1, B2):
    """Find a basis in the row(B1) intersect row(B2)."""
    U = B1.T
    V = B2.T
    size = np.shape(U)[0]
    Core_U = np.linalg.pinv(U.T @ U)
    Core_V = np.linalg.pinv(V.T @ V)

    Pu = U @ Core_U @ U.T
    Pv = V @ Core_V @ V.T

    M = Pu @ Pv - np.eye(size)
    vec, mat = np.linalg.eigh(M)
    idx = np.where(np.abs(vec) < 1e-8)[0]

    res = mat[:, idx]
    return res.T


def common_mechanism2(B1, B2, is_commute=True, B_com=None):
    """Calculate common mechanism of (B1, I1), (B2, I2).

    Currently we assume B2 = I, S2 = sigma^2*I.
    """

    # in this setting, B1 = A1 @ B2;
    if B_com is None:
        Bs = common_mat(B1, B2)
        assert np.shape(Bs)[0] > 0, "B1 B2 doesn't have common matrix."
    else:
        Bs = B_com
    A1 = Bs @ np.linalg.pinv(B1)
    A2 = Bs @ np.linalg.pinv(B2)
    Ss = minimal_upper_bound(A1, A2, is_commute)
    B_stand, S_stand = transform_to_standard(Bs, Ss)
    return B_stand, S_stand, Bs, Ss


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
    Ds = np.linalg.lstsq(B, y, rcond=None)[0]
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


def select_measure(args, yc):
    """Select based on signal-to-noise ratio.

    """
    # print("yc: ", yc)
    num_cell = len(yc)
    count_left = 0
    count_right = 0
    sigma = args.sigma_com
    scale = 3
    result = 0
    var = np.sqrt(args.cell_k) * args.sigma_noise
    for y in yc:
        thres_right = (y - scale*sigma) / var
        thres_left = (y + scale*sigma) / var
        if thres_right >= args.param_y:
            count_right += 1
        if thres_left <= args.param_y:
            count_left += 1
        # print("thres: ", thres_right, thres_left)
    ratio_right = count_right / num_cell
    ratio_left = count_left / num_cell
    # print("count: ", count_right, count_left)
    # print("ratios", ratio_right, ratio_left)
    if ratio_right >= args.param_x:
        result = 1
    elif ratio_left >= 1 - args.param_x:
        result = -1
    else:
        result = 0

    return result


def satisfy_user_target(args, yc):
    """Select based on signal-to-noise ratio.

    """
    num_cell = len(yc)
    count = 0
    # print("yc: ", yc)
    for y in yc:
        thres = y / (np.sqrt(args.cell_k) * args.sigma_noise)
        if thres >= args.param_y:
            # print(thres)
            count += 1
    ratio = count / num_cell
    # print("ratio: ", ratio)
    if ratio >= args.param_x:
        # there are enough signal, choose mech2
        choice = 1
    else:
        choice = -1

    return choice


def select_measure0(xc, Bc, B1, B10, B2, B20, args, S10, S20):
    """Select based on the performance measure.

    """
    size_n = np.size(xc)
    xc_mean = np.mean(xc)
    ms1 = np.linalg.norm(xc-xc_mean, ord=1)
    ms2 = size_n * args.sigma * np.sqrt(2/np.pi)

    return ms1, ms2


def select_measure1(xc, Bc, B1, B10, B2, B20, args, S10, S20):
    """Select based on the performance measure.

    """
    size_1 = np.shape(B10)[0]
    size_n = np.size(xc)
    B1_pinv = np.linalg.pinv(B10)
    x1 = B1_pinv @ B10 @ xc
    ms1 = np.linalg.norm(xc-x1, ord=1)
    ms2 = (size_1-size_n) * args.sigma * np.sqrt(2/np.pi)

    return ms1, ms2


def select_measure2(xc, Bc, B1, B10, B2, B20, args, S10, S20):
    """Calculate the maximum possible variance.

    """

    size_c, _ = np.shape(Bc)
    size_m1, size_n = np.shape(B1)
    size_m2, _ = np.shape(B2)
    count1 = 0
    count2 = 0
    diff = 0
    Bc_pinv = np.linalg.pinv(Bc)
    x0 = Bc_pinv @ Bc @ xc
    mat_res = np.eye(size_n) - Bc_pinv @ Bc
    for i in range(args.sample):
        vec = np.random.randn(size_n)
        d = x0 + mat_res @ vec
        d = xc
        # rv1 = noise(np.eye(size_m1))
        # y1 = B1 @ d + rv1
        # y01 = stand_to_origin(B1, y1, B10, S10)
        # SM1 = perform_measure(B10, d, y01, args)
        #
        # rv2 = noise(np.eye(size_m2))
        # y2 = B2 @ d + rv2
        # y02 = stand_to_origin(B2, y2, B20, S20)
        # SM2 = perform_measure(B20, d, y02, args)

        rv1 = noise(S10)
        y1 = B10 @ d + rv1
        SM1 = perform_measure(B10, d, y1, args)

        rv2 = noise(S20)
        y2 = B2 @ d + rv2
        SM2 = perform_measure(B20, d, y2, args)

        diff += (SM1 - SM2)
        # if SM1 <= SM2:
        #     count1 += 1
        # else:
        #     count2 += 1

    # mdic = {"H": Nu, "x": xc}
    # savemat("sample.mat", mdic)

    return diff, 0


def group_by_best(args, population, err1, err2, err_base, err_com, name='None'):
    """Draw the error graph"""
    # bins = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]) * scale
    ax_right, ax_diff = args.axis
    bins = np.arange(0, ax_right, ax_diff)
    # bins = np.arange(0, 300, 10)
    diff = (bins[1] - bins[0])*2
    num_bins = len(bins) - 1
    err_best = np.minimum(err1, err2)
    mean_err_ls = []

    mean_err_best, _, _ = stats.binned_statistic(err_best, err_best, statistic='mean', bins=bins)
    mean_err_base, _, _ = stats.binned_statistic(err_best, err_base, statistic='mean', bins=bins)
    mean_err_com, _, _ = stats.binned_statistic(err_best, err_com, statistic='mean', bins=bins)

    xp = bins[0:num_bins]
    # colors = ['orange', 'lightskyblue', 'darkviolet', 'limegreen']
    # for i, mean_err in enumerate(mean_err_ls):
    #     plt.plot(xp, mean_err, label='Algorithm'+str(i+1), color=colors[i])

    plt.plot(xp, mean_err_best, label='Best Oracle', color='orange')
    plt.plot(xp, mean_err_com, label='Select based on common mechanism', linestyle='dotted', color='red')
    plt.plot(xp, mean_err_base, label='Select based on objective function', linestyle='dotted', color='blue')

    # x_major_locator = MultipleLocator(diff)
    # ax = plt.gca()
    # ax.xaxis.set_major_locator(x_major_locator)
    # plt.xlim(-0.5*bins[1], (num_bins - 0.5)*bins[1])
    # lower = min(min(mean_err1), min(mean_err2))
    # upper = max(max(mean_err1), max(mean_err2))
    # plt.ylim(0, 10 + 1)
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


def com_to_base(args, yc, B, data):
    """Transfer com answers to base answers."""
    s = np.sqrt(1/args.ratio) * args.sigma
    sc = args.sigma_com
    if s > sc:
        # print("Baseline answer is worse than common mech answer.")
        return None
    sr = 1/np.sqrt(1/s**2 - 1/sc**2)

    lc = sr**2/(sc**2 + sr**2)
    lr = sc**2/(sc**2 + sr**2)

    yr = B @ data + noise(cov_mat(B) * sr**2)

    y = lc * yc + lr * yr
    return y


def relative_error(y_true, y_noisy):
    """Return relative error."""
    delta = 1e-8
    err = np.abs(y_noisy - y_true) / (y_true + delta)
    return np.min(err)


def privacy_loss_budget(B, S=None):
    """Return plb = rho."""
    if S is None:
        S = cov_mat(B)
    rho = np.max(np.diag(B.T @ np.linalg.pinv(S) @ B)) / 2.0
    return rho

