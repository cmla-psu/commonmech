import numpy as np
# import matplotlib.pyplot as plt
# from scipy import stats
from scipy.optimize import nnls, linprog
# import scipy as sp
# from scipy.io import savemat
# from matplotlib.pyplot import MultipleLocator
# import gurobipy as gp
import cvxpy as cp
# from gurobipy import GRB
# import datetime
# plt.switch_backend('agg')



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
    for i, y in enumerate(yc):
        thres = y / (np.sqrt(args.k_list[i]) * args.sigma_noise)
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


def mech_err(y, y_true):
    """Return mean err."""
    err = np.mean(np.abs(y-y_true))
    return err


def get_sigma(B, plb):
    """Calculate sigma."""
    plb_B = privacy_loss_budget(B)
    sigma = np.sqrt(plb_B / plb)
    return sigma


def test_err(size, sigma):
    """Test err with size."""
    err = noise(np.eye(size) * sigma**2)
    ms = np.mean(np.abs(err))
    return ms
