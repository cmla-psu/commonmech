import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import nnls


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
    return A1, A2, Bresidual, cov_mat(Bresidual)


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


def minimal_upper_bound(mat_list):
    length = len(mat_list)
    A1, A2 = mat_list
    P1 = A1 @ A1.T
    P2 = A2 @ A2.T

    vec, mat = np.linalg.eigh(P1-P2)
    idx = np.where(vec > 1e-8)[0]
    Core = P2 + (mat[:, idx] * (vec[idx])) @ mat[:, idx].T

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
