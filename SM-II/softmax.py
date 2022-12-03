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


import argparse
import numpy as np


def configuration():
    """
    Return configuration parameters.

    Returns
    -------
    args : parameters.

    """
    parser = argparse.ArgumentParser(description='Matrix Query')

    parser.add_argument('--maxiter', default=1000000, help='total iteration')
    parser.add_argument('--maxitercg', default=5,
                        help='maximum iteration for conjugate gradient method')
    parser.add_argument('--maxiterls', default=50,
                        help='maximum iteration for finding a step size')
    parser.add_argument('--theta', default=1e-10,
                        help='determine when to stop conjugate gradient method'
                        ' smaller theta makes the step direction more accurate'
                        ' but it takes more time')
    parser.add_argument('--beta', default=0.5, help='step size decrement')
    parser.add_argument('--sigma', default=1e-2,
                        help='determine how much decrement in '
                        'objective function is sufficient')
    parser.add_argument('--NTTOL', default=1e-3,
                        help='determine when to update self.param_t, '
                        'smaller NTTOL makes the result more accurate '
                        'and the convergence slower')
    parser.add_argument('--TOL', default=1e-3,
                        help='determine when to stop the whole program')
    parser.add_argument('--MU', default=2, help='increment for '
                        'barrier approximation parameter self.param_t')
    parser.add_argument('--init_mat', default='id_index',
                        help='id_index method is sufficient as an initialization.')
    parser.add_argument('--basis', default='work',
                        help='id: id mat; work: work mat')
    return parser.parse_args()


def workload(n, dtype=np.float32):
    """
    Create the upper triangular matrix with elements of 1.

    Return as the basis matrix.
    """
    mat_w = np.triu(np.ones([n, n]))
    return mat_w


def is_pos_def(A):
    """
    Check positive definiteness.

    Return true if psd.
    """
    # first check symmetry
    if np.allclose(A, A.T, 1e-5, 1e-8):
        # whether cholesky decomposition is successful
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False


def hb_strategy_matrix(n, k):
    """
    Strategy matrix of HB method.

    here requires n is a square number so that n = k*k
    special case : if n=2, return a given matrix

    Parameters
    ----------
    n : the number of nodes
    k : the branching factor
    """
    if n == 2:
        Tree = np.array([[1, 1], [1, 0], [0, 1]])
    else:
        m = 1 + k + n
        Tree = np.zeros([m, n])
        Tree[0, :] = 1
        for i in range(k):
            Tree[i+1, k*i: k*i+k] = 1
        Tree[k+1:, :] = np.eye(n)
    return Tree


def hb_variance(W, A, s):
    """
    Calculate the maximum variance for a single query in workload W.

    using HB method with privacy cost at most s.

    Parameters
    ----------
    A is the hb tree structure
    s is the privacy cost
    """
    m, n = A.shape
    mI = np.eye(m)
    pA = np.linalg.pinv(A)
    sigma = np.sqrt(3/s)
    Var = W @ pA
    BXB = Var @ (sigma**2*mI) @ Var.T
    a = np.diag(BXB)
    return a


def gm_variance(W, A, s):
    """
    Calculate the maximum variance for a single query in workload W.

    Parameters
    ----------
    s is the privacy cost
    """
    m, n = A.shape
    mI = np.eye(m)
    # pseudoinverse of matrix A
    pA = np.linalg.inv(A)
    sigma = np.sqrt(1/s)
    Var = W @ pA
    BXB = Var @ (sigma**2*mI) @ Var.T
    a = np.diag(BXB)
    return a


class matrix_query:
    """Class for matrix query optimization."""

    def __init__(self, args, mat_basis=None, mat_index=None, var_bound=None):
        """
        Get Inputs.

        Parameters
        ----------
        args : TYPE
            Configuration parameters.
        mat_basis : np.ndarray
            Basis matrix. The default is None.
        mat_index : np.ndarray
            Index matrix. The default is None.
        var_bound : np.ndarray
            Variance bound. The default is None.

        Returns
        -------
        None.

        """
        self.size_m, self.size_n = mat_index.shape
        self.mat_basis = mat_basis
        self.mat_index = mat_index
        self.mat_work = self.mat_index @ self.mat_basis
        self.var_bound = var_bound
        self.args = args
        self.initialization()

    def initialization(self):
        """
        Initialize covariance matrix and privacy cost.

        Returns
        -------
        None.

        """
        self.mat_id = np.eye(self.size_n)
        self.param_t = 1
        self.param_k = 1
        if self.args.init_mat == 'id_index':
            diag = np.diag(self.mat_index @ self.mat_index.T)
            sigma = np.min(self.var_bound/diag)
            self.cov = self.mat_id*self.size_n*sigma
        self.invcov = np.linalg.solve(self.cov, self.mat_id)
        self.f_var = self.func_var()
        self.f_pcost = self.func_pcost()

    def func_var(self):
        """
        Inequality constraint function for variance.

        Parameters
        ----------
        self.var_bound : variance bound
        self.mat_index : the index matrix
        self.cov : the co-variance matrix
        """
        # d = np.diag(self.mat_index @ self.cov @ self.mat_index.T)
        vec_d = ((self.mat_index @ self.cov) * self.mat_index).sum(axis=1)
        # vec_d = np.diag(self.cov)
        return vec_d / self.var_bound

    def func_pcost(self):
        """
        Inequality constraint function for privacy cost.

        Parameters
        ----------
        B : the basis matrix
        self.invcov : the inverse of the co-variance matrix X
        """
        # d = np.diag(self.mat_basis.T @ self.invcov @ self.mat_basis)
        vec_d = ((self.mat_basis.T @ self.invcov) * self.mat_basis.T).sum(
            axis=1)
        # vec_d = np.triu(self.invcov.cumsum(axis=1)).sum(axis=0)
        return vec_d

    def obj(self):
        """
        Objective function.

        Parameters
        ----------
        X : co-variance matrix
        self.param_t : privacy cost approximation parameter
        self.param_k : variance approximation parameter
        c : variance bound, self.mat_index: index matrix, B: basis matrix
        """
        const_t = self.param_t*np.max(self.f_pcost)
        const_k = self.param_k*np.max(self.f_var)
        log_sum_t = np.log(np.sum(np.exp(self.param_t*self.f_pcost - const_t)))
        log_sum_k = np.log(np.sum(np.exp(self.param_k*self.f_var - const_k)))
        f_obj = const_t + log_sum_t + const_k + log_sum_k
        return f_obj

    def derivative(self):
        """Calculate derivatives."""
        const_k = self.param_k * np.max(self.f_var)
        exp_k = np.exp(self.param_k*self.f_var - const_k)
        self.g_var = (exp_k/self.var_bound*self.mat_index.T) @ self.mat_index

        const_t = np.max(self.param_t * self.f_pcost)
        self.mat_bix = self.invcov.T @ self.mat_basis
        exp_t = np.exp(self.param_t*self.f_pcost-const_t)
        self.g_pcost = -(exp_t*self.mat_bix) @ self.mat_bix.T

        coef_k = self.param_k/np.sum(exp_k)
        coef_t = self.param_t/np.sum(exp_t)
        grad_k = self.g_var * coef_k
        grad_t = self.g_pcost * coef_t

        grad = grad_k + grad_t
        vec_grad = np.reshape(grad, [-1], 'F')
        return vec_grad

    def hess_times_p(self, vec_p):
        """Calculate H*p."""
        mat_p = np.reshape(vec_p, [self.size_n, self.size_n], 'F')

        # Calculate kron(Vmat, Vmat)*p
        const_k = self.param_k * np.max(self.f_var)
        exp_k = np.exp(self.param_k*self.f_var - const_k)
        h_var_1 = self.g_var @ mat_p @ self.g_var
        # Calculate kron(g_var, g_var)*p
        f_var_p = ((self.mat_index @ mat_p)*self.mat_index).sum(axis=1)
        coef = exp_k / self.var_bound**2 * f_var_p
        h_var_2 = (coef * self.mat_index.T) @ self.mat_index

        # Calculate kron(Bmat, Bmat)*p
        const_t = np.max(self.param_t * self.f_pcost)
        exp_t = np.exp(self.param_t*self.f_pcost-const_t)
        g_pcost_p = self.g_pcost @ mat_p
        h_pcost_1 = g_pcost_p @ self.g_pcost
        # Calculate kron(g_pcost, p_cost)*p
        f_pcost_p = ((self.mat_bix.T@mat_p)*self.mat_bix.T).sum(axis=1)
        h_pcost_21 = (exp_t * f_pcost_p * self.mat_bix) @ self.mat_bix.T
        mat_mul = -g_pcost_p @ self.invcov
        h_pcost_22 = mat_mul + mat_mul.T
        h_pcost_2 = h_pcost_21 + h_pcost_22

        coef_k1 = -(self.param_k/np.sum(exp_k))**2
        coef_k2 = self.param_k**2/np.sum(exp_k)
        coef_t1 = -(self.param_t/np.sum(exp_t))**2
        coef_t2 = self.param_t**2/np.sum(exp_t)
        hess = h_var_1*coef_k1 + h_pcost_1*coef_t1 + \
            h_pcost_2*coef_t2 + h_var_2*coef_k2
        vec_d = np.reshape(hess, [-1], 'F')
        return vec_d

    def newton_direction(self):
        """Calculate Newton's Direction."""
        vec_v = np.zeros(self.size_n**2)
        vec_r = -self.gradient
        vec_p = vec_r
        rr_old = vec_r.dot(vec_r)
        for i in range(self.args.maxitercg):
            hess_p = self.hess_times_p(vec_p)
            p_dot_hp = vec_p.dot(hess_p)
            if np.linalg.norm(p_dot_hp) < 1e-20:
                print("divided by 0.")
                break
            param_a = rr_old/p_dot_hp
            vec_v = vec_v + param_a * vec_p
            vec_r = vec_r - param_a * hess_p
            rr_new = vec_r.dot(vec_r)
            if rr_new < self.args.theta:
                break
            param_b = rr_new/rr_old
            vec_p = vec_r + param_b * vec_p
            rr_old = rr_new
        return vec_v, i

    def step_size(self):
        """Find proper step size."""
        mat_old = self.cov
        f_curr = self.obj()
        f_last = f_curr
        step = np.reshape(self.step, [self.size_n, self.size_n], 'F')
        for k in range(self.args.maxiterls):
            alpha = self.args.beta**(k)
            self.cov = mat_old + alpha*step
            if not is_pos_def(self.cov):
                continue
            self.invcov = np.linalg.solve(self.cov, self.mat_id)
            self.f_var = self.func_var()
            self.f_pcost = self.func_pcost()
            f_curr = self.obj()
            if f_curr < f_last + alpha*self.args.sigma*self.delta:
                break
        return f_curr, k

    def optimize(self):
        """Optimization."""
        for iters in range(self.args.maxiter):
            self.gradient = self.derivative()
            self.step, i = self.newton_direction()
            self.delta = self.step.dot(self.gradient)
            if self.delta >= 0:
                self.step = -self.gradient
                self.delta = self.step.dot(self.gradient)
            if np.abs(self.delta) < self.args.NTTOL:
                gap = (self.size_m+self.size_n)/self.param_t
                if gap < self.args.TOL:
                    break
                self.param_t = self.args.MU*self.param_t
                self.param_k = self.args.MU*self.param_t
                print('update t: {0}'.format(self.param_t))
            else:
                fcurr, k = self.step_size()
                print("iter:{0}, fobj:{1}, opt:{2}, cg:{3}, ls:{4}".
                      format(iters, fcurr, self.delta, i, k))

        pcost = np.max(self.f_var)*np.max(self.f_pcost)/2.0
        # pcost = 1
        # print('pcost =', pcost)
        # hb_strategy = hb_strategy_matrix(
        #     self.size_n, np.int(np.sqrt(self.size_n)))
        # self.gm_var = gm_variance(self.mat_work, self.mat_id, pcost)
        # self.hb_var = hb_variance(self.mat_work, hb_strategy, pcost)
        #
        # self.pcost = pcost
        # self.gm = self.gm_var
        # self.hm = self.hb_var


def func_var(cov, mat_index):
    """
    Inequality constraint function for variance.

    Parameters
    ----------
    self.var_bound : variance bound
    self.mat_index : the index matrix
    self.cov : the co-variance matrix
    """
    # d = np.diag(self.mat_index @ self.cov @ self.mat_index.T)
    vec_d = ((mat_index @ cov) * mat_index).sum(axis=1)
    # vec_d = np.diag(self.cov)
    return vec_d
