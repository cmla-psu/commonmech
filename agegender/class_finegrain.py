from utils import *
import argparse
import numpy as np
from scipy.linalg import block_diag
import copy
from collections import deque


def config():
    """Return configuration parameters."""
    parser = argparse.ArgumentParser(description='Common Mechanism')

    parser.add_argument('--blocks', default=1000, help='Number of blocks')
    parser.add_argument('--repeat', default=5, help='Repeat for each block')
    # parser.add_argument('--PM', default=True, help='Perform measure')
    parser.add_argument('--use_nnls', default=True, help='Use non-negative least square or not')
    parser.add_argument('--scale', default=1, help='Domain size range scale')
    parser.add_argument('--name', default='None', help='Figure name')
    parser.add_argument('--use_math', default=True, help='Use math formula or not to get the common mechanism')
    parser.add_argument('--time1', default=3, help='Time limit for mech1.')
    parser.add_argument('--time2', default=3, help='Time limit for mech2.')
    parser.add_argument('--lambd', default=0.0001, help='Regularization.')
    parser.add_argument('--sigma', default=2, help='Noise Level.')
    parser.add_argument('--norm', default=1, help='l1 norm or l2 norm.')
    parser.add_argument('--ratio', default=1 / 4, help='The ratio of privacy cost used in the selection step.')
    parser.add_argument('--sensitivity', default=3, help='Sensitivity.')
    parser.add_argument('--sample_size', default=10, help='How many samples needed to select a mechanism.')
    parser.add_argument('--thresholds', default=[9, 20, 93], help='Hand tuned thresholds.')
    parser.add_argument('--gamma', default=1 / 16, help='The ratio of privacy cost used estimating total sum.')
    parser.add_argument('--sigma_com', default=None, help='Noise Level of the common mech.')
    parser.add_argument('--sigma_noise', default=None, help='Noise Level of the mech2.')
    parser.add_argument('--param_x', default=None, help='Ratio of cells that have large signal.')
    parser.add_argument('--param_y', default=None, help='Threshold of signal-to-noise ratio.')
    return parser.parse_args()


class Mechanism:
    def __init__(self, args, B, S):
        self.args = args
        self.B0, self.S0 = B, S
        self.B, self.S = transform_to_standard(B, S)

        self.data = None
        self.B_com = None
        self.S_com = None

        self.S_res = None
        self.B_res = None
        self.A1 = None
        self.A2 = None

    def get_standard_query(self):
        return self.B

    def get_standard_cov(self):
        return self.S

    def get_origin_query(self):
        return self.B0

    def get_origin_cov(self):
        return self.S0

    def input_data(self, data):
        self.data = data

    def input_common_mech(self, B_com, S_com):
        self.B_com = B_com
        self.S_com = S_com
        self.A1, self.A2, \
        self.B_res, self.S_res = residual_mechanism(self.B_com, self.B)

    def get_performance_measure(self, y_com):
        size = np.shape(self.B)[0]
        y_res = self.B_res @ self.data + noise(self.S_res)
        S_add = np.eye(size) - (self.A1 @ self.A1.T + self.A2 @ self.A2.T)
        y_rebuild = self.A1 @ y_com + self.A2 @ y_res + noise(S_add)
        y = stand_to_origin(self.B, y_rebuild, self.B0, self.S0)
        pm = perform_measure(self.B0, self.data, y, self.args)
        return pm

    def get_measure(self):
        y = self.B0 @ self.data + noise(self.S0)
        pm = perform_measure(self.B0, self.data, y, self.args)
        return pm

    def get_pm(self, data, y):
        pm = perform_measure(self.B0, data, y, self.args)
        return pm

    def get_noisy_answer(self, data, B_com, S_com, y_com):
        self.input_common_mech(B_com, S_com)
        size = np.shape(self.B)[0]
        if np.shape(self.B_res)[0] == 0:
            y_rebuild = y_com
        else:
            coeff = 20
            y_res = self.B_res @ data / coeff + noise(self.S_res / coeff**2)
            y_res = y_res * coeff
            y_res = self.B_res @ data + noise(self.S_res)
            # y_res = self.B_res @ data
            S_add = np.eye(size) - (self.A1 @ self.A1.T + self.A2 @ self.A2.T)
            if np.max(np.abs(S_add)) < 1e-8:
                # S_add = 0
                y_rebuild = self.A1 @ y_com + self.A2 @ y_res
            else:
                # print("noise added")
                y_rebuild = self.A1 @ y_com + self.A2 @ y_res + noise(S_add)
            # y_res = self.B_res @ data
            # y_rebuild = self.A1 @ y_com + self.A2 @ y_res
        y = stand_to_origin(self.B, y_rebuild, self.B0, self.S0)
        # y = y_rebuild
        return y


class CommonMechanism:
    def __init__(self, args, mech_list, B=None):
        self.data = None
        self.args = args
        # [M1, M2, ..., Mk]
        self.mech_list = mech_list
        self.B_com, self.S_com, self.Bs, self.Ss = common_mechanism(mech_list, B)

    def input_data(self, data):
        self.data = data

    def get_query_cov(self):
        return self.B_com, self.S_com

    def get_performance_measure(self):
        y_com = self.B_com @ self.data + noise(self.S_com)
        x_est = least_square(self.B_com, y_com, self.args)

        idx = select_mech(x_est, self.B_com, self.mech_list, self.args)
        pm_list = []

        for mech in self.mech_list:
            mech.input_data(self.data)
            mech.input_common_mech(self.B_com, self.S_com)
            pm = mech.get_performance_measure(y_com)
            pm_list.append(pm)
        return pm_list, idx

    def _noisy_answer(self):
        y_com = self.B_com @ self.data + noise(self.S_com)
        return y_com

    def get_noisy_answer(self, data):
        self.input_data(data)
        y_com = self._noisy_answer()
        return y_com


def run_experiment_adaptive(args, dataset, mech_list, cm_list):
    """Run experiment."""
    N = np.shape(dataset)[0]

    mech1, mech2, mech3, mech4 = mech_list
    com_mech, com_mech234, com_mech34 = cm_list
    CM234 = Mechanism(args, com_mech234.B_com, com_mech234.S_com)
    CM34 = Mechanism(args, com_mech34.B_com, com_mech34.S_com)

    vec_k1 = np.array([8])
    vec_k2 = np.array([2, 3, 2, 2, 2, 3, 2, 2])
    vec_k3 = np.array([1, 3, 4, 2, 2, 2, 3, 3, 3, 1, 3, 4, 2, 2, 2, 3, 3, 3])

    total_count = 0
    correct_count = 0
    algo_count = np.zeros(4)
    for i in range(N):
        data = dataset[i, :]
        com_mech.input_data(data)

        for j in range(args.repeat):
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

            y_cm = com_mech.B_com @ data + noise(com_mech.S_com)
            y_cm234 = CM234.get_noisy_answer(data, com_mech.B_com, com_mech.S_com, y_cm)
            y_cm34 = CM34.get_noisy_answer(data, com_mech234.B_com, com_mech234.S_com, y_cm234)

            yc1 = stand_to_origin(com_mech.B_com, y_cm, com_mech.Bs, com_mech.Ss)
            yc2 = stand_to_origin(com_mech234.B_com, y_cm234, com_mech234.Bs, com_mech234.Ss)
            yc3 = stand_to_origin(com_mech34.B_com, y_cm34, com_mech34.Bs, com_mech34.Ss)
            is_mech_com2 = satisfy_user_target(args, yc1, vec_k1)
            if is_mech_com2 == -1:
                com_id = 0
            else:
                is_mech_com3 = satisfy_user_target(args, yc2, vec_k2)
                if is_mech_com3 == -1:
                    com_id = 1
                else:
                    is_mech_com4 = satisfy_user_target(args, yc3, vec_k3)
                    if is_mech_com4 == -1:
                        com_id = 2
                    else:
                        com_id = 3

            if true_id == com_id:
                correct_count += 1
            algo_count[true_id] += 1
            total_count += 1

        if i % 20000 == 0:
            print(f"*************************  Round {i}  ******************************")
            print("Algo Ratio: ", algo_count / total_count * 100)
            ratio = correct_count / total_count * 100
            print("Correct count: ", correct_count, "Ratio: ", ratio)

    print("----------------------------------------------------------------------------------")
    print("Algo Ratio: ", algo_count / total_count * 100)
    ratio = correct_count / total_count * 100
    print("Correct count: ", correct_count, "Ratio: ", ratio)
    print("########################################################################################")
    return com_mech



