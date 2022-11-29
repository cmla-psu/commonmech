from utils import *
import argparse
import numpy as np
# import scipy.stats as st
# import time


def config():
    """Return configuration parameters."""
    parser = argparse.ArgumentParser(description='Common Mechanism')

    parser.add_argument('--blocks', default=1000, help='Number of blocks')
    parser.add_argument('--repeat', default=1, help='Repeat for each block')
    parser.add_argument('--use_nnls', default=True, help='Use non-negative least square or not')
    parser.add_argument('--scale', default=1, help='Domain size range scale')
    parser.add_argument('--name', default='None', help='Figure name')
    parser.add_argument('--use_math', default=True, help='Use math formula or not to get the common mechanism')
    parser.add_argument('--gamma', default=None, help='Figure name')
    parser.add_argument('--time1', default=3, help='Time limit for mech1.')
    parser.add_argument('--time2', default=3, help='Time limit for mech2.')
    parser.add_argument('--lambd', default=0.0001, help='Regularization.')
    parser.add_argument('--sigma', default=None, help='Noise Level.')
    parser.add_argument('--plb_com', default=None, help='plb of the common mech.')
    parser.add_argument('--sigma_noise', default=None, help='Noise Level of the mech2.')
    parser.add_argument('--norm', default=1, help='l1 norm or l2 norm.')
    parser.add_argument('--ratio', default=None, help='The ratio of privacy cost used in the selection step.')
    parser.add_argument('--sensitivity', default=None, help='Sensitivity.')
    parser.add_argument('--sample', default=10, help='Sample size.')
    parser.add_argument('--axis', default=(500, 10), help='Axis scale, range(0, 500, 10).')
    parser.add_argument('--param_x', default=None, help='Ratio of cells that have large signal.')
    parser.add_argument('--param_y', default=None, help='Threshold of signal-to-noise ratio.')
    parser.add_argument('--k_list', default=None, help='list of k values.')
    return parser.parse_args()


class CM:
    def __init__(self, args, B1, S1, B2, S2):
        self.args = args
        self.data = None
        self.y_com = None
        self.x_est = None
        self.SM = [0, 0]
        self.PM = [0, 0]
        self.d1 = None
        self.d2 = None

        self.B01, self.S01 = B1, S1
        self.B02, self.S02 = B2, S2

        self.B1, self.S1 = transform_to_standard(B1, S1)
        self.B2, self.S2 = transform_to_standard(B2, S2)
        self.B_com, self.S_com, self.Bc, self.Sc = common_mechanism2(
            self.B1, self.B2, self.args.use_math, B1)
        # print(np.shape(self.B_com))

        self.diag_var = np.diag(self.Sc)
        self.plb_com = np.max(np.diag(self.B_com.T @ self.B_com)) / 2
        # print("plb_com: ", self.plb_com)

        self.A11, self.A12, self.B1_res, self.S1_res = residual_mechanism(self.B_com, self.B1)
        self.A21, self.A22, self.B2_res, self.S2_res = residual_mechanism(self.B_com, self.B2)

    def input_data(self, data):
        self.data = data

    def get_SM(self):
        """Return select measure."""
        y_com = self.B_com @ self.data + noise(self.S_com)
        yc = stand_to_origin(self.B_com, y_com, self.Bc, self.Sc)

        # yc = self.Bc @ self.data + noise(self.Sc)
        choice = satisfy_user_target(self.args, yc)
        self.y_com = y_com
        return choice

    def get_PM(self, choice):
        """Return perform measure."""
        if choice == -1:
            size = np.shape(self.B1)[0]
            y_res = self.B1_res @ self.data + noise(self.S1_res)
            S_add = np.eye(size) - (self.A11 @ self.A11.T + self.A12 @ self.A12.T)
            y_rebuild = self.A11 @ self.y_com + self.A12 @ y_res + noise(S_add)
            y = stand_to_origin(self.B1, y_rebuild, self.B01, self.S01)
        elif choice == 1:
            size = np.shape(self.B2)[0]
            y_res = self.B2_res @ self.data + noise(self.S2_res)
            S_add = np.eye(size) - (self.A21 @ self.A21.T + self.A22 @ self.A22.T)
            y_rebuild = self.A21 @ self.y_com + self.A22 @ y_res + noise(S_add)
            y = stand_to_origin(self.B2, y_rebuild, self.B02, self.S02)
        return y


def run_experiment(args, dataset, B1, S1, B2, S2):
    """Run experiment."""
    population = []
    N = np.shape(dataset)[0]
    com_mech = CM(args, B1, S1, B2, S2)

    plb = privacy_loss_budget(B1, S1)
    plb_saved = privacy_loss_budget(B1, np.diag(com_mech.diag_var))
    plb_ratio = plb_saved / plb
    plb_com = com_mech.plb_com

    total_count = 0
    com_count = 0
    base_count = 0
    algo1_count = 0
    algo2_count = 0

    for i in range(N):
        data = dataset[i, :]
        population.append(np.sum(data))
        com_mech.input_data(data)

        for j in range(args.repeat):
            choice = com_mech.get_SM()
            y_true1 = B1 @ data
            y_select = y_true1 + noise(np.diag(com_mech.diag_var))
            base_label = satisfy_user_target(args, y_select)
            algo1_label = -1
            algo2_label = 1

            true_label = satisfy_user_target(args, y_true1)

            if true_label == choice:
                com_count += 1
            if true_label == base_label:
                base_count += 1
            if true_label == algo1_label:
                algo1_count += 1
            if true_label == algo2_label:
                algo2_count += 1

            total_count += 1

        if i % 2000 == 0:
            print("-------------------------  ", i, "  ---------------------------------")

    print("algo1 correct: ", algo1_count, "ratio: ", algo1_count / total_count * 100)
    print("algo2 correct: ", algo2_count, "ratio: ", algo2_count / total_count * 100)
    print("base correct: ", base_count, "ratio: ", base_count / total_count * 100)
    print("com correct: ", com_count, "ratio: ", com_count / total_count * 100)

    print("PLB: ", plb, "PLB com: ", plb_com,  "Used PLB Ratio: ", plb_com / plb * 100)
    print("PLB saved: ", plb_saved, "Saved PLB Ratio: ", plb_ratio * 100)

    print("####################################################################")
    return com_mech

