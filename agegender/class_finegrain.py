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


class NodeNotFoundException(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class Node:
    def __init__(self, key, index, children=None, com_mat=None, mat=None):
        self.key = key
        self.index = index
        self.parent = None
        self.children = children or []
        self.mat = None
        self.selected = False
        self.y_noisy = None
        self.y_com = None
        if mat is not None:
            left, right = key
            self.mat = mat[index, left:right].reshape(1, -1)
            # self.mat = np.ones(right-left+1)
            self.cov = np.eye(self.mat.shape[0])

        self.com_mat = None
        if com_mat is not None:
            left, right = key
            self.com_mat = com_mat[index, left:right].reshape(1, -1)
            self.com_cov = np.eye(self.com_mat.shape[0])

    def child_query(self):
        query = []
        for child in self.children:
            query.append(child.mat)
        return block_diag(*query)

    def child_com_query(self):
        com_query = []
        for child in self.children:
            if child.com_mat is None:
                return None
            com_query.append(child.com_mat)
        return block_diag(*com_query)

    def select_node(self, y_noisy=None):
        self.selected = True

    def input_answer(self, y_noisy, y_com):
        self.y_noisy = y_noisy
        self.y_com = y_com

    def __str__(self):
        return str(self.key)


class NArrTree:
    def __init__(self):
        self.root = None
        self.size = 0
        self.query = []
        self.answer = []

    def find_node(self, node, key):
        if node is None or node.key == key:
            return node
        for child in node.children:
            return_node = self.find_node(child, key)
            if return_node:
                return return_node
        return None

    def add(self, new_key, index, parent_key=None, com_mat=None, mat=None):
        new_node = Node(new_key, index, None, com_mat, mat)
        if parent_key is None:
            self.root = new_node
            self.size = 1
        else:
            parent_node = self.find_node(self.root, parent_key)
            if not parent_key:
                raise NodeNotFoundException('No element was found with the informed parent key.')
            parent_node.children.append(new_node)
            new_node.parent = parent_node
            self.size += 1

    def print_tree(self, node, str_aux):
        if node is None:
            return ""
        str_aux += str(node) + '\n('
        for i in range(len(node.children)):
            child = node.children[i]
            end = ',' if i < len(node.children) - 1 else ''
            str_aux = self.print_tree(child, str_aux) + end
        str_aux += ')'
        return str_aux

    def assign_answer(self, y_list, y_com_list):
        depth = len(y_list) - 1
        root = self.root
        queue = deque()
        queue.append(root)
        queue.append(None)
        level = 0
        idx = 0
        while len(queue) > 0:
            node = queue.popleft()
            if node is None:
                if len(queue) > 0:
                    # end of current level
                    level += 1
                    idx = 0
                    queue.append(None)
                else:
                    # finished
                    break
            else:
                y = y_list[level][idx]
                if level == depth:
                    y_com = None
                else:
                    y_com = y_com_list[level][idx]
                node.input_answer(y, y_com)
                idx += 1

                for child in node.children:
                    queue.append(child)


    def bfs_print(self, tables):
        root = self.root
        queue = deque()
        queue.append(root)
        queue.append(None)
        level = 0
        result = []
        idx_ls = []
        while len(queue) > 0:
            node = queue.popleft()
            if node is None:
                result.append(idx_ls)
                if len(queue) > 0:
                    # end of current level
                    level += 1
                    idx_ls = []
                    queue.append(None)
                    # print("##########################################################")
                else:
                    # finished
                    break
            else:
                if node.selected:
                    table = tables[level]
                    idx = table[node.key]
                    idx_ls.append(idx)
                    # print(idx, node.key, "y: ", node.y_noisy)
                for child in node.children:
                    queue.append(child)
        return result

    def _query_answer(self, node):
        if node is None:
            return
        if node.selected:
            mat = node.mat.flatten()
            y = node.y_noisy.flatten()[0]
            self.query.append(mat)
            self.answer.append(y)
            return
        else:
            for child in node.children:
                self._query_answer(child)
            return

    def get_query_answer(self):
        self._query_answer(self.root)
        return self.query, self.answer

    def is_empty(self):
        return self.size == 0

    def lens(self):
        return self.size

    def __str__(self):
        return self.print_tree(self.root, "")


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


def dfs(args, node, data):
    """Run a dfs search on the NArrTree."""
    # nonlocal query
    if node is None:
        return
    if len(node.children) == 0:
        node.select_node()
        return
    else:
        left, right = node.key
        sub_query = node.child_query()
        sub_com_query = node.child_com_query()
        sigma_2 = 1/np.max(sub_com_query)
        size = sub_query.shape[0]
        B_mat = node.mat
        sigma_1 = 1 / np.max(B_mat)
        B_com = node.com_mat
        sigma_com = 1 / np.max(B_com)
        y_com = np.array([node.y_com])
        x_est = least_square(B_com, y_com, args)
        assert len(y_com) == 1, "Not sum query"
        # print(sigma_com*sigma_sub_com)

        length = right - left
        sub_length = length // size
        population = np.sum(x_est)
        # sig = np.sqrt(sigma_com**2 + sigma_1**2)
        # sm1 = 2 * (size-1)/size * (population - 0.25*sig)
        sig = sigma_1
        sm1 = 2 * (size-1)/size * population
        sm2 = np.sqrt(2.0 / np.pi) * size * sigma_com / 2

        # d = linear_program(B_com, x_est, args)
        # y1 = B_mat @ d + noise(cov_mat(B_mat))
        # sm1 = perform_measure(B_mat, d, y1, args)
        #
        # B_compare = sub_query
        # y2 = B_compare @ d + noise(cov_mat(B_compare))
        # sm2 = perform_measure(B_compare, d, y2, args)

        # print(length, sub_length, size)
        # print(population, sigma_1, sigma_2)
        # sm1 = (length-1)/length * 2 * (population + sigma_1)
        # sm2 = (sub_length-1)/sub_length * (2 * population) + (size-1)/2 * sigma_2
        # print("sm1: ", sm1, "  sm2: ", sm2, "  true sum: ", np.sum(data))
        # print("####################################################################")

        if sm1 < sm2:
            node.select_node()
            return
        else:
            for i, child in enumerate(node.children):
                dfs(args, child, data)


def run_experiment_adaptive(args, dataset, mech_list, cm_list):
    """Run experiment."""
    mech_size = len(mech_list)
    N = np.shape(dataset)[0]
    rand_index = np.random.choice(range(N), args.blocks, replace=True)

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



