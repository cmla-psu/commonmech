from utils import *
import argparse
import numpy as np
from scipy.linalg import block_diag


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
    # parser.add_argument('--gamma', default=0.9, help='Figure name')
    parser.add_argument('--time1', default=3, help='Time limit for mech1.')
    parser.add_argument('--time2', default=3, help='Time limit for mech2.')
    parser.add_argument('--lambd', default=0.0001, help='Regularization.')
    parser.add_argument('--sigma', default=2, help='Noise Level.')
    parser.add_argument('--norm', default=1, help='l1 norm or l2 norm.')
    parser.add_argument('--ratio', default=1 / 4, help='The ratio of privacy cost used in the selection step.')
    parser.add_argument('--gamma', default=1 / 4, help='The ratio of privacy cost used estimating total sum.')
    parser.add_argument('--sensitivity', default=3, help='Sensitivity.')
    parser.add_argument('--sample_size', default=10, help='How many samples needed to select a mechanism.')
    parser.add_argument('--thresholds', default=[9, 20, 93], help='Hand tuned thresholds.')
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
        if mat is not None:
            left, right = key
            self.mat = mat[index, left:right].reshape(1, -1)
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

    def __str__(self):
        return str(self.key)


class NArrTree:
    def __init__(self):
        self.root = None
        self.size = 0

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

    def get_noisy_answer(self, data, B_com, S_com, y_com):
        self.input_common_mech(B_com, S_com)
        size = np.shape(self.B)[0]
        if np.shape(self.B_res)[0] == 0:
            y_rebuild = y_com
        else:
            y_res = self.B_res @ data + noise(self.S_res)
            S_add = np.eye(size) - (self.A1 @ self.A1.T + self.A2 @ self.A2.T)
            y_rebuild = self.A1 @ y_com + self.A2 @ y_res + noise(S_add)
        y = stand_to_origin(self.B, y_rebuild, self.B0, self.S0)
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


def dfs(args, node, data, query, y_list, y_prev=None):
    """Run a dfs search on the NArrTree."""
    # nonlocal query
    if node is None:
        return
    left, right = node.key
    data_split = data[left: right]
    if len(node.children) == 0:
        return
    else:
        sub_query = node.child_query()
        sub_com_query = node.child_com_query()
        size = sub_query.shape[0]

        B_com = node.com_mat
        # if y_prev is None:
        #     y_com = B_com @ data_split + noise(cov_mat(B_com))
        # else:
        #     y_com = np.array(y_prev).flatten()

        y_com = B_com @ data_split + noise(cov_mat(B_com))
        x_est = least_square(B_com, y_com, args)
        assert len(y_com) == 1, "Not sum query"
        sm = np.sum(x_est) * 2 * (size - 1) / size

        if sm <= np.sqrt(2.0/np.pi) * size * args.sigma:
            query.append(node.mat)
            cur_mat = node.mat
            cur_cov = cov_mat(cur_mat)
            cur_mech = Mechanism(args, cur_mat, cur_cov)
            y_cur = cur_mech.get_noisy_answer(data_split, B_com, cov_mat(B_com), y_com)
            for y in y_cur:
                y_list.append(y)
            return
        else:
            if sub_com_query is None:
                sub_mat = node.child_query()
                sub_cov = cov_mat(sub_mat)
                sub_mech = Mechanism(args, sub_mat, sub_cov)
                y_sub = sub_mech.get_noisy_answer(data_split, B_com, cov_mat(B_com), y_com)
                for i, child in enumerate(node.children):
                    query.append(child.mat)
                    y_list.append(y_sub[i])
                return
            else:
                sub_com_mat = sub_com_query
                sub_com_cov = cov_mat(sub_com_mat)
                sub_com_mech = Mechanism(args, sub_com_mat, sub_com_cov)
                y_sub_com = sub_com_mech.get_noisy_answer(data_split, B_com, cov_mat(B_com), y_com)
                for i, child in enumerate(node.children):
                    y_sub = y_sub_com[i]
                    # child_left, child_right = child.key
                    # child_size = child_right - child_left
                    # print(B_com)
                    # mat_sub = np.max(sub_com_mat[i, :]) * np.ones(child_size)
                    dfs(args, child, data, query, y_list, y_sub)


def run_experiment_adaptive(args, dataset, mech_list, clf):
    """Run experiment."""
    err_thres = []
    population = []
    N = np.shape(dataset)[0]
    rand_index = np.random.choice(range(N), args.blocks, replace=True)

    com_mech = CommonMechanism(args, mech_list)
    correct_count = 0
    total_count = 0

    for i in range(N):
        data = dataset[i, :]
        data_sum = np.sum(data)
        population.append(data_sum)
        com_mech.input_data(data)

        vec_k1 = np.array([8])
        vec_k2 = np.array([2, 3, 2, 2, 2, 3, 2, 2])
        vec_k3 = np.array([1, 3, 4, 2, 2, 2, 3, 3, 3, 1, 3, 4, 2, 2, 2, 3, 3, 3])
        mech1, mech2, mech3, mech4 = mech_list
        noisy_sum = np.sum(data) + noise(np.eye(1)*args.sigma**2 / args.gamma)
        # mech_id = get_mech_id(noisy_sum, thres)
        mech_id = clf.predict([noisy_sum])[0]
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
            is_correct = (true_id == mech_id)
            if is_correct:
                correct_count += 1
            total_count += 1

        if i % 20000 == 0:
            print(f"----------------------- Round {i}  --------------------------")

    print("Correct count: ", correct_count, "Ratio: ", correct_count/total_count*100)
    return com_mech



