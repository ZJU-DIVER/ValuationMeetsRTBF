# -*- coding: utf-8 -*-
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
sshap.utils
~~~~~~~~~~~
This module provides utility functions that are used within sshap.
"""

import scipy.stats as st
import numpy as np
import torch

import pandas as pd

from sklearn.model_selection import train_test_split

import time
from itertools import chain, combinations
from sklearn import metrics
from typing import Iterator

from sklearn.datasets import load_svmlight_file

from libsvm.svmutil import *

from .shards import ShardedStructure
from .models.basic_operators import batch_agg


def gen_perm_seed(m, num_proc, seed_limit=10000):
    """
    Generate the splited permutation numbers and seeds for processes
    """
    local_ms = split_permutation_num(m, num_proc)
    seeds = np.random.choice(seed_limit, num_proc, replace=False)
    args = [(local_m, seed) for (local_m, seed) in zip(local_ms, seeds)]
    return args


def reproduce(seed=3407) -> None:
    """
    For reproducibility, set random seed
    :return: None
    """
    # Set numpy print precision
    np.set_printoptions(precision=6, suppress=True)
    np.random.seed(seed)
    torch.manual_seed(seed)


def eval_utility(x_train, y_train, x_test, y_test, model, ls=None,
                 metric=metrics.accuracy_score, aggregation='voting') -> float:
    """Evaluate the coalition utility.
    """
    if len(y_train) == 0:
        return 0

    if ls is not None:
        if len(ls.idxes_available) == 0:
            return 0
        # Check the type
        if not isinstance(ls, ShardedStructure):
            raise Exception('Unsupported ls parameter')
        else:
            # Check depth
            if ls.depth == 1:
                # Do shards
                x_trs, y_trs = list(), list()
                for partition in ls.nl:
                    idxes = list(set(partition) & set(ls.idxes_available))
                    # Skip empty list
                    if len(idxes) == 0:
                        continue
                    x_trs.append(x_train[idxes])
                    y_trs.append(y_train[idxes])
                return ensemble_eval_utility(x_trs, y_trs, x_test, y_test, model,
                                             metric, aggregation)
            else:
                raise Exception('Unsupported ls depth, only eval 1-level')

    single_pred_label = (True if len(np.unique(y_train)) == 1
                         else False)

    if single_pred_label:
        y_pred = [y_train[0]] * len(y_test)
    else:
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

    return metric(y_test, y_pred)


def ensemble_eval_utility(x_trs, y_trs, x_test, y_test, model,
                          metric=metrics.accuracy_score,
                          aggregation='voting', device='cpu'):
    t_time = 0  # training
    a_time = 0  # aggregation, include inference

    sub_model_pred = list()
    for (x_tr, y_tr) in zip(x_trs, y_trs):
        if len(y_tr) == 0:
            # @deprecated: Empty shard then fill pred with None
            # sub_model_pred.append([None] * len(y_test))
            continue
        # Train data only one label
        single_pred_label = (True if len(np.unique(y_tr)) == 1
                             else False)
        if single_pred_label:
            pred = np.zeros(len(y_test), dtype=np.int8)
            pred.fill(y_tr[0])  # fill with the only label (class)
            sub_model_pred.append(pred)
        else:
            begin = time.time()
            model.fit(x_tr, y_tr)
            end = time.time()
            t_time += end - begin

            begin = time.time()
            sub_model_pred.append(np.array(model.predict(x_test),
                                           dtype=np.int8))
            end = time.time()
            a_time += end - begin

    sub_model_pred = np.array(sub_model_pred, dtype=np.int8)

    # Aggregation the results in sub_model_pred
    final_pred = list()
    if aggregation == 'voting':
        begin = time.time()
        if device == 'cpu':
            # e.g. sub_model_pred = [[1, 2], [1, 1]] => [1, 2]
            for i in range(len(y_test)):
                unique, counts = np.unique(sub_model_pred[:, i], return_counts=True)
                # TODO: Consider the same votes
                final_pred.append((unique[np.argsort(counts)])[-1])
        else:
            # TODO: not practical, out of mem
            gpu_data = torch.tensor(sub_model_pred, dtype=torch.int8).cuda(device=torch.device('cuda:0'))
            res = batch_agg(gpu_data.T, 3)
            final_pred = res.numpy()  # TODO + 1
            del gpu_data
        end = time.time()
        a_time += end - begin
        return metric(y_test, final_pred), t_time, a_time
    # elif aggregation == 'weighted voting':
    #     weight = list()
    #     for i in range(len(y_trs)):
    #         if w := len(y_trs[i]):
    #             weight.append(w)
    #
    #     for i in range(len(y_test)):
    #         votes = list()
    #         for j in range(len(sub_model_pred)):
    #             votes += [sub_model_pred[j, i]] * weight[j]
    #         unique, counts = np.unique(votes, return_counts=True)
    #         final_pred.append((unique[np.argsort(counts)])[-1])
    #     return metric(y_test, final_pred)
    else:
        raise Exception('Unsupported aggregation method')


def power_set(iterable) -> Iterator:
    """Generate the power set of the all elements of an iterable obj.
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r)
                               for r in range(1, len(s) + 1))


def time_function(f, *args) -> float:
    """Call a function f with args and return the time (in seconds)
    that it took to execute.
    """

    tic = time.time()
    f(*args)
    toc = time.time()
    return toc - tic


def get_ele_idxes(ele, ele_list) -> list:
    """Return all index of a specific element in the element list
    """
    idx = -1
    if not isinstance(ele_list, list):
        ele_list = list(ele_list)
    n = ele_list.count(ele)
    idxes = [0] * n
    for i in range(n):
        idx = ele_list.index(ele, idx + 1, len(ele_list))
        idxes[i] = idx
    return idxes


def eval_svc(new_sv, origin_sv) -> np.ndarray:
    """Return Shapley value change array
    """
    return new_sv - origin_sv


def split_permutation_num(m, num) -> np.ndarray:
    """Split a number into num numbers

    e.g. split_permutations(9, 2) -> [4, 5]
         split_permutations(9, 3) -> [3, 3, 3]

    :param m: the original num
    :param num: split into num numbers
    :return: np.ndarray
    """

    assert m > 0
    quotient = int(m / num)
    remainder = m % num
    if remainder > 0:
        perm_arr = [quotient] * (num - remainder) + [quotient + 1] * remainder
    else:
        perm_arr = [quotient] * num
    return np.asarray(perm_arr)


def split_permutations_t_list(permutations, t_list, num) -> list:
    """Split permutations and t_list
    :param permutations: the original num
    :param t_list: the t list
    :param num: split into num numbers
    :return: list
    """

    m = len(permutations)
    m_list = split_permutation_num(m, num)
    res = list()
    for local_m in m_list:
        res.append([permutations[:local_m], t_list[:local_m]])
        permutations = permutations[local_m:]
        t_list = t_list[local_m:]
    return res

# ====================================================
# Metrics
# ====================================================

def mse(list1, list2):
    """
    compare the mse * n (list2 to list1)
    """
    try:
        if round(float(np.sum(list1)), 10) != round(float(np.sum(list2)), 10):
            raise Exception('[*] Variance is invalid with different means!\n'
                            '[*] Info: mean1 = %f mean2 = %f'
                            % (float(np.mean(list1)), float(np.mean(list2))))
    except Exception as err:
        print(err)
    else:
        return ((np.copy(list2) - np.copy(list1)) ** 2).sum()


def normalize(list1, list2):
    """
    normalize list2 to list1
    """
    coef = np.sum(list1) / np.sum(list2)
    return coef * list2


def z_to_p(z_score, two_sided=False):
    p = (1 - st.norm.cdf(abs(z_score)))
    if two_sided:
        p *= 2
    return p


def t_to_p(t_score, df, two_sided=False):
    p = st.t.sf(abs(t_score), df)
    if two_sided:
        p *= 2
    return p


def get_p_value(diff_arr, test='t', two_sided=False):
    if test == 't':
        # use t test to get p value
        avg = np.average(diff_arr)
        square_sum = np.sum(diff_arr ** 2)
        t_score = avg / (np.sqrt((square_sum - avg ** 2 * diff_arr.size) / ((diff_arr.size - 1) * diff_arr.size)))
        return t_to_p(t_score, diff_arr.size - 1, two_sided)
    elif test == 'p':
        raise Exception()
    else:
        raise Exception()


def mean_square_error(x, y):
    return np.average((x - y) ** 2)


def eval_v(v_list1, v_list2, normalize=0.0):
    assert v_list1.shape == v_list2.shape

    p_val = get_p_value(v_list1 - v_list2)

    v_list1_avg = np.average(v_list1, axis=0)
    v_list2_avg = np.average(v_list2, axis=0)
    if normalize != 0:
        v_list1 *= (normalize / np.sum(v_list1_avg))
        v_list2 *= (normalize / np.sum(v_list2_avg))

    mse = mean_square_error(v_list1_avg, v_list2_avg)
    k_tau, k_p = st.kendalltau(v_list1_avg, v_list2_avg)

    return mse, k_tau, k_p, p_val


def eval_mse_kt(v_list1, v_list2, normalize=0.0):
    assert v_list1.shape == v_list2.shape

    if normalize != 0:
        v_list1 *= (normalize / np.sum(v_list1))
        v_list2 *= (normalize / np.sum(v_list2))

    mse = mean_square_error(v_list1, v_list2)
    k_tau, k_p = st.kendalltau(v_list1, v_list2)

    return mse, k_tau, k_p

# ====================================================
# preprocessing datasets
# ====================================================

def load_data(file_name, form='libsvm', path_prefix='', m_test_size=None, m_seed=None):
    
    test_size   = 0.2 if m_test_size == None else m_test_size
    random_seed = 42  if m_seed      == None else m_seed

    if form == 'csv':
        if file_name == 'iris':
            df = pd.read_csv(path_prefix + "datasets/iris/iris.data", header=None)

            # Column names
            df.columns = ['SepalLength', 'SepalWidth',
                          'PetalLength', 'PetalWidth', 'Class']

            # Changes string to float
            df.SepalLength = df.SepalLength.astype(float)
            df.SepalWidth = df.SepalWidth.astype(float)
            df.PetalLength = df.PetalLength.astype(float)
            df.PetalWidth = df.PetalWidth.astype(float)

            # Sets label name as Y
            df = df.rename(columns={'Class': 'Y'})  
            X = df.drop(columns=['Y']).values
            y = df.Y.values

        elif file_name == 'breast_cancer':
            df = pd.read_csv(path_prefix + "datasets/breast_cancer/breast-cancer-wisconsin.data")
            # Drop index
            df.drop(columns=['i'])
            X = df.drop(columns=['Y']).values
            y = df.Y.values
        
        else:
            raise Exception('unsupported dataset name')

    elif form == 'libsvm':
        if file_name == 'a9a':
            n_features = 123
            X_train, y_train = load_svmlight_file("./datasets/a9a/a9a", n_features=n_features)
            X_test, y_test = load_svmlight_file("./datasets/a9a/a9a.t", n_features=n_features)
            X_train = X_train.todense()
            X_test = X_test.todense()
            X = np.concatenate((X_train, X_test))
            y = np.concatenate((y_train, y_test))
        elif file_name == 'iris':
            n_features = 4
            X, y = load_svmlight_file("./datasets/iris/iris.scale", n_features=n_features)
            X = X.todense()
        elif file_name == 'wine':
            n_features = 13
            X, y = load_svmlight_file("./datasets/wine/wine.scale", n_features=n_features)
            X = X.todense()
        else:
            raise Exception('unsupported dataset name')
    else:
        raise Exception('unsupported dataset format')
    
    # Check y value type, force to 0, 1, 2, ... 
    classes = np.unique(y)
    new_y = np.zeros(len(y), dtype=np.int8)
    for i, c in enumerate(classes):
        new_y[y==c] = i
    y = new_y
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        random_state=random_seed)
    return X_train, y_train, X_test, y_test


# ====================================================
# for compared algorithm
# ====================================================

def beta_weight(alpha, beta, j, n):
    """
    gen weight for beta shapley
    :param alpha:
    :param beta:
    :param j: current pos !from 1 to n
    :param n: total num
    :return:
    """
    w = float(n)
    for k in range(1, n): # from 1 to n - 1
        w /= alpha + beta + k - 1
        if k <= j - 1:
            w *= beta + k - 1
            # solve C(n-1 j-1)-1 by multiply C(n-1 j-1)
            w *= (n-k) / k
        else:
            # from j to n - 1
            temp_k = k - j + 1
            w *= alpha + temp_k - 1
    return w


def beta_weight_list(alpha, beta, n):
    w_list = []
    for j in range(1, n+1):
        w_list.append(beta_weight(alpha, beta, j, n))
    return w_list
