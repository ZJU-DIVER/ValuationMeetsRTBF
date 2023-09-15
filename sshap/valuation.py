# -*- coding: utf-8 -*-
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
sshap.valuation
~~~~~~~~~~~~~~~
This module provides various valuation methods:
Part 1. S-Shapley value computation methods (ours);
Part 2. Shapley value computation methods (for comparison);
Part 3. leave-one-out (LOO) score (for comparison).
"""

import torch.multiprocessing as mp

import itertools

import numpy as np
from functools import partial
from tqdm import trange

from .utils import (eval_utility, gen_perm_seed, beta_weight_list)
from .shards import (sample_perm, check_perm_valid, ShardedStructure, nl2l)

from .config import SHOW_TIMER


# ================= Part 1. Sharded Shapley value (Ours) =================
def exact_ssv(x_tr, y_tr, x_val, y_val, model, ss: ShardedStructure, 
              method='perm') -> np.ndarray:
    """ Computing the exact S-Shapley value

    :param x_tr:  features of train dataset
    :param y_tr:  labels of train dataset
    :param x_val: features of valid dataset
    :param y_val: labels of valid dataset
    :param model: selected constituent models
    :param ss:    sharded structure
    :param method: 'perm' - permutation based; 
                   'coal' - coalition based
    :return: S-Shapley value array `ssv`
    :rtype: np.ndarray
    """

    n = len(ss.idxes_available)
    ssv = np.zeros(n)
    cnt = 0
    if method == 'perm':
        perms = itertools.permutations(np.arange(n))
        for p in perms:
            old_u = 0
            if check_perm_valid(p, ss) is False:
                continue  # invalid then omit
            for j in range(1, n + 1):
                ss.idxes_available = p[:j]
                temp_u = eval_utility(x_tr, y_tr, x_val, y_val, model, ss)
                margin_contrib = temp_u - old_u
                ssv[p[j - 1]] += ssv[p[j - 1]] * cnt / (1 + cnt)  \
                                 + margin_contrib / (cnt + 1)
                cnt += 1
                old_u = temp_u
    else:
        # TODO: coalition based
        raise Exception('coalition based method has not been supported')
    
    return ssv


def monte_carlo_ssv(x_tr, y_tr, x_val, y_val, model, ss: ShardedStructure,
                    m: int, num_proc=1) -> np.ndarray:
    """ Approximating S-Shapley value via Monte Carlo simulation

    :param x_tr:  features of train dataset
    :param y_tr:  labels of train dataset
    :param x_val: features of valid dataset
    :param y_val: labels of valid dataset
    :param model: selected model
    :param ss: sharded structure
    :param m: the number of permutations
    :param num_proc: set proc num for parallel
    :return: S-Shapley value array `ssv`
    :rtype: np.ndarray
    """

    if num_proc < 0:
        raise ValueError('Invalid number of processes.')

    # Assign the permutation and seed of each proc
    args = gen_perm_seed(m, num_proc)

    pool = mp.Pool()
    func = partial(_srs_ssv_sub_task, x_tr, y_tr, x_val, y_val, model, ls)
    ret = pool.map(func, args)
    pool.close()
    pool.join()

    ret_arr = np.asarray(ret)
    return np.sum(ret_arr, axis=0) / m


def _srs_ssv_sub_task(x_tr, y_tr, x_val, y_val, model, ls: ShardedStructure, arg) -> np.ndarray:
    n = len(ls.idxes_available)
    ssv = np.zeros(n)

    local_m = arg[0]  # the number of the permutations in this proc
    seed = arg[1]
    prng = np.random.RandomState(seed)

    t_times = 0
    v_times = 0

    for _ in trange(local_m):
        # Sample a valid permutation
        perm = sample_perm(ls.nl, prng)
        old_u = 0
        for j in range(1, n + 1):
            # Evaluate utility of j-length coalition in the perm
            ls.idxes_available = perm[:j]
            temp_u, t_time, v_time = eval_utility(x_tr, y_tr, x_val, y_val, model, ls)
            t_times += t_time
            v_times += v_time
            contrib = temp_u - old_u
            ssv[perm[j - 1]] += contrib
            old_u = temp_u

    if SHOW_TIMER:
        print(f't {t_times} v {v_times}')
    return ssv


def us_ssv(x_tr, y_tr, x_val, y_val, model, ls: ShardedStructure, m: int, num_proc=1) -> np.ndarray:
    """ Approximating S-Shapley value through utility sampling (multi-process)

    :param x_tr: features of train dataset
    :param y_tr: labels of train dataset
    :param x_val: features of valid dataset
    :param y_val: labels of valid dataset
    :param model: selected model
    :param ls: levels structure as a nested list
    :param m: the number of permutations
    :param num_proc: set proc num for parallel execution
    :return: S-Shapley value array `ssv`
    :rtype: np.ndarray
    """

    if num_proc < 0:
        raise ValueError('Invalid proc num.')
    
    n = len(ls.idxes_available)
    # target: estimate two average utility for each data point
    au1 = np.zeros(n)
    au2 = np.zeros(n)
    
    # Assign the permutation and seed of each proc
    args = gen_perm_seed(m, num_proc)

    pool = mp.Pool()
    func = partial(_us_ssv_sub_task, x_tr, y_tr, x_val, y_val, model, ls)
    ret = pool.map(func, args)
    pool.close()
    pool.join()
    
    max_num_strata = ls.max_leaf_size * len(ls.nl)
    p_cnt = np.zeros(shape=(n, max_num_strata))
    n_cnt = np.zeros(shape=(n, max_num_strata))
    p_contrib = np.zeros(shape=(n, max_num_strata))
    n_contrib = np.zeros(shape=(n, max_num_strata))

    for (r1, r2, r3, r4) in ret:
        p_cnt += r1
        n_cnt += r2
        p_contrib += r3
        n_contrib += r4
    
    # Average over each strata
    # p_cnt[p_cnt == 0] = 1  # avoid div 0
    n_cnt[:, 0] = 1
    
    num_shards = len(ls.nl)
    for i in range(n):
        print(n_cnt)
        print(p_cnt)
        exact_num_strata = num_shards * ls.find_part_len(i)
        au1[i] = np.average(p_contrib[i][:exact_num_strata] / p_cnt[i][:exact_num_strata])
        au2[i] = np.average(n_contrib[i][:exact_num_strata] / n_cnt[i][:exact_num_strata])
    return au1 - au2


def _us_ssv_sub_task(x_tr, y_tr, x_val, y_val, model, ls: ShardedStructure, args):
    assert ls.depth == 1
    
    # load config
    local_m = args[0]
    seed = args[1]
    prng = np.random.RandomState(seed)

    n = len(ls.idxes_available)
    max_num_strata = ls.max_leaf_size * len(ls.nl)

    # For each data point, the num of strata is different
    p_cnt = np.zeros(shape=(n, max_num_strata))
    n_cnt = np.zeros(shape=(n, max_num_strata))
    p_contrib = np.zeros(shape=(n, max_num_strata))
    n_contrib = np.zeros(shape=(n, max_num_strata))

    for _ in trange(local_m):
        # Generate a perm in nest form and flat form
        nest_perm = sample_perm(ls.nl, prng, form='nest')
        flat_perm = nl2l(nest_perm)

        # Sample pos from 1 to n, exclude 0 as empty utility is zero
        cursor = prng.randint(1, n+1)
        ls.idxes_available = flat_perm[:cursor]
        u, _, _ = eval_utility(x_tr, y_tr, x_val, y_val, model, ls)

        # Save utility into corresponding strata
        part_r = 0
        for part_no, part in enumerate(nest_perm):
            part_l = part_r
            if part_l >= cursor:
                break
            part_r += len(part)
            # Current part - flat_perm[part_l: part_r]
            if part_r == cursor and OPT_BOUNDARY:
                # Hit the boundary
                for i in range(0, cursor):
                    # Point in coalition flat_perm[:cursor] +
                    strata = (part_no + 1) * ls.find_part_len(flat_perm[i]) - 1  # [part_no][part_len-1]
                    p_contrib[flat_perm[i]][strata] += u
                    p_cnt[flat_perm[i]][strata] += 1
                for i in range(cursor, n):
                    # Point NOT in coalition flat_perm[:cursor] -
                    strata = (part_no + 1) * ls.find_part_len(flat_perm[i])
                    n_contrib[flat_perm[i]][strata] += u
                    n_cnt[flat_perm[i]][strata] += 1
                continue
            # Normal case, only used in the partition
            if part_r >= cursor:
                part_len = len(part)
                for i in range(part_l, cursor):
                    strata = part_no * part_len + (cursor - 1 - part_l)
                    p_contrib[flat_perm[i]][strata] += u
                    p_cnt[flat_perm[i]][strata] += 1
                for i in range(cursor, part_r):
                    strata = part_no * part_len + (cursor - part_l)
                    n_contrib[flat_perm[i]][strata] += u
                    n_cnt[flat_perm[i]][strata] += 1
    return p_cnt, n_cnt, p_contrib, n_contrib


def us_paired_ssv(x_tr, y_tr, x_val, y_val, model, ls: ShardedStructure, m: int, num_proc=1) -> np.ndarray:
    """ Paried version of us_ssv
    """

    if num_proc < 0:
        raise ValueError('Invalid proc num.')
    
    n = len(ls.idxes_available)
    # target: estimate two average utility for each data point
    au1 = np.zeros(n)
    au2 = np.zeros(n)
    
    # Assign the permutation and seed of each proc
    args = gen_perm_seed(m // 2, num_proc)

    pool = mp.Pool()
    func = partial(_us_paired_ssv_sub_task, x_tr, y_tr, x_val, y_val, model, ls)
    ret = pool.map(func, args)
    pool.close()
    pool.join()
    
    max_num_strata = ls.max_leaf_size * len(ls.nl)
    p_cnt = np.zeros(shape=(n, max_num_strata))
    n_cnt = np.zeros(shape=(n, max_num_strata))
    p_contrib = np.zeros(shape=(n, max_num_strata))
    n_contrib = np.zeros(shape=(n, max_num_strata))

    for (r1, r2, r3, r4) in ret:
        p_cnt += r1
        n_cnt += r2
        p_contrib += r3
        n_contrib += r4
    
    # Average over each strata
    # p_cnt[p_cnt == 0] = 1  # avoid div 0
    n_cnt[:, 0] = 1
    
    num_shards = len(ls.nl)
    for i in range(n):
        print(n_cnt)
        print(p_cnt)
        exact_num_strata = num_shards * ls.find_part_len(i)
        au1[i] = np.average(p_contrib[i][:exact_num_strata] / p_cnt[i][:exact_num_strata])
        au2[i] = np.average(n_contrib[i][:exact_num_strata] / n_cnt[i][:exact_num_strata])
    return au1 - au2


def _us_paired_ssv_sub_task(x_tr, y_tr, x_val, y_val, model, ls: ShardedStructure, args):
    """Each sample here will evaluate two utility
    """
    assert ls.depth == 1
    
    # load config
    local_m = args[0]
    seed = args[1]
    prng = np.random.RandomState(seed)

    n = len(ls.idxes_available)
    max_num_strata = ls.max_leaf_size * len(ls.nl)

    # For each data point, the num of strata is different
    p_cnt = np.zeros(shape=(n, max_num_strata))
    n_cnt = np.zeros(shape=(n, max_num_strata))
    p_contrib = np.zeros(shape=(n, max_num_strata))
    n_contrib = np.zeros(shape=(n, max_num_strata))

    for _ in trange(local_m):
        # Generate a perm in nest form and flat form
        nest_perm_1 = sample_perm(ls.nl, prng, form='nest')
        flat_perm_1 = nl2l(nest_perm_1)

        # Sample pos from 1 to n, exclude 0 as empty utility is zero
        cursor_1 = prng.randint(1, n+1)
        
        nest_perm_2 = nest_perm_1[::-1]
        flat_perm_2 = flat_perm_1[::-1]
        cursor_2 = n - cursor_1
        
        for (nest_perm, flat_perm, cursor) in [(nest_perm_1, flat_perm_1, cursor_1), 
                                               (nest_perm_2, flat_perm_2, cursor_2)]:
            ls.idxes_available = flat_perm[:cursor]
            u, _, _ = eval_utility(x_tr, y_tr, x_val, y_val, model, ls)

            # Save utility into corresponding strata
            part_r = 0
            for part_no, part in enumerate(nest_perm):
                part_l = part_r
                if part_l >= cursor:
                    break
                part_r += len(part)
                # Current part - flat_perm[part_l: part_r]
                if part_r == cursor and OPT_BOUNDARY:
                    # Hit the boundary
                    for i in range(0, cursor):
                        # Point in coalition flat_perm[:cursor] +
                        strata = (part_no + 1) * ls.find_part_len(flat_perm[i]) - 1  # [part_no][part_len-1]
                        p_contrib[flat_perm[i]][strata] += u
                        p_cnt[flat_perm[i]][strata] += 1
                    for i in range(cursor, n):
                        # Point NOT in coalition flat_perm[:cursor] -
                        strata = (part_no + 1) * ls.find_part_len(flat_perm[i])
                        n_contrib[flat_perm[i]][strata] += u
                        n_cnt[flat_perm[i]][strata] += 1
                    continue
                # Normal case, only used in the partition
                if part_r >= cursor:
                    part_len = len(part)
                    for i in range(part_l, cursor):
                        strata = part_no * part_len + (cursor - 1 - part_l)
                        p_contrib[flat_perm[i]][strata] += u
                        p_cnt[flat_perm[i]][strata] += 1
                    for i in range(cursor, part_r):
                        strata = part_no * part_len + (cursor - part_l)
                        n_contrib[flat_perm[i]][strata] += u
                        n_cnt[flat_perm[i]][strata] += 1
    return p_cnt, n_cnt, p_contrib, n_contrib


# ========================== Part 2. Shapley value (for comparison) ==========================
def monte_carlo_sv(x_tr, y_tr, x_val, y_val, model, ls: ShardedStructure, m: int, num_proc=1) -> np.ndarray:
    """ Shapley value computation via Monte Carlo
    """
    if num_proc < 0:
        raise ValueError('Invalid proc num.')
    # Assign the permutation and seed of each proc
    args = gen_perm_seed(m, num_proc)

    pool = mp.Pool()
    func = partial(_monte_carlo_sv_sub_task,
                   x_tr, y_tr, x_val, y_val, model, ls)
    ret = pool.map(func, args)
    pool.close()
    pool.join()

    ret_arr = np.asarray(ret)
    return np.sum(ret_arr, axis=0) / m


def _monte_carlo_sv_sub_task(x_tr, y_tr, x_val, y_val, model, ls: ShardedStructure, arg) -> np.ndarray:
    n = len(y_tr)
    sv = np.zeros(n)

    local_m = arg[0]
    seed = arg[1]
    prng = np.random.RandomState(seed)

    for _ in trange(local_m):
        perm = np.arange(n)
        prng.shuffle(perm)
        old_u = 0
        for j in range(1, n + 1):
            ls.idxes_available = perm[:j]
            temp_u = eval_utility(x_tr, y_tr, x_val, y_val, model, ls)
            contrib = temp_u - old_u
            sv[perm[j - 1]] += contrib
            old_u = temp_u
    return sv


def monte_carlo_sv_beta(x_tr, y_tr, x_val, y_val, model, ls: ShardedStructure, m: int, num_proc=1):
    """ Shapley value and beta Shapley value computation via Monte Carlo
    """
    if num_proc < 0:
        raise ValueError('Invalid proc num.')
    # Assign the permutation and seed of each proc
    args = gen_perm_seed(m, num_proc)
    n = len(y_tr)
    w_lists = [beta_weight_list(4, 1, n),
               beta_weight_list(16, 1, n),
               beta_weight_list(1, 4, n),
               beta_weight_list(1, 16, n)]
    pool = mp.Pool()
    func = partial(_monte_carlo_sv_beta_sub_task,
                   x_tr, y_tr, x_val, y_val, model, ls, w_lists)
    ret = pool.map(func, args)
    pool.close()
    pool.join()

    sv_arr = np.asarray(ret[:, 0])
    beta41_arr, beta161_arr, beta14_arr, beta116_arr = np.asarray(ret[:, 1]), \
                                                       np.asarray(ret[:, 2]), \
                                                       np.asarray(ret[:, 3]), \
                                                       np.asarray(ret[:, 4])
    return np.sum(sv_arr, axis=0) / m, \
           np.sum(beta41_arr, axis=0) / m, \
           np.sum(beta161_arr, axis=0) / m, \
           np.sum(beta14_arr, axis=0) / m, \
           np.sum(beta116_arr, axis=0) / m


def _monte_carlo_sv_beta_sub_task(x_tr, y_tr, x_val, y_val, model, ls: ShardedStructure, w_lists, arg):
    n = len(y_tr)
    sv = np.zeros(n)
    beta41 = np.zeros(n)
    beta161 = np.zeros(n)
    beta14 = np.zeros(n)
    beta116 = np.zeros(n)

    local_m = arg[0]
    seed = arg[1]
    prng = np.random.RandomState(seed)

    for _ in trange(local_m):
        perm = np.arange(n)
        prng.shuffle(perm)
        old_u = 0
        for j in range(1, n + 1):
            ls.idxes_available = perm[:j]
            temp_u = eval_utility(x_tr, y_tr, x_val, y_val, model, ls)
            contrib = temp_u - old_u
            sv[perm[j - 1]] += contrib
            beta41[perm[j - 1]] += contrib * w_lists[0][j-1]
            beta161[perm[j - 1]] += contrib * w_lists[1][j-1]
            beta14[perm[j - 1]] += contrib * w_lists[2][j-1]
            beta116[perm[j - 1]] += contrib * w_lists[3][j-1]
            old_u = temp_u
    return sv, beta41, beta161, beta14, beta116


# ========================== Part 3. LOO (for comparison) ==========================
def loo(x_tr, y_tr, x_val, y_val, model, ls: ShardedStructure) -> np.ndarray:
    """ LOO computation
    """
    n = len(y_tr)
    loo_values = np.zeros(n)
    u = eval_utility(x_tr, y_tr, x_val, y_val, model, ls)
    for i in trange(n):
        idxes = list(np.arange(n))
        idxes.pop(i)
        ls.idxes_available = idxes
        loo_values[i] = u - eval_utility(x_tr, y_tr, x_val, y_val, model, ls)
    return loo_values
