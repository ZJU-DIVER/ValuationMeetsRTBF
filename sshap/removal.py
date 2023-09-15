# -*- coding: utf-8 -*-
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
sshap.removal
~~~~~~~~~~~~~
This module provides various level shapley value dynamic computation methods.
"""
import copy
from typing import List
from functools import partial
from multiprocessing import Pool
import numpy as np
from tqdm import trange

from .levels import LevelStruct, sample_perm, nl2l
from .utils import gen_perm_seed, remove_ele_in_nl, eval_utility


class Delta(object):
    """
    Efficient algorithms for lsv updating after data removal
    """
    def __init__(self, x_tr, y_tr, x_val, y_val, model, ls: LevelStruct, init_lsv):
        self.x_tr = x_tr
        self.y_tr = y_tr
        self.x_val = x_val
        self.y_val = y_val

        self.model = model
        self.ls = ls
        self.init_lsv = init_lsv

    def del_point(self, delete_idx: int, m: int, proc_num: int, con_update=True) -> np.ndarray:
        """
        Algo Delta in the paper
        delete single data point
        :param delete_idx: the index of the point which need to be removed
        :param m: the number of permutations
        :param proc_num: the number of process
        :param con_update: continuous update
        :return: updated lsv
        """
        if proc_num < 0:
            raise ValueError('Invalid proc num.')
        ls = self.ls
        model = self.model
        n = len(ls.idxes_available)
        # target: estimate two average utility for each data point
        dau1 = np.zeros(n) # + 
        dau2 = np.zeros(n) # -
        m_num = len(self.ls.nl)
        # Assign the permutation and seed of each proc
        args = gen_perm_seed(m, proc_num)

        pool = Pool()
        func = partial(self._del_point_sub_task, self.x_tr, self.y_tr, 
                       self.x_val, self.y_val, model, copy.deepcopy(ls), delete_idx)
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
        # print(p_cnt)
        # Average over each strata
        # p_cnt[p_cnt == 0] = 1  # avoid div 0
        # n_cnt[:, 0] = 1
        p_cnt[p_cnt == 0] = 1
        n_cnt[n_cnt == 0] = 1
        delta = np.zeros(n)
        print('p', p_contrib)
        print('p cnt', p_cnt)
        print('n', n_contrib)
        print('n cnt', n_cnt)
        num_shards = len(ls.nl)
        new_ls = LevelStruct(1, remove_ele_in_nl(ls.nl, delete_idx))
        del_part_idxes = list(filter(lambda x: delete_idx in x, ls.nl))[0]
        for i in range(n-1):
            # print(n_cnt)
            # print(p_cnt)
            exact_num_strata = num_shards * new_ls.find_part_len(i)
            if i in del_part_idxes:
                for stra in range(exact_num_strata):
                    c = (stra) % (new_ls.find_part_len(i))
                    coef = (c) / (new_ls.find_part_len(i))
                    delta[i] += coef * (p_contrib[i][stra] / p_cnt[i][stra] - n_contrib[i][stra] / n_cnt[i][stra])
            else:
                for stra in range(exact_num_strata):
                    b = (stra) // (new_ls.find_part_len(i)) + 1
                    coef = (b-1) / (m_num-1)
                    delta[i] += coef * (p_contrib[i][stra] / p_cnt[i][stra] - n_contrib[i][stra] / n_cnt[i][stra])
                print(delta[i] / exact_num_strata)
                # delta[i] = np.sum(p_contrib[i][:exact_num_strata] / p_cnt[i][:exact_num_strata] - n_contrib[i][:exact_num_strata] / n_cnt[i][:exact_num_strata])
            delta[i] /= exact_num_strata 
        print(delta)
        res = (self.init_lsv + delta)[np.delete(np.arange(len(self.init_lsv)), [delete_idx])]
        if con_update:
            self.init_lsv = res  # update for con update
            self.ls = new_ls
        return delta

    @staticmethod
    def _del_point_sub_task(x_tr, y_tr, x_val, y_val, model, ls : LevelStruct, del_idx, args):
        assert ls.depth == 1
        old_ls = copy.deepcopy(ls)
        new_ls = LevelStruct(1, remove_ele_in_nl(copy.deepcopy(old_ls.nl), del_idx))
    
        local_m = args[0]
        seed = args[1]
        prng = np.random.RandomState(seed)

        n = len(ls.idxes_available)
        max_num_strata = ls.max_leaf_size * len(ls.nl)

        p_cnt = np.zeros(shape=(n, max_num_strata))
        n_cnt = np.zeros(shape=(n, max_num_strata))
        p_contrib = np.zeros(shape=(n, max_num_strata))
        n_contrib = np.zeros(shape=(n, max_num_strata))

        del_part_idxes_with_del_idx = list(filter(lambda x: del_idx in x, ls.nl))[0]
        del_part_idxes_no_del_idx = list(del_part_idxes_with_del_idx)
        del_part_idxes_no_del_idx.remove(del_idx)

        for _ in trange(local_m):
            old_ls = copy.deepcopy(ls)
            nest_perm = sample_perm(old_ls.nl, prng, form='nest')
            flat_perm = nl2l(nest_perm)
            flat_perm.remove(del_idx)

            start_idx = 0
            # get the start idx of del part
            for cnt, idx in enumerate(flat_perm):
                if idx in del_part_idxes_no_del_idx:
                    start_idx = cnt
                    break
            
            # flat_perm.insert(del_idx)
            
            # for i in range(start_idx+1, n):
            for _ in range(1):
                i = np.random.randint(start_idx+1, n)
                old_ls.idxes_available = flat_perm[:i]
                if i == 0:
                    u_no_del = 0
                else:
                    u_no_del, _, _ = eval_utility(x_tr, y_tr, x_val, y_val, model, old_ls)

                old_ls.idxes_available.insert(0, del_idx)
                u_with_del, _, _ = eval_utility(x_tr, y_tr, x_val, y_val, model, old_ls)
                delta_u = u_no_del - u_with_del

                cursor_index = flat_perm[i-1]
                cols_no_del = flat_perm[:i]
                flag_bound = True
                current_part = list(filter(lambda x: cursor_index in x, new_ls.nl))[0]

                if len(current_part) == len(set(current_part).intersection(set(cols_no_del))):
                    flag_bound = True
                else:
                    flag_bound = False

                b = 0
                for part in new_ls.nl:
                    if len(set(part).intersection(set(cols_no_del))) != 0:
                        b += 1
            
                c = len(set(current_part).intersection(set(cols_no_del)))
           
                if flag_bound:
                    assert c == len(current_part)
                    for idx in flat_perm:
                        if idx in flat_perm[:i]:
                            strata = b * new_ls.find_part_len(idx) - 1
                            p_contrib[idx][strata] += delta_u
                            p_cnt[idx][strata] += 1
                        else:
                            strata = b * new_ls.find_part_len(idx)
                            n_contrib[idx][strata] += delta_u
                            n_cnt[idx][strata] += 1
                else:
                    assert c < len(current_part)
                    for idx in current_part:
                        if idx in flat_perm[:i]:
                            strata = (b-1) * new_ls.find_part_len(idx) + (c-1)
                            p_contrib[idx][strata] += delta_u
                            p_cnt[idx][strata] += 1
                        else:
                            strata = (b-1) * new_ls.find_part_len(idx) + c
                            n_contrib[idx][strata] += delta_u
                            n_cnt[idx][strata] += 1  
        return p_cnt, n_cnt, p_contrib, n_contrib


    def del_points(self, delete_idxes: List[int], m: int, proc_num: int) -> np.ndarray:
        """
        algo BDelta in the paper
        delete multiple data points
        :param delete_idxes: the indexes of the points which need to be removed
        :param m: the number of permutations
        :param proc_num: the number of process
        :return: updated lsv
        """
        if proc_num < 0:
            raise ValueError('Invalid proc num.')
        ls = self.ls
        n = len(self.ls.idxes_available)
        # target: estimate two average utility for each data point
        dau1 = np.zeros(n) # + 
        dau2 = np.zeros(n) # -
        
        # Assign the permutation and seed of each proc
        args = gen_perm_seed(m, proc_num)

        pool = Pool()
        func = partial(self._del_points_sub_task, self.x_tr, self.y_tr, 
                       self.x_val, self.y_val, model, self.ls)
        ret = pool.map(func, args)
        pool.close()
        pool.join()
        
        max_num_strata = self.ls.max_leaf_size * len(self.ls.nl)
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
        p_cnt[p_cnt == 0] = 1
        n_cnt[n_cnt == 0] = 1
        delta = np.zeros(n)
        num_shards = len(self.ls.nl)
        new_ls = copy.deepcopy(self.ls)
        for del_idx in delete_idxes:
            new_ls = LevelStruct(1, remove_ele_in_nl(new_ls.nl, del_idx))
            
        for i in range(n):
            # print(n_cnt)
            # print(p_cnt)
            exact_num_strata = num_shards * new_ls.find_part_len(i)
            delta[i] = np.average(p_contrib[i][:exact_num_strata] / p_cnt[i][:exact_num_strata] - n_contrib[i][:exact_num_strata] / n_cnt[i][:exact_num_strata])
        # new_sv = (self.init_lsv + delta)[np.delete(np.arange(n), delete_idxes)]
        return delta

    @staticmethod
    def _del_points_sub_task(x_tr, y_tr, x_val, y_val, model, ls: LevelStruct, del_idxes, args):
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
            # sample a perm without 
            nest_perm = sample_perm(ls.nl, prng, form='nest')
            flat_perm = nl2l(nest_perm)

            cursor = prng.randint(1, n+1)
            cols = flat_perm[:cursor]
            cursor_index = cols[-1]
            current_part = list(filter(lambda x: cursor_index in x, ls.nl))[0]
            flag_bound = True
            if len(current_part) == len(set(current_part).intersection(set(cols))):
                flag_bound = True
            else:
                flag_bound = False
                
            cols_no_del = np.delete(cols, del_idxes)
            ls.idxes_available = col
            u_with_del, _, _ = eval_utility(x_tr, y_tr, x_val, y_val, model, ls)  
            ls.idxes_available = cols_no_del
            u_no_del, _, _ = eval_utility(x_tr, y_tr, x_val, y_val, model, ls)
            
            delta_u = u_no_del - u_with_del
                    
            b = 0
            for part in ls.nl:
                if len(set(part).intersection(set(cols))) != 0:
                    b += 1

            c = len(set(current_part).intersection(set(current_part[:cursor])) - set(del_idxes))

            # get new ls
            new_ls = copy.deepcopy(ls)
            
            if flag_bound:
                assert c == len(current_part)
                for idx in flat_perm:
                    if idx in flat_perm[:cursor]:
                        strata = b * new_ls.find_part_len(idx) - 1
                        p_contrib[idx][strata] += delta_u
                        p_cnt[idx][strata] += 1
                    else:
                        strata = b * new_ls.find_part_len(idx)
                        n_contrib[idx][strata] += delta_u
                        n_cnt[idx][strata] += 1
            else:
                assert c < len(current_part)
                for idx in current_part:
                    if idx in flat_perm[:cursor]:
                        strata = (b-1) * len(current_part) + (c-1)
                        p_contrib[idx][strata] += delta_u
                        p_cnt[idx][strata] += 1
                    else:
                        strata = (b-1) * len(current_part) + c
                        n_contrib[idx][strata] += delta_u
                        n_cnt[idx][strata] += 1  
        return p_cnt, n_cnt, p_contrib, n_contrib