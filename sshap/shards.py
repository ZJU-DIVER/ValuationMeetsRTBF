# -*- coding: utf-8 -*-
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
sshap.shards
~~~~~~~~~~~~
This module provides sharded structure relation functions that are used within sshap.
We use a nested list to represent the sharded structure.

e.g. A Sharded Structure represented as [[0],[1,2]]
    {0,1,2}
    /     \
   {0}   {1,2}
    |     / \
   {0}  {1} {2}
"""

from typing import List
import numpy as np
import itertools


class ShardedStructure(object):
    nl: List  # a nested list to represent sharded structure
    idxes_available: List  # data for training
    
    num_ele: int

    def __init__(self, nl):
        self.nl = nl
        self.check_nl(nl)
        
        self.idxes_available = nl2l(nl)
        self.num_ele = len(self.idxes_available)
        
        self.corr_shard = np.zeros(self.num_ele)
        self.len_shard = np.zeros(self.num_ele)

        # save each data point corresponding shard idx
        for i, s in enumerate(self.nl):
            len_s = len(s)
            for idx in s:
                self.corr_shard[idx] = i
                self.len_shard[idx] = len_s

    @property
    def max_shard_size(self):
        return np.max(np.array(self.len_shard))
    
    @staticmethod
    def check_nl(nl: List):
        assert isinstance(nl, List)
        assert isinstance(nl[0], List)
        assert not isinstance(nl[0][0], List)


def sample_perm(nl: List, prng=np.random, form='flat'):
    """Sample a valid permutation with a given sharded structure.
    """
    num_el = len(nl)
    idxes = np.arange(num_el)
    prng.shuffle(idxes)
    temp_nl = list()
    for i in range(num_el):
        temp_nl.append(nl[idxes[i]])
    nl = temp_nl  # shuffle in the current level
    for i, el in enumerate(nl):
        if isinstance(el, List):
            nl[i] = sample_perm(el, prng, form)
    return nl2l(nl) if form == 'flat' else nl


def gen_all_permutations(ls: ShardedStructure):
    """Generate a list contains all valid permutations

    e.g. [[0],[1,2]] -> [[0,1,2], [0,2,1], [1,2,0], [2,1,0]]
    """
    perms = list()
    for p in list(itertools.permutations(np.arange(ls.ele_num))):
        p = list(p)
        if check_perm_valid(p, ls.nl):
            perms.append(p)
    return perms


def check_perm_valid(perm, ss, flag=True) -> bool:
    """Check whether a permutation is valid for the given L.S.
    """
    nl = ss.nl
    if not isinstance(nl[0], List):
        return True
    i = 0
    while i < len(perm):
        # Check the first element
        el_idx = perm[i]
        l, l_size = find_list_in_nl(nl, el_idx)
        # Cut piece
        piece = perm[i:i + l_size]
        i += l_size
        if set(piece) != set(nl2l(l)):
            return False
        else:
            flag = check_perm_valid(piece, l, flag) and flag
    return flag


def find_list_in_nl(nl: List, el_idx: int):
    """Find the list in nested list contains the el_idx

    e.g. [[1],[2]] 2 -> [2]
    """
    if not isinstance(nl[0], List):
        return None, 0
    for i in range(len(nl)):
        flat_l = nl2l(nl[i])
        if el_idx in flat_l:
            return nl[i], len(flat_l)
    return None, 0


def nl2l(nl: List):
    """Remove all nested bracket

    e.g. [[1],[2]] -> [1,2]
    """
    while isinstance(nl[0], List):
        nl = [item for sublist in nl for item in sublist]
    return nl
