# -*- coding: utf-8 -*-
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from sshap.shards import *

class TestShards(object):
    def setup(self):
        pass
    
    def test_perm(self):
        for _ in range(100):
            # draw one permutation from Omega_1
            pi = nl2l(sample_perm([[0], [1, 2]]))
            assert pi != [1, 0, 2]
            assert pi != [2, 0, 1]

        assert check_perm_valid([1, 0, 2], [[0], [1, 2]]) is False
        assert check_perm_valid([0, 1, 2], [[0], [1, 2]]) is True

        assert check_perm_valid([1, 0, 2, 3, 4], [[[0], [1, 2]], [[3], [4]]]) is False
        assert check_perm_valid([0, 1, 2, 3, 4], [[[0], [1, 2]], [[3], [4]]]) is True

        ls_level1 = ShardedStructure([[0], [1, 2]])
        assert gen_all_permutations(ls_level1) == [[0, 1, 2], [0, 2, 1], [1, 2, 0], [2, 1, 0]]
        
if __name__ == '__main__':
    pytest.main("-s test_shards.py")