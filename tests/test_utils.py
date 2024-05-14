import sys, os
sys.path.insert(
    0, 
    '/'.join(os.path.abspath(__file__).split('/')[:-2] + ['src', 'inSituML'])
    )


import utilities
import torch

class TestDataStream:

    def test_search_for_directories_with_simulations(self):
        x = torch.rand(2, 3)
        y = torch.rand(2, 3)

        mmd = utilities.MMD_multiscale(x, y)
        assert mmd is not None
