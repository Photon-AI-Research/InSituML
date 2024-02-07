import pytest
from encoder_decoder import Encoder, MLPDecoder
from unittest.mock import patch, Mock
import os 
import torch



def test_main_e2d():
    args_to_add = '--pathpattern1 data/{}.npy '
    args_to_add += '--pathpattern2 data/{}.npy '
    args_to_add += '--t0 1999 '
    args_to_add += '--t1 2001 '
    
    os.system("python3 main.py " + args_to_add)
    assert False
