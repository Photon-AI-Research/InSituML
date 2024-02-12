import pytest
import os 
import torch



def test_main_e2d():
    args_to_add = '--pathpattern1 data/{}.npy '
    args_to_add += '--pathpattern2 data/{}.npy '
    args_to_add += '--t0 1999 '
    args_to_add += '--t1 2001 '
    args_to_add += '--timebatchsize 2 '
    args_to_add += '--particlebatchsize 2 '
    args_to_add += '--val_boxes [] '
    
    os.system("python3 main.py " + args_to_add)
    assert False
