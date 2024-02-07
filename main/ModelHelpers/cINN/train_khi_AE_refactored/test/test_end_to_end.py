import pytest
from encoder_decoder import Encoder, MLPDecoder
from unittest.mock import patch, Mock
import os 
import torch

args_to_add = '--pathpattern1 data/ '
args_to_add += '--pathpattern2 data/ '



def test_main_e2d():
    os.system("python3 main.py " + args_to_add)
    assert False
