## test_read_model.py
## author: Ian Gault
## 2025-12-11

# This script tests the read_model function used in the \
# 18_model_testing.py script.

# Usage: pytest from command line root

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.functions.read_model import read_model

def test_read_model():
    '''
    Docstring for test_read_model
    This function confirms that our models were read in correctly before testing their accuracy.
    '''
    lr_pipe, linear_svc_pipe = read_model("results/models")
    assert lr_pipe is not None, "lr_pipe was not read correctly"
    assert linear_svc_pipe is not None, "linear_svc_pipe was not read correctly"
