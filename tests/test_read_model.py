import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.functions.read_model import read_model

def test_read_model():
    lr_pipe, linear_svc_pipe = read_model("results/models")
    assert lr_pipe is not None, "lr_pipe was not read correctly"
    assert linear_svc_pipe is not None, "linear_svc_pipe was not read correctly"
