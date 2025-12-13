import os
import pickle

def read_model(out_dir):
    """
    This function reads in models from a designation directory

    Parameters
    ----------
    out_dir: directory where model are stored

    Returns:
        two models needed for testing

    Examples
    --------
    >>> lr_pipe, linear_svc_pipe = read_model(out_dir)
    """

    lr_pipe_path = os.path.join(out_dir, "trained_lr_pipe.pkl")
    linear_svc_pipe_path = os.path.join(out_dir, "trained_linear_svc_pipe.pkl")

    with open(lr_pipe_path, "rb") as f:
        lr_pipe = pickle.load(f)

    with open(linear_svc_pipe_path, "rb") as f:
        linear_svc_pipe = pickle.load(f)

    return lr_pipe, linear_svc_pipe
