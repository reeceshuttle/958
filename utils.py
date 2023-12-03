import torch
import numpy.linalg as linalg
from scipy.stats import entropy
import numpy as np

def calc_entropy(M: torch.Tensor):
    """
    Calculates the row-wise entropy of a matrix.
    """
    raise NotImplementedError('Implement me!')

def calc_stable_rank(M: torch.Tensor) -> float:
    """
    Takes a matrix M and calculates the stable rank of it:
    stable_rank = (sv_1+...+sv_n)/sv_1,
    where sv_1 is the biggest singular value of matrix M.
    ref: https://www.cs.ubc.ca/~nickhar/W12/Lecture15Notes.pdf
    """
    S = linalg.svd(M.detach(), compute_uv=False)
    max_sv = np.max(S)
    stable_rank = sum([val**2 for val in S])/(max_sv**2)
    return stable_rank, [float(val) for val in S] # to avoid float32 error when storing data