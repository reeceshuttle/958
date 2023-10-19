import torch
import numpy.linalg as linalg
from scipy.stats import entropy

def calc_entropy(M: torch.Tensor):
    """
    Calculates the row-wise entropy of a matrix.
    """
    raise NotImplementedError('Implement me!')

def calc_presoftmax_entropy(A: torch.Tensor):
    """
    Calculates the row-wise entropy before softmax.
    A is the matrix of activation scores before softmax'd.
    A = X(W_q)(W_k^T)X^T
    1. We apply probability dist: P_i = |a_{1i}|/\sum^n_{j=1}|a_{1j}|
    2. using these probabilities, we calculate entropy.
    note that doing the scaling here doesnt matter with the probability dist
    we use since it factors out.
    """
    A_ = torch.abs(A.detach())
    P = A_/torch.sum(A_, axis=1, keepdims=True)
    return entropy(P, axis=1)

def calc_stable_rank(M: torch.Tensor) -> float:
    """
    Takes a matrix M and calculates the stable rank of it:
    stable_rank = (sv_1+...+sv_n)/sv_1,
    where sv_1 is the biggest singular value of matrix M.
    ref: https://www.cs.ubc.ca/~nickhar/W12/Lecture15Notes.pdf
    """
    _, S, _ = linalg.svd(M.detach()) # U, S, V
    stable_rank = sum([val**2 for val in S])/S[0]**2
    return stable_rank