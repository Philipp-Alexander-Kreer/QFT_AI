import numpy as np
import torch as t

from experimental_setup import init_data
from itertools import combinations
from scipy import integrate


def K_Gauss(x1, x2, sig_b=1, sig_w=1):
    """Two point function Gaussian activation function Eq. (2.31)"""
    d_in = 1
    return sig_b ** 2 + sig_w ** 2 * np.exp(-sig_w ** 2 * np.abs(x1 - x2) ** 2 / 2 / d_in)


def K_ReLU(x1, x2, sig_b=0, sig_w=1):
    """Two point function ReLU activation function Eq. (2.28)"""

    dim_in = 1

    sqrt = np.sqrt((sig_b ** 2 + sig_w ** 2 / dim_in * x1 * x1) * (sig_b ** 2 + sig_w ** 2 / dim_in * x2 * x2))

    # theta = np.arccos( (sig_b**2 + sig_w**2/dim_in * x1*x2) / sqrt)
    # becasue sig_b = 0 in experiment
    theta = 0

    y = sig_b ** 2 + sig_w ** 2 / 2 / np.pi * sqrt * (np.sin(theta) + (np.pi - theta) * np.cos(theta))

    return y


def two_pt_func(K, X):
    """Computes free two point function for given Kernel K and input vector X"""
    return K(X[0], X[1])


def four_pt_func(K, X):
    """Computes free four point function for given Kernel K and input vector X"""

    WickContractions2 = (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2)))

    return sum([K(X[tup[0][0]], X[tup[0][1]]) * K(X[tup[1][0]], X[tup[1][1]]) for tup in WickContractions2])


def six_pt_func(K, X):
    """Computes free six point function for given Kernel K and input vector X"""

    WickContractions3 = (
        ((0, 1), (2, 3), (4, 5)), ((0, 1), (3, 4), (2, 5)), ((0, 1), (3, 5), (2, 4)),
        ((0, 2), (1, 3), (4, 5)), ((0, 2), (1, 4), (3, 5)), ((0, 2), (1, 5), (3, 4)),
        ((0, 3), (1, 2), (4, 5)), ((0, 3), (1, 4), (2, 5)), ((0, 3), (1, 5), (2, 4)),
        ((0, 4), (1, 2), (3, 5)), ((0, 4), (1, 3), (2, 5)), ((0, 4), (1, 5), (2, 5)),
        ((0, 5), (1, 2), (3, 4)), ((0, 5), (1, 3), (2, 4)), ((0, 5), (1, 4), (2, 3))
    )

    return sum([K(X[tup[0][0]], X[tup[0][1]]) * K(X[tup[1][0]], X[tup[1][1]]) * K(X[tup[2][0]], X[tup[2][1]])
                for tup in WickContractions3])


def th_n_pt_free(data, n, K):
    """Theory prediction of n-pt function in the free theory"""

    d_in = data.shape[0]

    if not (n in [2, 4, 6]):
        raise ValueError("n can only be [2,4,6]")
    if n > d_in:
        raise ValueError("d_in must be larger than or equal to n")

    # Generate combinations of indices for n-pt functions
    tupl = list(combinations(range(d_in), n))

    if n == 2:
        return [two_pt_func(K, data[list(tup)]) for tup in tupl]

    if n == 4:
        return [four_pt_func(K, data[list(tup)]) for tup in tupl]

    if n == 6:
        return [six_pt_func(K, data[list(tup)]) for tup in tupl]

    return 0

def th_n_free(Relu=True):
    """Theory prediction of n-pt function in the free theory"""

    X = init_data(Relu)
    K = init_kernel(Relu)

    return th_n_pt_free(X, 6, K)[0]


def init_kernel(ReLU=True):
    """Initializes Kernel K for ReLU Network (ReLU=True) or Gauss Network."""
    if ReLU:
        return K_ReLU  # lambda x1, x2: K_ReLU(x1, x2, 0, 1)
    return K_Gauss


def norm_int( X, K_w, cut_off):
    return integrate.quad(lambda y: (K_w(X[0], y) * K_w(X[1], y) * K_w(X[2], y) * K_w(X[3], y)).item(), -cut_off,
                          cut_off)


def theo_int(X, K_w, cut_off):
    """Computes the integral in Eq. 3.32"""

    # X is the input vector, K_w is the kernel, cut_off is the cut off of the integral
    d_in = X.shape[0]
    tupl = list(combinations(range(d_in), 4))

    # See Eq. 3.32
    return t.tensor([24 * norm_int( X[list(tup)], K_w, cut_off)[0] for tup in tupl], dtype=t.float32)


def match_lambda(exp_4pt, cut_off, ReLU=True):
    """Computes the matching factor lambda in Eq. 3.32"""
    X = init_data(ReLU)
    K = init_kernel(ReLU)

    K_w = K

    if not (ReLU):
        K_w = K - 1

    theo_1 = t.stack(th_n_pt_free(X, 4, K)).view(-1)
    theo_2 = theo_int(X, K_w, cut_off)


    return (theo_1 - exp_4pt) / theo_2


def theo_6_pt_lambda(cut_off, lambda_N, ReLU=True):
    """Predicts the 6-pt function in the interacting theory, see Eq. 3.37"""

    X = init_data(ReLU)
    K = init_kernel(ReLU)

    K_w = K

    if not(ReLU):
        K_w = K - 1

    theo_1 = t.stack(th_n_pt_free(X, 6, K)).view(-1)

    # X is the input vector, K_w is the kernel, cut_off is the cut off of the integral
    d_in = X.shape[0]
    tupl = list(combinations(range(d_in), 4))

    #complement of tupl
    tupl_c = [tuple(set(range(d_in)) - set(tup)) for tup in tupl]
    coeff = t.tensor([K(X[tup[0]], X[tup[1]]).item() for tup in tupl_c], dtype=t.float32)


    th_6pt = theo_1 + t.sum((theo_int( X, K_w, cut_off)*lambda_N)*coeff)

    return th_6pt

