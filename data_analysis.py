import numpy as np
import torch as t
import matplotlib.pyplot as plt
from itertools import combinations
import theory_prediction


def average_nets(n_point_fcts):
    """Averages n-pt functions over different networks"""

    # Average over all networks
    return t.mean(n_point_fcts, dim=2)


def average_experiments(avg_nets):
    """Averages n-pt functions over different experiments"""

    # Average over all experiments
    return t.mean(avg_nets, dim=1)


def exp_n_point(data, n):
    """Measures n-pt functions experimentally"""

    # n_exp: number of experiments, n_net: number of networks, d_in: number of input data points
    (n_exp, n_net, d_in) = data.shape

    if n % 2 != 0:
        raise ValueError("Input must be an even number")
    if n > d_in:
        raise ValueError("d_in must be larger than or equal to n")

    # Generate combinations of indices for n-pt functions
    tupl = list(combinations(range(d_in), n))

    # Use torch.prod for element-wise multiplication along the third dimension
    npt = t.stack([t.prod(data[:, :, list(tup)], dim=2) for tup in tupl])

    # shape of npt: (n_combinations, n_exp, n_net)
    return npt


def average_n_pt_from_data(data_id, ReLU=True):
    """Averages n-pt functions for data set with id data_id"""
    data = t.load(f'data/exp_data_{data_id}_ReLU_{ReLU}.txt')

    exp_value = exp_n_point(data, 2)
    avg_exp_2 = average_experiments(average_nets(exp_value))

    exp_value = exp_n_point(data, 4)
    avg_exp_4 = average_experiments(average_nets(exp_value))

    exp_value = exp_n_point(data, 6)
    avg_exp_6 = average_experiments(average_nets(exp_value))

    return avg_exp_2, avg_exp_4, avg_exp_6


def measure_lambda(data_id, ReLU=True):
    """Match lambda for data set with id data_id"""
    _, avg_exp_4, _ = average_n_pt_from_data(data_id, ReLU=ReLU)

    # match lambda
    lambda_n = -t.mean(theory_prediction.match_lambda(avg_exp_4, 100), dim=0)

    # return lambda as numpy array
    return lambda_n.detach().numpy()


def plot_lambda_prediction(cut_offs, th_lambda_pred, th_free_pl):
    """Plots 6pt function prediction in comparison with experimental measurement  for different cut-offs."""

    # Reference line for G6_th/G6_exp = 1 (perfect prediction)
    reference_line = np.full_like(cut_offs, 1)

    # Plot results
    plt.plot(cut_offs, th_free_pl, label='free theory')
    plt.plot(cut_offs, th_lambda_pred, label='lambda prediction')
    plt.plot(cut_offs, reference_line, linestyle='--')

    plt.xlabel('Cut off')
    plt.ylabel('G6_th/G6_exp')
    plt.xscale('log')
    plt.ylim(0, 1.1)
    plt.legend()

    plt.show()

    return None
