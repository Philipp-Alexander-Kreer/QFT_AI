import torch as t
import matplotlib.pyplot as plt
import numpy as np

from itertools import combinations

import experimental_setup
import theory_prediction


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


def average_nets(n_point_fcts):
    """Averages n-pt functions over different networks"""

    # Average over all networks
    return t.mean(n_point_fcts, dim=2)


def average_experiments(avg_nets):
    """Averages n-pt functions over different experiments"""

    # Average over all experiments
    return t.mean(avg_nets, dim=1)


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


def experiment_2b(data_id, ReLU=True):
    """Implementation of experiment 2b. Equivalent to Fig. 5 right."""

    _, avg_exp_4, avg_exp_6 = average_n_pt_from_data(data_id, ReLU=ReLU)

    th_free = theory_prediction.th_n_free(Relu=True)

    # various cut_offs from 2 to 100000 in log scale
    cut_offs = np.logspace(1, 6, 20)

    th_lambda_pred = []

    for cut_off in cut_offs:
        # match lambda for cut_off
        lambda_n = -t.mean(theory_prediction.match_lambda(avg_exp_4, cut_off))

        # theory prediction for matched lambda
        th_lambda_pred.append(theory_prediction.theo_6_pt_lambda(cut_off, lambda_n, ReLU=ReLU).detach().numpy())

    # convert list to numpy array for plotting
    avg_exp_6 = avg_exp_6.detach().numpy()
    th_lambda_pred = np.array(th_lambda_pred) / avg_exp_6
    th_free_pl = np.full_like(cut_offs, th_free.detach().numpy() / avg_exp_6)

    plot_lambda_prediction(cut_offs, th_lambda_pred, th_free_pl)

    return 0


def measure_lambda(data_id, ReLU=True):
    """Match lambda for data set with id data_id"""
    _, avg_exp_4, _ = average_n_pt_from_data(data_id, ReLU=ReLU)

    # match lambda
    lambda_n = -t.mean(theory_prediction.match_lambda(avg_exp_4, 100), dim=0)

    # return lambda as numpy array
    return lambda_n.detach().numpy()


def experiment_2a(hidden_layer_sizes, ReLU=True):
    """Implementation of experiment 2a. Equivalent to Fig. 5 left."""

    # Measure lambdas for different hidden sizes
    measured_lambdas = [measure_lambda(hidden_size) for hidden_size in hidden_layer_sizes]

    # Plot measured lambdas for hidden sizes
    plt.plot(hidden_layer_sizes, measured_lambdas, 'o')
    plt.xlabel('Hidden size')
    plt.ylabel('Match lambda')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

    return 0

def experiment_1(hidden_layer_sizes, ReLU=True):
    """Implementation of experiment 1. Equivalent to Fig. 2."""

    # measure 2-pt function
    avg_exp_2 = [average_n_pt_from_data(layer_size, ReLU=ReLU)[0] for layer_size in hidden_layer_sizes]

    # init data and kernel
    X = experimental_setup.init_data(ReLU)
    K = theory_prediction.init_kernel(ReLU)

    # free theory prediction for 2-pt function
    th_2pt = theory_prediction.th_n_pt_free(X, 2, K)

    # turn list of tensors into one tensor
    th_2pt = t.stack(th_2pt).view(-1)

    deviation = [t.mean((th_2pt - avg_2pt) / th_2pt).item() for avg_2pt in avg_exp_2]


    # Plot normalized deviation from free theory prediction
    plt.plot(hidden_layer_sizes, deviation)
    plt.xlabel('Absolute deviation from free theory prediction')
    plt.ylabel('Number of width of hidden layer')
    plt.xscale('log')
    #plt.yscale('log')
    plt.show()

    return 0


if __name__ == '__main__':


    hidden_sizes = [2, 3, 4, 5, 10, 20, 50, 100]#, 1000]
    experimental_setup.generate_data(1, 1000, 1, 100000, 10, True)
    # Generate data for hidden sizes
    # [experimental_setup.generate_data(1, hidden_size, 1, 100000, 100, True) for hidden_size in hidden_sizes]

    experiment_1(hidden_sizes, ReLU=True)

    # experiment_2a(hidden_sizes, ReLU=True)

    # experiment_2b(20, ReLU=True)
