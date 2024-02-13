import torch as t
import matplotlib.pyplot as plt
import numpy as np

# custom imports
import experimental_setup
import theory_prediction
import data_analysis


def experiment_1(hidden_layer_sizes, ReLU=True):
    """Implementation of experiment 1. Equivalent to Fig. 2."""

    # measure 2-pt function
    avg_exp_2 = [data_analysis.average_n_pt_from_data(layer_size, ReLU=ReLU)[0] for layer_size in hidden_layer_sizes]

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


def experiment_2a(hidden_layer_sizes, ReLU=True):
    """Implementation of experiment 2a. Equivalent to Fig. 5 left."""

    # Measure lambdas for different hidden sizes
    measured_lambdas = [data_analysis.measure_lambda(hidden_size) for hidden_size in hidden_layer_sizes]

    # Plot measured lambdas for hidden sizes
    plt.plot(hidden_layer_sizes, measured_lambdas, 'o')
    plt.xlabel('Hidden size')
    plt.ylabel('Match lambda')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

    return 0


def experiment_2b(data_id, ReLU=True):
    """Implementation of experiment 2b. Equivalent to Fig. 5 right."""

    _, avg_exp_4, avg_exp_6 = data_analysis.average_n_pt_from_data(data_id, ReLU=ReLU)

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

    data_analysis.plot_lambda_prediction(cut_offs, th_lambda_pred, th_free_pl)

    return 0


if __name__ == '__main__':

    hidden_sizes = [2, 3, 4, 5, 10, 20, 50, 100, 1000]
    # experimental_setup.generate_data(1, 1000, 1, 100000, 10, True)
    # Generate data for hidden sizes
    # [experimental_setup.generate_data(1, hidden_size, 1, 100000, 100, True) for hidden_size in hidden_sizes]

    # experiment_1(hidden_sizes, ReLU=True)

    # experiment_2a(hidden_sizes, ReLU=True)

    # experiment_2b(20, ReLU=True)
