
import torch as t
import torch.nn as nn
# import matplotlib.pyplot as plt
import numpy as np
# Define the neural networks for the experiments


class ReLU_Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ReLU_Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

        # Initialize weights with a normal distribution (mean=0, std=0.01)
        nn.init.normal_(self.fc1.weight, mean=0, std=1)
        nn.init.normal_(self.fc2.weight, mean=0, std=1 / np.sqrt(hidden_size))

        # ReLU has bias 0
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class Gauss_Activation(nn.Module):
    def __init__(self, sig_b, sig_W):
        super(Gauss_Activation, self).__init__()

        self.sig_b = sig_b
        self.sig_W = sig_W
        self.din = 1

    def forward(self, x, z):
        numerator = t.exp(z)
        denominator = t.sqrt(t.exp(2 * (self.sig_b ** 2 + self.sig_W / self.din * x ** 2)))
        return numerator / denominator


class Gauss_Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Gauss_Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.gauss = Gauss_Activation(1, 1)
        self.fc2 = nn.Linear(hidden_size, output_size)

        # Initialize weights with a normal distribution (mean=0, std=0.01)
        nn.init.normal_(self.fc1.weight, mean=0, std=1)
        nn.init.normal_(self.fc2.weight, mean=0, std=1 / np.sqrt(hidden_size))

        # Initialize biases with normal distribution
        nn.init.normal_(self.fc1.bias, mean=0, std=1)
        nn.init.normal_(self.fc2.bias, mean=0, std=1)

    def forward(self, x):
        z = self.fc1(x)
        x = self.gauss(x, z)
        x = self.fc2(x)
        return x


def init_data(ReLU=True):
    """Initializes data for ReLU Network (ReLU=True) or Gauss Network. See Tab. 2"""

    if ReLU:
        return t.tensor([[0.2], [0.4], [0.6], [0.8], [1.0], [1.2]], dtype=t.float32)

    return t.tensor([[-0.01], [-0.006], [-0.002], [0.002], [0.006], [0.01]], dtype=t.float32)


def init_model(input_size, hidden_size, output_size, ReLU=True):
    """Initializes ReLU_Net (ReLU=True) or Gauss_Net model."""
    if ReLU:
        return ReLU_Net(input_size, hidden_size, output_size)

    return Gauss_Net(input_size, hidden_size, output_size)


def generate_data(input_size, hidden_size, output_size, n_net, n_exp, ReLU=True):
    """Create experimental values for network outputs"""

    input_data = init_data(ReLU)
    exp_data = []

    for i in range(n_exp):

        print(f"Run experiment {i + 1} / {n_exp}")
        exp_run = []

        for _ in range(n_net):
            model = init_model(input_size, hidden_size, output_size, ReLU)
            exp_run.append(model(input_data))

        exp_data.append(t.stack(exp_run, dim=1))

    exp_data = t.squeeze(t.transpose(t.stack(exp_data, dim=0), 1, 2), -1)

    # Specify the file path
    file_path = 'data/exp_data_' + str(hidden_size) + '_ReLU_' + str(ReLU) + '.txt'
    print("Save experimental results in: ", file_path)

    # Save the tensors to the file
    t.save(exp_data, file_path)

    return 0


