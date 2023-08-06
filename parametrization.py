import torch
from torch.nn.utils.parametrize import register_parametrization
from numpy import prod


def parameters_in_linear(input_size: int, output_size: int):
    return input_size * output_size + output_size


def get_hidden_layer_size(input_size: int, output_size: int, target_n_parameters: int):
    # here to propose the largest layer that would fit with target parameters given input and output sizes
    return (target_n_parameters - output_size) // (input_size + output_size + 1)


def get_hidden_units_structure(input_size: int, output_size: int, parameters_target: int,
                               n_hidden=5, max_hidden_layer_size: int = 99999999):
    units_list = [input_size]
    single_interm = get_hidden_layer_size(input_size, output_size, parameters_target)
    if single_interm <= 400:
        units_list.append(single_interm)
        return units_list

    first_layer_params = parameters_in_linear(input_size, 100)
    last_layer_params = parameters_in_linear(100, output_size)
    remaining_parameters = parameters_target - first_layer_params - last_layer_params

    units_per_layer = int(max(
        min(
            [(remaining_parameters / n_hidden) ** 0.5, max_hidden_layer_size]
        ), output_size)
    )

    units_list.extend(units_per_layer for _ in range(n_hidden))
    units_list.append(100)

    return units_list


class MLP(torch.nn.Module):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 parameters_target: int,
                 max_hidden_layer_units: int,
                 n_hidden_layers: int,
                 ):
        super().__init__()
        # first layer starts from 2
        # last layer ends in output_size
        # each layer has in * out + out parameters
        units_list = get_hidden_units_structure(input_size, output_size, parameters_target,
                                                n_hidden_layers, max_hidden_layer_units)

        self.net = torch.nn.Sequential()
        self.net.add_module('BN1', torch.nn.BatchNorm1d(units_list[0]))
        for i, k in enumerate(units_list[1:]):
            # i marks the previous element, k is the number of units for the current one
            self.net.add_module(f'linear{i}', torch.nn.Linear(units_list[i], k))
            self.net.add_module(f'activation{i}', torch.nn.ReLU(inplace=True))
        self.net.add_module('BN2', torch.nn.BatchNorm1d(units_list[-1]))
        self.net.add_module('output', torch.nn.Linear(units_list[-1], output_size))

    def forward(self, x):
        return self.net(x)


def get_n_parameters(convolution):
    return sum(p.numel() for p in convolution.parameters())


class MLP_parametrization(torch.nn.Module):
    def __init__(self,
                 convolution: torch.nn.Conv2d,
                 max_hidden_units_per_mlp_layer: int = 250,
                 n_hidden_layers: int = 5,
                 ):
        super().__init__()
        self.kernel_shape = list(convolution.weight.shape[2:])
        weight = convolution.weight
        conv_parameters_total = get_n_parameters(convolution)
        out_mlp_layer_size = int(prod(self.kernel_shape))
        self.MLP = MLP(2,
                       out_mlp_layer_size,
                       conv_parameters_total,
                       max_hidden_units_per_mlp_layer,
                       n_hidden_layers)

        n_1 = weight.shape[0]
        n_2 = weight.shape[1]
        x = torch.linspace(-1, 1, n_1)
        y = torch.linspace(-1, 1, n_2)
        meshgrid = torch.stack(torch.meshgrid(x, y, indexing='ij'), -1)
        self.register_buffer('meshgrid', meshgrid)
        # meshgrid.requires_grad = False

    def forward(self, _):
        return self.MLP(self.meshgrid.reshape([self.meshgrid.shape[0]* self.meshgrid.shape[1], 2])).reshape(
            [self.meshgrid.shape[0], self.meshgrid.shape[1]] + self.kernel_shape
        )
    #
    # def right_inverse(self, weight):
    #     return weight.ravel()[0]


def parametrize_cnn(module, max_units_per_mlp_layer = 250, n_hidden_layers=5):
    if isinstance(module, torch.nn.Conv2d):
        register_parametrization(module, 'weight', MLP_parametrization(module,
                                                                       max_units_per_mlp_layer,
                                                                       n_hidden_layers), unsafe=True)


if __name__ == '__main__':
    from torchvision.models import resnet18

    resnet = resnet18()
    conv = resnet.conv1
    parametrize_cnn(conv)
    print('New weights', conv.weight[0, 0, 0])
    print('Old weights', conv.parametrizations.weight.original[0, 0, 0])
    print(f'Requires grad = {conv.weight.requires_grad}')
    print('conv.parametrizations', conv.parametrizations)
