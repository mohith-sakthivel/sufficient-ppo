from typing import Union, List

import torch as torch
import torch.nn as nn


def conv(*args, init_scale: Union[int, float] = 1.0, **kwargs) -> nn.Module:
    """ Conv2D layer with baselines init """
    conv_layer = nn.Conv2d(*args, **kwargs)
    baselines_init(conv_layer, init_scale)
    return conv_layer


def fc(*args, init_scale: Union[float, int] = 1.0, **kwargs) -> nn.Module:
    """ Linear layer with baselines init """
    fc_layer = nn.Linear(*args, **kwargs)
    baselines_init(fc_layer, init_scale)
    return fc_layer


def baselines_init(layer, init_scale: Union[float, int] = 1.0):
    for name, param in layer.named_parameters():
        if name == 'weight':
            nn.init.orthogonal_(param, init_scale)
        elif name == 'bias':
            nn.init.constant_(param, 0.0)
        else:
            raise NotImplementedError


class MLP(nn.Module):

    def __init__(self,
                 input_dim: int,
                 layer_dims: List[int],
                 act_fn: nn.Module = nn.ReLU):

        super(MLP, self).__init__()
        in_d = input_dim
        if len(layer_dims) == 0:
            self._net = nn.Identity()
        else:
            layers = []
            for out_d in layer_dims:
                layers.append(nn.Linear(in_d, out_d))
                layers.append(act_fn())
                in_d = out_d
            self._net = nn.Sequential(*layers)
        self.out_dim = in_d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._net(x)


class LayerNorm(nn.Module):
    
    def __init__(self, epsilon):
        super(LayerNorm, self).__init__()
        self._epsilon = epsilon

    def forward(self, x):
        std, mean = torch.std_mean(x, dim=-1, keepdim=True)
        return (x - mean)/(std + self._epsilon)
