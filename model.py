import numpy as np
import torch
import torch.nn as nn


ACTIVATIONS = {'sigmoid': nn.Sigmoid(),
               'tanh': nn.Tanh()}


class SimpleCNN(nn.Module):
    def __init__(self, width=16, depth=3, filter_length=512, activation='sigmoid'):
        super(SimpleCNN, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = ACTIVATIONS[activation]
        for d in range(depth):
            channels_in = 1 if d == 0 else width
            channels_out = 1 if d == depth-1 else width
            self.layers.append(nn.ConstantPad1d((filter_length-1, 0), 0.))
            self.layers.append(nn.Conv1d(channels_in, channels_out, kernel_size=filter_length, stride=1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
        return x

