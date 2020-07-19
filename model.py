import numpy as np
import torch
import torch.nn as nn


ACTIVATIONS = {'sigmoid': nn.Sigmoid(),
               'tanh': nn.Tanh(),
               'softsign': nn.Softsign(),
               'relu': nn.ReLU()}


class SimpleCNN(nn.Module):
    def __init__(self, width=16, depth=3, filter_length=512, activation='sigmoid'):
        super(SimpleCNN, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = ACTIVATIONS[activation]
        for d in range(depth):
            channels_in = 1 if d == 0 else width
            channels_out = width
            self.layers.append(nn.ConstantPad1d((filter_length-1, 0), 0.))
            self.layers.append(nn.Conv1d(channels_in, channels_out, kernel_size=filter_length, stride=1))
        # output gain
        self.layers.append(nn.Conv1d(width, 1, kernel_size=1, stride=1, bias=False))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
        x = self.layers[-1](x)
        return x


class WaveNet(nn.Module):
    def __init__(self, activation='softsign', gated_activation='softsign', filter_length=3, width=16):
        super(WaveNet, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = ACTIVATIONS[activation]
        self.gate_activation = ACTIVATIONS[gated_activation]

        self.start_conv = nn.Conv1d(in_channels=1, out_channels=width, kernel_size=1, bias=False)

        self.pads = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.res = nn.ModuleList()
        self.dilations = [2**i for i in range(9)]*2
        for layer_idx, dilate in enumerate(self.dilations):
            self.pads.append(nn.ConstantPad1d((dilate*(filter_length-1), 0), 0.))
            self.convs.append(nn.Conv1d(width, width, kernel_size=filter_length, dilation=dilate, stride=1))
            self.gate_convs.append(nn.Conv1d(width, width, kernel_size=filter_length, dilation=dilate, stride=1))
            if layer_idx < len(self.dilations) - 1:
                self.res.append(nn.Conv1d(width, width, kernel_size=1, stride=1))
        # output mixer
        self.mixer = nn.Conv1d(width*len(self.dilations), 1, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        outputs = []
        x = self.start_conv(x)
        for layer_idx in range(len(self.dilations)):
            y = self.pads[layer_idx](x)
            feature = self.convs[layer_idx](y)
            feature = self.activation(feature)
            gate = self.gate_convs[layer_idx](y)
            gate = self.gate_activation(gate)
            z = torch.mul(feature, gate)
            outputs.append(z)
            if layer_idx < len(self.dilations) - 1:
                scaled = self.res[layer_idx](z)
                x = x+scaled
        outputs = torch.cat(outputs, dim=1)
        return self.mixer(outputs)
