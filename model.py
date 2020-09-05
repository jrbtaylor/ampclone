import numpy as np
import torch
import torch.nn as nn
import torchaudio

from melfilters import gabor
from filterbank import filterbank
from iir_ops import m_lfilter


ACTIVATIONS = {'sigmoid': nn.Sigmoid(),
               'tanh': nn.Tanh(),
               'softsign': nn.Softsign(),
               'relu': nn.ReLU()}


def get_model(model_type, model_kwargs):
    models = {'simple': SimpleCNN,
              'wavnet': WaveNet,
              'fixedfilter': FixedFilters,
              'crossover': Crossover,
              'blender': Blender}
    return models[model_type](**model_kwargs)


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


class MelFilter(nn.Module):
    def __init__(self, n_filters, f_min, f_max, fs, filter_length):
        super(MelFilter, self).__init__()
        filters = np.array(gabor(n_filters, f_min, f_max, fs, filter_length))
        # expand dims so filters is shaped (n_filters, in_channels=1, filter_length)
        filters = np.expand_dims(filters, axis=1)
        self.filters = torch.nn.Parameter(torch.from_numpy(filters).type(torch.get_default_dtype()),
                                          requires_grad=False)
        self.pad = nn.ConstantPad1d((filter_length - 1, 0), 0.)

    def forward(self, x):
        x = self.pad(x)
        return torch.nn.functional.conv1d(x, self.filters)


class IIRFilter(nn.Module):
    def __init__(self, n_filters=30, f_min=40, f_max=16000, fs=44100, filter_length=7):
        super(IIRFilter, self).__init__()
        assert filter_length % 2 == 1
        filter_order = (filter_length-1)//2
        filters = filterbank(n_filters, f_min, f_max, fs, filter_order)
        # self.filter_bs = torch.nn.ParameterList(
        #     [torch.nn.Parameter(torch.from_numpy(f[0]).type(torch.get_default_dtype()), requires_grad=False)
        #      for f in filters])
        # self.filter_as = torch.nn.ParameterList(
        #     [torch.nn.Parameter(torch.from_numpy(f[1]).type(torch.get_default_dtype()),requires_grad=False)
        #      for f in filters])
        self.filter_bs = nn.Parameter(
            torch.stack([torch.from_numpy(f[0]).type(torch.get_default_dtype()) for f in filters]), requires_grad=False)
        self.filter_as = nn.Parameter(
            torch.stack([torch.from_numpy(f[1]).type(torch.get_default_dtype()) for f in filters]), requires_grad=False)

        # note: padding is automatic in torchaudio.functional.lfilter
        # self.pad = nn.ConstantPad1d((filter_length - 1, 0), 0.)

    def forward(self, x):
        # x = self.pad(x)
        # y = []
        # for b, a in zip(self.filter_bs, self.filter_as):
        #     y.append(torchaudio.functional.lfilter(x, a, b))  # , clamp=False)
        # return torch.cat(y, 1)
        y = m_lfilter(x, self.filter_as, self.filter_bs)
        return y


class FixedFilters(nn.Module):
    def __init__(self, width=20, depth=7, filter_length=64, activation='tanh',
                 f_min=40, f_max=10000, fs=44100, bias=True, filter_type='mel'):
        """
        Notes:  - bias is significantly better than no bias
                - filter lengths 64 and 128 is noticeably better than 128, 256 or 32 (at 22.1 kHz fs)
        """
        super(FixedFilters, self).__init__()
        self.filter_type = filter_type
        if filter_type == 'mel':
            self.filters = MelFilter(width, f_min, f_max, fs, filter_length)
        else:  # IIR filter
            self.filters = nn.ModuleList([IIRFilter(width // 2, f_min, f_max, fs, filter_length),
                                          IIRFilter(width, f_min, f_max, fs, filter_length)])
        self.recombines = nn.ModuleList()
        self.activation = ACTIVATIONS[activation]
        self.depth = depth

        for d in range(depth):
            if self.filter_type == 'mel':
                conv_layer = torch.nn.Conv1d(width, 1, kernel_size=1, stride=1, bias=bias)
            else:
                conv_layer = torch.nn.Conv1d(width // 2 if d % 2 == 0 else width, 1, kernel_size=1, stride=1, bias=bias)
            conv_layer.weight.data.fill_(1.)
            if bias:
                conv_layer.bias.data.fill_(0.)
            self.recombines.append(conv_layer)
        # output gain
        self.output_gain = nn.Conv1d(1, 1, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        for idx in range(self.depth):
            if self.filter_type == 'mel':
                x = self.filters(x)
            else:
                x = self.filters[idx % len(self.filters)](x)
            x = self.recombines[idx](x)
            x = self.activation(x)
        x = self.output_gain(x)
        return x


class Blender(nn.Module):
    def __init__(self, width=20, depth=7, filter_length=64, activation='tanh',
                 f_min=40, f_max=20000, fs=44100, bias=True, filter_type='mel'):
        """
        Similar to FixedFilters model but each layer has a learned "blend" parameter that blends its input and output
        """
        super(Blender, self).__init__()
        self.filter_type = filter_type
        if filter_type == 'mel':
            self.filters = MelFilter(width, f_min, f_max, fs, filter_length)
        else:  # IIR filter
            self.filters = nn.ModuleList([IIRFilter(width // 2, f_min, f_max, fs, filter_length),
                                          IIRFilter(width, f_min, f_max, fs, filter_length)])
        self.recombines = nn.ModuleList()
        self.activation = ACTIVATIONS[activation]
        self.depth = depth
        self.blends = torch.nn.ParameterList()
        self.blend_sigmoid = torch.nn.Sigmoid()

        for d in range(depth):
            if self.filter_type == 'mel':
                conv_layer = torch.nn.Conv1d(width, 1, kernel_size=1, stride=1, bias=bias)
            else:
                conv_layer = torch.nn.Conv1d(width // 2 if d % 2 == 0 else width, 1, kernel_size=1, stride=1, bias=bias)
            conv_layer.weight.data.fill_(1.)
            if bias:
                conv_layer.bias.data.fill_(0.)
            self.recombines.append(conv_layer)
            self.blends.append(
                torch.nn.Parameter(torch.from_numpy(np.array(1.)).type(torch.get_default_dtype()),
                                   requires_grad=True))
        # output gain
        self.output_gain = nn.Conv1d(1, 1, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        for idx in range(self.depth):
            if self.filter_type == 'mel':
                y = self.filters(x)
            else:
                y = self.filters[idx % len(self.filters)](x)
            y = self.recombines[idx](y)
            y = self.activation(y)
            blend = self.blend_sigmoid(self.blends[idx])
            x = (1-blend)*x+blend*y
        x = self.output_gain(x)
        return x


class Crossover(nn.Module):
    def __init__(self, width=20, depth=7, filter_length=64, activation='tanh',
                 f_min=40, f_max=10000, fs=44100, bias=True):
        super(Crossover, self).__init__()
        self.filters = MelFilter(width, f_min, f_max, fs, filter_length)
        self.recombines = nn.ModuleList()
        self.eqs = nn.ModuleList()
        self.activation = ACTIVATIONS[activation]
        self.biases = []
        self.gains = []
        self.depth = depth

        for d in range(depth):
            self.biases.append(nn.Parameter(torch.tensor([0.], requires_grad=True)))
            self.gains.append(nn.Parameter(torch.tensor([1.], requires_grad=True)))

            conv_layer = torch.nn.Conv1d(width, 1, kernel_size=1, stride=1, bias=bias)
            conv_layer.weight.data.fill_(1.)
            if bias:
                conv_layer.bias.data.fill_(0.)
            self.recombines.append(conv_layer)

            conv_layer = torch.nn.Conv1d(width, 1, kernel_size=1, stride=1, bias=bias)
            conv_layer.weight.data.fill_(1.)
            if bias:
                conv_layer.bias.data.fill_(0.)
            self.eqs.append(conv_layer)
        # output gain
        self.output_gain = nn.Conv1d(1, 1, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        for idx in range(self.depth):
            x_pos = nn.ReLU()(x)
            x_neg = nn.ReLU()(-x)
            x_pos = self.filters(x_pos)
            x_neg = self.filters(x_neg)
            x_pos = self.recombines[idx](x_pos)
            x_neg = self.recombines[idx](x_neg)
            x_pos = nn.Sigmoid()(x_pos)
            x_neg = nn.Sigmoid()(x_neg)
            x_pos = x_pos+self.biases[idx]
            x_neg = x_neg+self.biases[idx]

            x = x_pos-x_neg
            x = self.filters(x)
            x = self.eqs[idx](x)
            x = self.activation(x)
        x = self.output_gain(x)
        return x
