import numpy as np
import torch.nn as nn
import torch

from utils import RATE


class OffsetMSE(nn.Module):
    def __init__(self):
        """
        Calculates the MSE over a range of offsets between the signals, since latency is unknown and not optimized
        Takes the minimum loss over the offsets
        Note: 3ms at 32kHz sampling is 64 samples, seems reasonable
        """
        super(OffsetMSE, self).__init__()
        self.maxshift = int(RATE*3*1e-3)

    def forward(self, predict, target):
        """
        MSE at the estimated latency value
        :param predict: model output (batch, 1, length)
        :param target: groundtruth (batch, 1, length)
        :return:
        """
        losses = []
        shifts = range(1, self.maxshift)
        for shift in shifts:
            losses.append(nn.MSELoss(reduction='mean')(predict[:, 0, shift:], target[:, 0, :-shift]))
        losses = torch.stack(losses, dim=-1)
        return torch.min(losses)
