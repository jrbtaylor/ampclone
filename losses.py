import numpy as np
import torch.nn as nn
import torch
import torchaudio


from utils import RATE


# def get_fft_loss(pred, label):
#     loss = 0
#     n_ffts = [1024, 4096, 4096 * 4]
#     for n_fft in n_ffts:
#         spectro = torchaudio.transforms.Spectrogram(n_fft=n_fft)
#         pred_spect = spectro(pred)
#         label_spect = spectro(label)
#         loss += nn.L1Loss(reduction='mean')(torch.log10(pred_spect), torch.log10(label_spect))
#     loss /= len(n_ffts)
#     return loss
def get_fft_loss(pred, label):
    loss = 0
    n_ffts = [4096 * 2, 4096 * 4]
    for n_fft in n_ffts:
        # spectro = torchaudio.transforms.Spectrogram(n_fft=n_fft)
        spectro = torchaudio.transforms.MelSpectrogram(sample_rate=RATE, n_fft=n_fft)
        pred_spect = spectro(pred)
        label_spect = spectro(label)
        loss += nn.L1Loss(reduction='mean')(torch.log10(pred_spect), torch.log10(label_spect))
    loss /= len(n_ffts)
    return loss


def get_ssim_loss(pred, label):
    # assumes means are zero, reduces SSIM equation accordingly
    cov = torch.mean(pred * label) - torch.mean(pred) * torch.mean(label)
    c2 = 0.03
    ssim = (2 * cov + c2) / (torch.var(pred) + torch.var(label) + c2)
    return (1 - ssim) / 2


class Loss(nn.Module):
    def __init__(self, mse_weight=0.1, std_weight=0.1, fft_weight=0.1, ssim_weight=1.):
        """
        MSE (may need to correct for latency later)
        Difference in std
        Spectral difference
        """
        super(Loss, self).__init__()
        self.mse_weight = mse_weight
        self.std_weight = std_weight
        self.fft_weight = fft_weight
        self.ssim_weight = ssim_weight

    def forward(self, pred, label):
        """
        :param pred: model output (batch, 1, length)
        :param label: groundtruth (batch, 1, length)
        :return:
        """
        mse_loss = nn.MSELoss(reduction='mean')(pred, label)

        std_loss = torch.square(torch.std_mean(pred)[0] - torch.std_mean(label)[0])

        # TODO: add frequency weighting based on perceptual sensitivity (https://en.wikipedia.org/wiki/Mel_scale)
        fft_loss = get_fft_loss(pred, label)

        ssim_loss = get_ssim_loss(pred, label)

        total_loss = self.mse_weight * mse_loss + self.std_weight * std_loss + self.fft_weight * fft_loss \
                     + self.ssim_weight * ssim_loss

        return total_loss
