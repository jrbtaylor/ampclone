import numpy as np
import torch.nn as nn
import torch
import torchaudio


from utils import RATE


def _ssim_loss(x, y):
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    var_x = torch.var(x)
    var_y = torch.var(y)
    cov = torch.mean(x*y)-mean_x*mean_y
    c1, c2 = 0.01**2, 0.03**2
    ssim = (2*mean_x*mean_y+c1)*(2*cov+c2)/(torch.square(mean_x)+torch.square(mean_y)+c1)/(var_x+var_y+c2)
    return (1-ssim)/2


def _spectral_loss(diff, spec, n_ffts):
    if not isinstance(n_ffts, list):
        n_ffts = [n_ffts]

    def _loss_fn(pred, label):
        loss = 0
        for n_fft in n_ffts:
            # TODO: add frequency weighting based on perceptual sensitivity (https://en.wikipedia.org/wiki/Mel_scale)
            if spec == 'mel':
                spec_fn = torchaudio.transforms.MelSpectrogram(sample_rate=RATE, n_fft=n_fft)
            else:
                spec_fn = torchaudio.transforms.Spectrogram(n_fft=n_fft)
            pred_spec = torch.log10(spec_fn(pred))
            label_spec = torch.log10(spec_fn(label))
            if diff == 'l1':
                loss += nn.L1Loss(reduction='mean')(pred_spec, label_spec)
            elif diff == 'l2':
                loss += nn.MSELoss(reduction='mean')(pred_spec, label_spec)
            elif diff == 'ssim':
                loss += _ssim_loss(pred_spec, label_spec)
        return loss/len(n_ffts)
    return _loss_fn


def get_l1_mel_loss(n_ffts):
    return _spectral_loss('l1', 'mel', n_ffts)


def get_l2_mel_loss(n_ffts):
    return _spectral_loss('l2', 'mel', n_ffts)


def get_l1_fft_loss(n_ffts):
    return _spectral_loss('l1', 'fft', n_ffts)


def get_l2_fft_loss(n_ffts):
    return _spectral_loss('l2', 'fft', n_ffts)


def get_ssim_time_loss():
    # Time-domain SSIM loss
    return _ssim_loss


def get_ssim_mel_loss(n_ffts):
    return _spectral_loss('ssim', 'mel', n_ffts)


def get_mse_time_loss():
    return torch.nn.MSELoss(reduction='mean')


class Loss(nn.Module):
    def __init__(self, loss_type, loss_kwargs):
        super(Loss, self).__init__()
        loss_dict = {'l2': get_mse_time_loss,
                     'ssim_time': get_ssim_time_loss,
                     'ssim_mel': get_ssim_mel_loss,
                     'l1_mel': get_l1_mel_loss,
                     'l2_mel': get_l2_mel_loss,
                     'l1_fft': get_l1_fft_loss,
                     'l2_fft': get_l2_fft_loss}
        self.loss_fn = loss_dict[loss_type](**loss_kwargs)

    def forward(self, pred, label):
        return self.loss_fn(pred, label)


# class Loss(nn.Module):
#     def __init__(self, mse_weight=0.1, std_weight=0.1, fft_weight=0.1, ssim_weight=1.):
#         """
#         MSE (may need to correct for latency later)
#         Difference in std
#         Spectral difference
#         """
#         super(Loss, self).__init__()
#         self.mse_weight = mse_weight
#         self.std_weight = std_weight
#         self.fft_weight = fft_weight
#         self.ssim_weight = ssim_weight
#
#     def forward(self, pred, label):
#         """
#         :param pred: model output (batch, 1, length)
#         :param label: groundtruth (batch, 1, length)
#         :return:
#         """
#         mse_loss = nn.MSELoss(reduction='mean')(pred, label)
#
#         std_loss = torch.square(torch.std_mean(pred)[0] - torch.std_mean(label)[0])
#
#
#         fft_loss = get_fft_loss(pred, label)
#
#         ssim_loss = get_ssim_loss(pred, label)
#
#         total_loss = self.mse_weight * mse_loss + self.std_weight * std_loss + self.fft_weight * fft_loss \
#                      + self.ssim_weight * ssim_loss
#
#         return total_loss
