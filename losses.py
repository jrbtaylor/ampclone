import numpy as np
import torch.nn as nn
import torch
import torchaudio

from config import FS


def _ssim_loss(x, y):
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    var_x = torch.var(x)
    var_y = torch.var(y)
    cov = torch.mean(x * y) - mean_x * mean_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim = (2 * mean_x * mean_y + c1) * (2 * cov + c2) / (torch.square(mean_x) + torch.square(mean_y) + c1) / (
            var_x + var_y + c2)
    return (1 - ssim) / 2


def _spectral_loss(diff, spec, n_ffts):
    if not isinstance(n_ffts, list):
        n_ffts = [n_ffts]

    def _loss_fn(pred, label):
        loss = 0
        for n_fft in n_ffts:
            # TODO: add frequency weighting based on perceptual sensitivity (https://en.wikipedia.org/wiki/Mel_scale)
            if spec == 'mel':
                spec_fn = torchaudio.transforms.MelSpectrogram(sample_rate=FS, n_fft=n_fft)
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
        return loss / len(n_ffts)

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


def get_blend_loss(n_ffts, time_weight):
    l2_mel_loss = get_l2_mel_loss(n_ffts)
    mse_loss = get_mse_time_loss()

    def _blend(pred, label):
        return l2_mel_loss(pred, label) + time_weight * mse_loss(pred, label)

    return _blend


class Loss(nn.Module):
    def __init__(self, loss_type, loss_kwargs):
        super(Loss, self).__init__()
        loss_dict = {'l2_time': get_mse_time_loss,
                     'ssim_time': get_ssim_time_loss,
                     'ssim_mel': get_ssim_mel_loss,
                     'l1_mel': get_l1_mel_loss,
                     'l2_mel': get_l2_mel_loss,
                     'l1_fft': get_l1_fft_loss,
                     'l2_fft': get_l2_fft_loss,
                     'blend': get_blend_loss}
        self.loss_fn = loss_dict[loss_type](**loss_kwargs)

    def forward(self, pred, label):
        return self.loss_fn(pred, label)


def phase_loss(net):
    filters = net.filters.filters
    real = torch.reshape(filters, [-1, filters.shape[-1]])  # n_filters x filter_length
    # zero-pad for additional freq resolution
    # zeros = torch.zeros([real.shape[0], 64-real.shape[-1]])
    # real = torch.cat([real, zeros], dim=-1)

    # concat zeros for imaginary part for FFT
    imag = torch.zeros_like(real)
    comp_filters = torch.stack([real, imag], dim=-1)  # n_filters x filter_length x 2
    fft = torch.Tensor.fft(comp_filters, 1)
    phase = torch.atan2(fft[:, :, 1], fft[:, :, 0])

    # estimate group delay (derivative of phase wrt freq) by finite differences
    group_delay = phase[:, 1:] - phase[:, :-1]
    cos = torch.cos(group_delay)
    sin = torch.sin(group_delay)

    return torch.var(sin) + torch.var(cos)
