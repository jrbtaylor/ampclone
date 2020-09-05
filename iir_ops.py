"""
Taken from https://discuss.pytorch.org/t/how-to-create-a-rnn-that-applies-n-different-recurrence-relations-to-the-input/87709
"""

import torch


def m_lfilter(
        waveform: torch.Tensor,
        a_coeffs: torch.Tensor,
        b_coeffs: torch.Tensor
) -> torch.Tensor:
    r"""Perform an IIR filter by evaluating difference equation.

    NB : contrary to the original version this one does not requires normalized input and does not ouput normalized sequences.

    Args:
        waveform (Tensor): audio waveform of dimension of `(..., number_of_filters, time)`.
        a_coeffs (Tensor): denominator coefficients of difference equation of dimension of `(n_order + 1)`.
                                Lower delays coefficients are first, e.g. `number_of_filters*[a0, a1, a2, ...]`.
                                Must be same size as b_coeffs (pad with 0's as necessary).
        b_coeffs (Tensor): numerator coefficients of difference equation of dimension of `(n_order + 1)`.
                                 Lower delays coefficients are first, e.g. `number_of_filters*[b0, b1, b2, ...]`.
                                 Must be same size as a_coeffs (pad with 0's as necessary).

    Returns:
        Tensor: Waveform with dimension of `(..., number_of_filters, time)`.


    Note :
      The main difference with the original version is that we are not packing anymore  the batches (since we need to apply different filters)
    """

    # Jason: added this to broadcast input to number of filters
    waveform = waveform.expand(-1, a_coeffs.size(0), -1).clone()

    shape = waveform.size()  # should returns [batch_size, number_of_filters, size_of_the_sequence]

    assert (a_coeffs.size(0) == b_coeffs.size(0))
    assert (len(waveform.size()) == 3)
    assert (waveform.device == a_coeffs.device)
    assert (b_coeffs.device == a_coeffs.device)
    device = waveform.device
    dtype = waveform.dtype
    n_channel, n_filters, n_sample = waveform.size()
    n_order = a_coeffs.size(1)
    assert (a_coeffs.size(
        0) == n_filters)  # number of filters to apply - for each filter k, the coefs are in a_coeffs[k] and b_coeffs[k]
    n_sample_padded = n_sample + n_order - 1
    assert (n_order > 0)

    # Pad the input and create output
    padded_waveform = torch.zeros(n_channel, n_filters, n_sample_padded, dtype=dtype, device=device)
    padded_waveform[:, :, (n_order - 1):] = waveform
    padded_output_waveform = torch.zeros(n_channel, n_filters, n_sample_padded, dtype=dtype,
                                         device=device)  # padded_output_waveform = torch.zeros(n_channel, n_sample_padded, dtype=dtype, device=device)

    # Set up the coefficients matrix
    # Flip coefficients' order
    a_coeffs_flipped = a_coeffs.flip(1).unsqueeze(0)
    b_coeffs_flipped = b_coeffs.flip(1).t()

    # calculate windowed_input_signal in parallel
    # create indices of original with shape (n_channel, n_order, n_sample)
    window_idxs = torch.arange(n_sample, device=device).unsqueeze(0) + torch.arange(n_order, device=device).unsqueeze(1)
    window_idxs = window_idxs.repeat(n_channel, 1, 1)

    window_idxs += (torch.arange(n_channel, device=device).unsqueeze(-1).unsqueeze(-1) * n_sample_padded)
    window_idxs = window_idxs.long()

    # (n_filters, n_order) matmul (n_channel, n_order, n_sample) -> (n_channel, n_filters, n_sample)
    A = torch.take(padded_waveform, window_idxs).permute(0, 2, 1)  # taking the input coefs
    input_signal_windows = torch.matmul(torch.take(padded_waveform, window_idxs).permute(0, 2, 1),
                                        b_coeffs_flipped).permute(1, 0, 2)
    # input_signal_windows size : n_samples x batch_size x n_filters

    for i_sample, o0 in enumerate(input_signal_windows):
        # added clone here for back propagation
        windowed_output_signal = padded_output_waveform[:, :, i_sample:(i_sample + n_order)].clone()
        o0.sub_(torch.mul(windowed_output_signal, a_coeffs_flipped).sum(dim=2))
        o0.div_(a_coeffs[:, 0])

        padded_output_waveform[:, :, i_sample + n_order - 1] = o0

    output = padded_output_waveform[:, :, (n_order - 1):]
    return output
