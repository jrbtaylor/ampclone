import numpy as np
import scipy.signal
from scipy.signal import firwin


def _fir(n_filters, filter_order, bands, f_min, fs, window='blackmanharris', min_phase=False):
    filters = []
    group_delays = []
    for idx in range(n_filters):
        h = firwin(filter_order, (bands[idx], bands[idx + 1]), pass_zero='bandpass', fs=fs, window=window)
        if min_phase:
            h = scipy.signal.minimum_phase(h)
        filters.append(h)

        freq, delay = scipy.signal.group_delay((h, 1.), w=2048 * 16, fs=fs)
        delay = np.max(delay[np.where(np.greater(freq, f_min))])
        group_delays.append(delay)
    return filters, group_delays


def filterbank_fir(n_filters, f_min, f_max, fs, max_delay, window='hann', min_phase=False, offset=250):
    filter_order = int(2*max_delay+1)

    print('Window function: '+str(window)+' '*8+'Offset: '+str(offset))
    bands = np.geomspace(f_min+offset, f_max+offset, n_filters+1, endpoint=True)-offset
    print(bands)

    while True:
        filters, group_delays = _fir(n_filters, filter_order, bands, f_min, fs, window, min_phase)
        if np.max(group_delays) > max_delay:
            filter_order = filter_order-1
        else:
            break

    print('max group delay: ' + str(np.max(group_delays)))
    print('filter length: ' + str(filters[0].size))

    return filters


def vis_filters():
    import matplotlib
    matplotlib.use("Agg")

    from scipy import signal
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(4)
    fs = 44100
    filters = filterbank_fir(30, 80., 16000., fs, fs*2e-3/7)

    filter_sum = None
    filter_sum_truncated = None
    plot_min = 5
    plot_max = 1000  # fs/2
    # for idx, (b, a) in enumerate(filters):
    #     print((len(b), len(a)))
    #     freq, response = signal.freqz(b, a, worN=2048*16, fs=fs)
    for idx, b in enumerate(filters):
        freq, response = signal.freqz(b, worN=2048*16, fs=fs)
        # response *= hp_response
        response = response[np.where(np.greater(freq, plot_min))]
        freq = freq[np.where(np.greater(freq, plot_min))]
        if filter_sum is None:
            filter_sum = np.abs(response)
        else:
            filter_sum += np.abs(response)
        _ = axs[0].semilogy(freq, np.abs(response))[0]
        response = response[np.where(np.logical_and(np.less(freq, plot_max), np.greater(freq, plot_min)))]
        freq_truncated = freq[np.where(np.logical_and(np.less(freq, plot_max), np.greater(freq, plot_min)))]
        if idx < 10:
            _ = axs[2].semilogy(freq_truncated, np.abs(response))[0]
        if filter_sum_truncated is None:
            filter_sum_truncated = np.abs(response)
        else:
            filter_sum_truncated += np.abs(response)

    axs[0].set_xlabel('Frequency (Hz)')
    axs[0].grid(True)

    _ = axs[1].semilogy(freq, filter_sum)
    _ = axs[3].semilogy(freq_truncated, filter_sum_truncated)

    fig.tight_layout()
    plt.savefig('/home/jtaylor/filter_debug.png')
