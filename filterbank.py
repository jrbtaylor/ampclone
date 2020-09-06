import numpy as np
import scipy.signal
from scipy.signal import firwin, firwin2


def filterbank_iir(n_filters, f_min, f_max, fs, filter_order=3):
    # Note: filter length is 2*filter_order+1
    center_freqs = np.geomspace(f_min, f_max, n_filters, endpoint=True)
    gaps = center_freqs[1:]-center_freqs[:-1]
    gaps = np.stack([gaps[0]**2/gaps[1]]+list(gaps))
    multiplier = {40: 0.43, 30: 0.47, 20: 0.5, 10: 0.54, 5: 0.68}  # found empirically to give approx flat filter sum
    multiplier = np.polyval(np.polyfit(sorted(list(multiplier.keys())),
                                       [multiplier[k] for k in sorted(list(multiplier.keys()))], 2),
                            n_filters)
    print(multiplier)
    passbands = [(center_freqs[idx]-gaps[idx]*multiplier, center_freqs[idx]+gaps[idx]*multiplier)
                 for idx in range(n_filters)]
    stopbands = [(f_min//4 if idx == 0 else center_freqs[idx-1],
                  f_max+0.75*(fs/2-f_max) if idx == n_filters-1 else center_freqs[idx+1])
                 for idx in range(n_filters)]

    print(passbands)
    print(stopbands)

    filters = []
    ftype = 'ellip'
    for idx in range(n_filters):
        passband_loss = 0.5 if passbands[idx][0] > 100 else 1
        stopband_atten = 50 if passbands[idx][0] > 100 else 30
        b, a = scipy.signal.iirfilter(filter_order, passbands[idx], rp=passband_loss, rs=stopband_atten,
                                      btype='bandpass', ftype=ftype, output='ba', fs=fs)
        filters.append((b, a))
    return filters


def _fir(n_filters, filter_order, bands, f_min, fs, min_phase=False):
    filters = []
    group_delays = []
    for idx in range(n_filters):
        h = firwin(filter_order, (bands[idx], bands[idx + 1]), pass_zero='bandpass', fs=fs, window='triang')
        if min_phase:
            h = scipy.signal.minimum_phase(h)
        filters.append(h)

        freq, delay = scipy.signal.group_delay((h, 1.), w=2048 * 16, fs=fs)
        delay = np.max(delay[np.where(np.greater(freq, f_min))])
        group_delays.append(delay)
    return filters, group_delays


def filterbank_fir(n_filters, f_min, f_max, fs, max_delay, min_phase=False, offset=250):
    filter_order = 211
    if not min_phase:
        filter_order = (filter_order-1)/2

    bands = np.geomspace(f_min+offset, f_max+offset, n_filters+1, endpoint=True)-offset
    print(bands)

    while True:
        filters, group_delays = _fir(n_filters, filter_order, bands, f_min, fs, min_phase)
        if np.max(group_delays) > max_delay:
            filter_order = filter_order-2
        else:
            break

    if min_phase:
        print(group_delays)
    print('max group delay: ' + str(np.max(group_delays)))

    return filters


def vis_filters():
    import matplotlib
    matplotlib.use("Agg")

    from scipy import signal
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(4)
    fs = 44100
    filters = filterbank_fir(20, 80., 16000., fs, 12)

    # hp = scipy.signal.minimum_phase(
    #     firwin(85, 80., pass_zero='highpass', fs=fs, window='boxcar'))
    # hp_freq, hp_delay = scipy.signal.group_delay((hp, 1.), w=2048 * 16, fs=fs)
    # hp_delay = np.max(hp_delay[np.where(np.greater(hp_freq, 80))])
    # print('dc block delay: '+str(hp_delay))
    # _, hp_response = signal.freqz(hp, worN=2048*16, fs=fs)

    # filters = filterbank_iir(40, 80., 16000., fs, 3)
    # filters = filterbank_iir(30, 40., 16000., fs, 3)
    # filters = filterbank_iir(20, 40., 16000., fs, 3)
    # filters = filterbank_iir(10, 40., 16000., fs, 3)
    # filters = filterbank_iir(5, 80., 10000., fs, 3)
    # filters = filterbank_iir(6, 60., 11000., fs, 3)
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
