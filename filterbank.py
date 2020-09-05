import numpy as np
import scipy.signal
from scipy.signal import firwin, firwin2


def filterbank(n_filters, f_min, f_max, fs, filter_order=3):
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
    passband_loss = 0.5
    stopband_atten = 20
    ftype = 'ellip'
    for idx in range(n_filters):
        b, a = scipy.signal.iirfilter(filter_order, passbands[idx], rp=passband_loss, rs=stopband_atten,
                                      btype='bandpass', ftype=ftype, output='ba', fs=fs)
        filters.append((b, a))
    return filters


def vis_filters():
    import matplotlib
    matplotlib.use("Agg")

    from scipy import signal
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3)
    fs = 44100
    # filters = filterbank(40, 40., 16000., fs, 3)
    # filters = filterbank(30, 40., 16000., fs, 3)
    # filters = filterbank(20, 40., 16000., fs, 3)
    # filters = filterbank(10, 40., 16000., fs, 3)
    # filters = filterbank(5, 80., 10000., fs, 3)
    filters = filterbank(6, 60., 11000., fs, 3)
    filter_sum = None
    plot_min = 5
    plot_max = 500  # fs/2
    for idx, (b, a) in enumerate(filters):
        print((len(b), len(a)))
        freq, response = signal.freqz(b, a, worN=2048*4, fs=fs)
        response = response[np.where(np.greater(freq, plot_min))]
        freq = freq[np.where(np.greater(freq, plot_min))]
        _ = axs[0].semilogy(freq, np.abs(response))[0]
        response = response[np.where(np.logical_and(np.less(freq, plot_max), np.greater(freq, plot_min)))]
        freq = freq[np.where(np.logical_and(np.less(freq, plot_max), np.greater(freq, plot_min)))]
        _ = axs[1].semilogy(freq, np.abs(response))[0]
        if filter_sum is None:
            filter_sum = np.abs(response)
        else:
            filter_sum += np.abs(response)

    axs[0].set_xlabel('Frequency (Hz)')
    axs[0].grid(True)

    _ = axs[2].semilogy(freq, filter_sum)

    fig.tight_layout()
    plt.savefig('/home/jtaylor/filter_debug.png')
