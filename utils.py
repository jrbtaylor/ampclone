import numpy as np
import scipy.io.wavfile

LEN_CLIP = 5
LEN_SILENCE = 0.25
OUTPUT_LEVELS = 40
RATE = 22050


def save_wav(data, filename, rate=RATE):
    scipy.io.wavfile.write(filename, rate, data)


def load_wav(filename):
    rate, data = scipy.io.wavfile.read(filename)
    return data, rate


def generate_noise(length, std=0.1):
    return np.random.normal(loc=0., scale=std, size=[length])


def generate_chirp(length, f_start, f_end, harmonic_ratio):
    n_harmonics = np.floor(np.log2(RATE/2/max([f_start, f_end])))
    if max([f_start, f_end])*2**n_harmonics >= RATE/2:
        raise ValueError('Cannot have %i harmonics. Highest frequency is above the Nyquist rate' % n_harmonics)
    t = np.linspace(0, length, int(length*RATE))
    f = np.linspace(f_start, f_end, int(length*RATE), endpoint=True)
    signal = np.zeros(int(length*RATE), dtype='float')
    for h in range(n_harmonics):
        fh = f*2**h
        signal += (harmonic_ratio**h)*np.sin(2*np.pi*fh*t)
    return signal


def generate_train_signal(saveto):
    """
    generate segments of length LEN_CLIP seconds with LEN_SILENCE second of silence at the end of each

    :param saveto: path to save training wav file
    :return:
    """
    length = int((LEN_CLIP-LEN_SILENCE)*RATE)
    signal = []
    for std in np.logspace(-3.5, -1, OUTPUT_LEVELS, endpoint=True):
        signal.append(generate_noise(length, std=std))
        signal.append(np.zeros([int(RATE*LEN_SILENCE)]))
    signal = np.concatenate(signal)
    save_wav(signal, saveto, RATE)


def generate_train_signal2(saveto):
    signal = []
    f_low = 40  # bass low E
    f_high = 80*8  # twelfth fret high E
    harmonic_ratio = 0.5  # each harmonic is half the magnitude of the prior one
    chirp = generate_chirp(LEN_CLIP - LEN_SILENCE, f_low, f_high, harmonic_ratio)
    for i, std in enumerate(np.logspace(-3.5, -0.6, OUTPUT_LEVELS, endpoint=True)):
        signal.append(np.sqrt(2)*std*chirp)
        signal.append(np.zeros([int(RATE*LEN_SILENCE)]))
    signal = np.concatenate(signal)
    save_wav(signal, saveto, RATE)


def split_signal(data):
    split_length = 96000  # this appears to be the limit for laptop cpu training
    if data.size % split_length != 0:
        data = np.concatenate([data, np.zeros(split_length-(data.size % split_length), dtype=data.dtype)], axis=0)
    data = np.array(np.split(data, np.ceil(data.size/split_length)))
    data = np.expand_dims(data, axis=1)
    return data


def clear_print(string='', end=''):
    clear = '\x1b[2K\r'
    print(clear+string, end=end)
