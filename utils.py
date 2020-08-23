import numpy as np
import scipy.io.wavfile

LEN_CLIP = 10
LEN_SILENCE = 0.25
OUTPUT_LEVELS = 40
RATE = 22050


def save_wav(data, filename, rate=RATE):
    scipy.io.wavfile.write(filename, rate, data)


def load_wav(filename):
    if isinstance(filename, list):
        data = []
        for f in filename:
            d, rate = load_wav(f)
            data.append(d)
        return np.concatenate(data, axis=0), rate
    rate, data = scipy.io.wavfile.read(filename)
    return data, rate


def find_latency(train_signal='chirp'):
    from train import BASE_DIR
    if train_signal == 'chirp':
        train_signal = BASE_DIR + 'train2.wav'
        train_label = BASE_DIR + 'train2_2204_212.wav'
    elif train_signal == 'noise':
        train_signal = BASE_DIR + 'train.wav'
        train_label = BASE_DIR + 'train_2204_212.wav'
    elif train_signal == 'validation':
        train_signal = BASE_DIR + 'validation.wav'
        train_label = BASE_DIR + 'validation_2204_212.wav'
    train_signal, _ = load_wav(train_signal)
    train_label, _ = load_wav(train_label)
    train_label = train_label[:train_signal.size]

    # def normalize(x):
    #     x = x-np.mean(x)
    #     x = x/np.std(x)
    #     return x
    def normalize(x):
        x = np.sign(x).astype('float')
        return x
    train_label = normalize(train_label)
    train_signal = normalize(train_signal)

    max_shift = int(RATE*3*1e-3)
    shifts = np.arange(0, max_shift)
    mses = [np.mean(np.square(train_signal-train_label)) if shift == 0
            else np.mean(np.square(train_signal[shift:]-train_label[:-shift]))
            for shift in shifts]
    print(mses)
    print(np.mean(np.square(train_label)))
    print(shifts[np.argmin(mses)])


def generate_noise(length, std=0.1):
    return np.random.normal(loc=0., scale=std, size=[length])


def generate_chirp(length, f_start, f_end, harmonic_ratio):
    n_harmonics = int(np.floor(np.log2(RATE/2/max([f_start, f_end]))))
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
    for harmonic_ratio in [0.1*i for i in range(10)]:
        chirp = generate_chirp(LEN_CLIP - LEN_SILENCE, f_low, f_high, harmonic_ratio)
        for i, std in enumerate(np.logspace(-3.5, -0.6, OUTPUT_LEVELS, endpoint=True)):
            x = np.sqrt(2)*std*chirp
            signal.append(x)
            signal.append(np.zeros([int(RATE*LEN_SILENCE)]))
    signal = np.concatenate(signal)
    save_wav(signal, saveto, RATE)


def split_signal(data):
    # split_length = 96000  # this appears to be the limit for laptop cpu training
    split_length = int(np.ceil(data.size/(data.size//1e5)))
    if data.size % split_length != 0:
        data = np.concatenate([data, np.zeros(split_length-(data.size % split_length), dtype=data.dtype)], axis=0)
    data = np.array(np.split(data, np.ceil(data.size/split_length)))
    data = np.expand_dims(data, axis=1)
    return data
    # data = np.expand_dims(np.expand_dims(data, axis=0), axis=0)
    # return data


def clear_print(string='', end=''):
    clear = '\x1b[2K\r'
    print(clear+string, end=end)
