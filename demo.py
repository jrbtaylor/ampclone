import os
from collections import OrderedDict
from PIL import Image, ImageFont, ImageDraw

import imageio
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from scipy.ndimage import gaussian_filter1d
import torch

from utils import load_wav, save_wav
from model import get_model
from config import FS, IR_FILE, DEVICE, VAL_SPLIT_LENGTH

FINAL_WIDTH = 60  # TODO: update this later
FINAL_DEPTH = 6

DI_DIR = '/home/jtaylor/cloner/data_44100'
MODEL_DIR = '/home/jtaylor/cloner/final3_models/'
OUTPUT_DIR = '/home/jtaylor/cloner/demo_squareweights/'
PLOT_DIR = '/home/jtaylor/cloner/plots/'
DISPLAY = {'5150': '5150',
           'ac30': 'AC30',
           'jcm800': 'JCM800',
           'plexi': 'SuperLead',
           'recto': 'TripleRectifier',
           'twin': 'Twin'}


def _get_ckpt(amp):
    folder = os.path.join(MODEL_DIR, amp)
    return next(os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.pth'))


def load_ckpt(ckpt_path, model_type='blender', width=FINAL_WIDTH, model_kwargs=None):
    if model_kwargs is None:
        model_kwargs = dict([])
    model_kwargs['width'] = width
    net = get_model(model_type, model_kwargs)
    net.load_state_dict(torch.load(ckpt_path))
    return net


def combine_ckpts(ckpt0, ckpt1, alpha, model_type='blender', width=FINAL_WIDTH, depth=FINAL_DEPTH, model_kwargs=None,
                  return_weights=True):
    """
    Returns a models with the parameters given as (1-alpha)*ckpt0+alpha*ckpt1
    if alpha > 1 or < 0, the result is extrapolation

    if return_weights, returns weights as numpy array shaped [n_layers, n_bands]
    """
    param0 = torch.load(ckpt0)
    param1 = torch.load(ckpt1)
    # for squared weights, need to square, average, then take square root. for all other nonsquared params, just average
    def square_avg(key):  # note: clamp avoids NaNs
        return torch.sqrt(torch.clamp((1-alpha)*torch.square(param0[key])+alpha*torch.square(param1[key]),
                                      0., 100.))
    def avg(key):
        return (1-alpha)*param0[key]+alpha*param1[key]
    state_dict = OrderedDict(
        [(key, square_avg(key) if 'recombines' in key else avg(key))
         for key in param0.keys()])

    if model_kwargs is None:
        model_kwargs = dict([])
    model_kwargs['width'] = width
    model_kwargs['depth'] = depth
    net = get_model(model_type, model_kwargs)
    net.load_state_dict(state_dict)
    if not return_weights:
        return net
    else:
        # weights = np.concatenate(
        #     [state_dict['recombines.%i.weight' % idx].data.cpu().numpy() for idx in range(depth)], axis=0)
        weights = np.concatenate(
            [state_dict['recombines.%i' % idx].data.cpu().numpy() for idx in range(depth)], axis=0)
        weights = np.squeeze(weights, axis=-1)
        return net, weights


def inference(net, input_signal, return_activations=True):
    if isinstance(input_signal, str):
        assert os.path.isfile(input_signal)
        input_signal, fs = load_wav(input_signal)
        assert fs == FS
        # if input_signal.size > VAL_SPLIT_LENGTH:
        #     print('****   WARNING: TRUNCATING INPUT TO %i SAMPLES DUE TO MEMORY LIMITATIONS   ****' % VAL_SPLIT_LENGTH)
        #     input_signal = input_signal[:int(VAL_SPLIT_LENGTH)]
        input_signal = torch.tensor(input_signal).unsqueeze(0).unsqueeze(0)
        input_signal.to(DEVICE)
    if return_activations:
        net.demo_mode()
        output, activations = net(input_signal)
        output = output.data.cpu().numpy().reshape([-1])
        activations = [act.data.cpu().numpy() for act in activations]
        activations = np.concatenate(activations, axis=0)  # [n_layers, n_bands, n_samples]
        activations = np.transpose(activations, axes=[2, 0, 1])  # [n_samples, n_layers, n_bands]
        return output, activations
    else:
        return net(input_signal).data.cpu().numpy()


def run_demo(ckpt0, ckpt1, di_path, save_to, alpha=0.5, apply_ir=True, width=FINAL_WIDTH):
    net, weights = combine_ckpts(ckpt0, ckpt1, alpha, model_type='blender', width=width, return_weights=True)
    net.to(DEVICE)
    output, activations = inference(net, di_path, return_activations=True)
    if apply_ir:
        ir, ir_fs = load_wav(IR_FILE)
        assert ir_fs == FS
        length = output.size
        output = np.convolve(output, ir)
        remove = output.size - length
        output = output[:-remove]
    # normalize
    output = 0.17*output/np.nanstd(output)
    if np.nanmax(np.abs(output)) > 1.:
        output = output/np.nanmax(np.abs(output))
    print('WAV MAX: '+str(np.max(np.abs(output))))
    save_vis(weights, activations, output, save_to, framerate=30)


def colormap():
    norm = matplotlib.colors.Normalize(vmin=0, vmax=255, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap='magma')
    values = list(range(256))
    cmap = []
    for v in values:
        cmap.append(mapper.to_rgba(v)[:3])
    return np.array(cmap)


def save_vis(weights, activations, model_output, save_to, framerate=30):
    """
    :param weights: ndarray of shape [n_layers, n_bands]
    :param activations: ndarray of shape [n_samples, n_layers, n_bands]
    :param model_output: ndarray of shape [n_samples]
    :param save_to: filename to save to in OUTPUT_DIR
    :param framerate: video frames-per-second
    :return:
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    audio_save = os.path.join(OUTPUT_DIR, save_to + '.wav')
    save_wav(model_output, audio_save)

    # smooth the activations and downsample to appropriate framerate for video
    activations = np.abs(activations)
    assert FS % framerate == 0  # only integer downsampling is implemented here
    downsample_ratio = int(FS / framerate)
    chunk_size = downsample_ratio
    if activations.shape[0] % chunk_size != 0:
        pad = int(np.ceil(activations.shape[0] / chunk_size) * chunk_size - activations.shape[0])
        zeros = np.zeros([pad, activations.shape[1], activations.shape[2]])
        activations = np.concatenate([activations, zeros], axis=0)
    activations = np.transpose(activations, axes=[1, 2, 0])  # time last: [n_layers, n_bands, n_samples]
    activations = np.reshape(activations, [activations.shape[0], activations.shape[1], -1, chunk_size])
    activations = np.mean(activations, axis=-1)
    activations = np.transpose(activations, axes=[2, 0, 1])  # time first: [n_samples, n_layers, n_bands]
    activations = gaussian_filter1d(activations, sigma=1, axis=0, mode='constant')  # smooth a bit more

    # weights = np.square(weights)
    weights = np.power(np.abs(weights), 0.2)  # rescale non-linearly for clearer & more interesting visualization
    min_w = np.percentile(weights, 5, axis=-1, keepdims=True)
    max_w = np.percentile(weights, 95, axis=-1, keepdims=True)
    weights = (255 * np.clip((weights - min_w) / (max_w - min_w), 0., 1.)).astype('uint8')
    min_a = np.percentile(activations, 5, axis=0, keepdims=True)
    max_a = np.percentile(activations, 95, axis=0, keepdims=True)
    activations = (255 * np.clip((activations - min_a) / (max_a - min_a), 0., 1.)).astype('uint8')

    weights = np.expand_dims(weights, axis=0)
    imgs = np.zeros([activations.shape[0], activations.shape[1] + weights.shape[1], activations.shape[2]],
                    dtype='uint8')
    imgs[:, 0::2] = activations
    imgs[:, 1::2] = weights

    norm = matplotlib.colors.Normalize(vmin=0, vmax=255, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap='afmhot')
    shape = imgs.shape
    imgs = mapper.to_rgba(imgs.reshape(-1))[..., :3].reshape(list(shape) + [3])
    imgs = (255 * imgs).astype('uint8')

    video_save = os.path.join(OUTPUT_DIR, save_to + '_videoonly.mp4')
    fontsize = 76
    font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", fontsize)
    writer = imageio.get_writer(video_save, fps=framerate)
    display_text = save_to.replace('_', ' ')
    display_text = ' '.join([DISPLAY.get(w, w) for w in display_text.split(' ')])
    for idx in range(imgs.shape[0]):
        x = imgs[idx]
        x = np.concatenate([190*np.ones([1, x.shape[1], 3], dtype='uint8'), x], axis=0)
        x = Image.fromarray(x).resize((40 * x.shape[1], 96 * x.shape[0]), Image.NEAREST)
        draw = ImageDraw.Draw(x)
        # draw.text((5, 10), display_text, (0, 255, 150), font=font)
        draw.text((5, 5), display_text, (0, 0, 0), font=font)
        writer.append_data(np.array(x))
    writer.close()

    # merge the audio file and video file, save to demo.mp4
    final_output = os.path.join(OUTPUT_DIR, save_to + '.mp4')
    if os.path.isfile(final_output):
        os.remove(final_output)  # avoids prompt to overwrite in ffmpeg command
    os.system('ffmpeg -i %s -i %s -c:v copy -map 0:v:0 -map 1:a:0 -c:a aac -b:a 192k %s'
              % (video_save, audio_save, final_output))
    os.remove(video_save)


def vis_activations():
    ckpt0 = _get_ckpt('jcm800')
    di_path = os.path.join(DI_DIR, 'test.wav')
    net, weights = combine_ckpts(ckpt0, ckpt0, alpha=0., model_type='blender', return_weights=True)
    net.to(DEVICE)
    input_signal, _ = load_wav(di_path)
    output, activations = inference(net, di_path, return_activations=True)  # [n_samples, n_layers, n_bands]

    offset = int(5e5)
    length = int(3e3)
    input_signal = np.squeeze(input_signal[offset:offset+length])                   # [n_samples]
    activations = np.transpose(np.squeeze(activations[offset:offset+length, 1, :]))  # [n_bands, n_samples]
    output = np.squeeze(output[offset:offset+length])                               # [n_samples]

    def plot(y, name):
        plt.figure(1)
        plt.plot(y)
        ax = plt.gca()
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(os.path.join(PLOT_DIR, name+'.jpeg'), bbox_inches='tight', pad_inches=0)
        plt.clf()

    plot(input_signal, 'input')
    plot(output, 'output')
    for idx in range(activations.shape[0]):
        plot(activations[idx], 'activation_%i.jpeg' % idx)


def full_demo():
    di_path = os.path.join(DI_DIR, 'test.wav')
    amps = ['5150', 'ac30', 'jcm800', 'twin', 'recto', 'plexi']

    # INTERPOLATION AND SINGLE-AMP EXPERIMENTS
    for idx0 in range(len(amps)):
        for idx1 in range(idx0+1):  # also include same amp
            print('*' * 32 + '\n' + '    INTERPOLATION TEST: ' + amps[idx0] + ' + ' + amps[idx1] + '\n' + '*' * 32)
            if idx0 == idx1:
                save_to = amps[idx0]
            else:
                save_to = 'interpolate_0.5_from_' + amps[idx0] + '_to_' + amps[idx1]
            run_demo(_get_ckpt(amps[idx0]), _get_ckpt(amps[idx1]), di_path, save_to, alpha=0.5)

    # EXTRAPOLATION EXPERIMENTS
    for extrap in [1.25, 1.5, 2.]:
        for idx0 in range(len(amps)):
            for idx1 in range(len(amps)):
                if idx0 == idx1:
                    continue
                print('*' * 32 + '\n' + '    EXTRAPOLATION TEST %.2f: ' % extrap + amps[idx0] + ' + ' + amps[
                    idx1] + '\n' + '*' * 32)
                save_to = 'extrapolate_%.2f_from_' % extrap + amps[idx0] + '_to_' + amps[idx1]
                run_demo(_get_ckpt(amps[idx0]), _get_ckpt(amps[idx1]), di_path, save_to, alpha=extrap)
