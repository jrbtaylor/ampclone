import os
from collections import OrderedDict

import imageio
import moviepy
import numpy as np
from scipy.ndimage import gaussian_filter1d
import torch

from utils import load_wav, save_wav
from model import get_model
from config import FS, IR_FILE, DEVICE, VAL_SPLIT_LENGTH


# TODO: update these (to ~95th percentile) after running the training set through a few models
MIN_WEIGHT = 0.
MAX_WEIGHT = 1.
MIN_ACTIVATION = 0.
MAX_ACTIVATION = 1.

FINAL_WIDTH = 60  # TODO: update this later
FINAL_DEPTH = 6

DI_DIR = '/home/jtaylor/cloner/'
MODEL_DIR = '/home/jtaylor/cloner/models_44100/'


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
    state_dict = OrderedDict([(key, (1-alpha)*param0[key]+alpha*param1[key]) for key in param0.keys()])

    if model_kwargs is None:
        model_kwargs = dict([])
    model_kwargs['width'] = width
    model_kwargs['depth'] = depth
    net = get_model(model_type, model_kwargs)
    net.load_state_dict(state_dict)
    if not return_weights:
        return net
    else:
        weights = np.concatenate(
            [state_dict['recombines.%i.weight' % idx].data.cpu().numpy() for idx in range(depth)], axis=0)
        weights = np.squeeze(weights, axis=-1)
        return net, weights


def inference(net, input_signal, return_activations=True):
    if isinstance(input_signal, str):
        assert os.path.isfile(input_signal)
        input_signal, fs = load_wav(input_signal)
        assert fs == FS
        if input_signal.size > VAL_SPLIT_LENGTH:
            print('****   WARNING: TRUNCATING INPUT TO %i SAMPLES DUE TO MEMORY LIMITATIONS   ****' % VAL_SPLIT_LENGTH)
            input_signal = input_signal[:int(VAL_SPLIT_LENGTH)]
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


def run_demo(ckpt0, ckpt1, di_path, save_dir, alpha=0.5, apply_ir=True, width=60):
    if apply_ir:
        ir, ir_fs = load_wav(IR_FILE)
        assert ir_fs == FS

    net, weights = combine_ckpts(ckpt0, ckpt1, alpha, model_type='blender', width=width, return_weights=True)
    net.to(DEVICE)
    output, activations = inference(net, di_path, return_activations=True)
    if apply_ir:
        output = np.convolve(output, ir)
    save_vis(weights, activations, output, save_dir, framerate=30)


def save_vis(weights, activations, model_output, save_dir, framerate=30):
    """
    :param weights: ndarray of shape [n_layers, n_bands]
    :param activations: ndarray of shape [n_samples, n_layers, n_bands]
    :param model_output: ndarray of shape [n_samples]
    :param save_dir: path to output directory, to save "output.wav" (audio only) and "output.mp4" (video w/ audio)
    :param framerate: video frames-per-second
    :return:
    """
    os.makedirs(save_dir, exist_ok=True)

    audio_save = os.path.join(save_dir, 'output.wav')
    save_wav(model_output, audio_save)

    # smooth the activations and downsample to appropriate framerate for video
    activations = np.abs(activations)
    assert FS % framerate == 0  # only integer downsampling is implemented here
    downsample_ratio = int(FS/framerate)
    # activations = gaussian_filter1d(activations, sigma=3*downsample_ratio, axis=0, mode='constant')  # TODO: fix this, it's so slow it may never finish
    # activations = activations[::downsample_ratio]
    # TODO: try this version tomorrow, else debug above
    chunk_size = downsample_ratio
    if activations.shape[0] % chunk_size != 0:
        print(activations.shape)
        zeros = np.zeros([activations.shape[0] % chunk_size, activations.shape[1], activations.shape[2]])
        activations = np.concatenate([activations, zeros], axis=0)
    activations = np.transpose(activations, axes=[1, 2, 0])  # time last: [n_layers, n_bands, n_samples]
    activations = np.reshape(activations, [activations.shape[0], activations.shape[1], -1, chunk_size])
    activations = np.mean(activations, axis=-1)
    activations = np.transpose(activations, axes=[2, 0, 1])  # time first: [n_samples, n_layers, n_bands]
    activations = gaussian_filter1d(activations, sigma=1, axis=0, mode='constant')  # smooth a bit more

    # TODO: update this normalization
    weights = (255*np.clip((weights-MIN_WEIGHT)/(MAX_WEIGHT-MIN_WEIGHT), 0., 1.)).astype('uint8')
    activations = (255*np.clip((activations-MIN_ACTIVATION)/(MAX_ACTIVATION-MIN_ACTIVATION), 0., 1.)).astype('uint8')

    weights = np.expand_dims(weights, axis=0)
    imgs = np.zeros([activations.shape[0], activations.shape[1]+weights.shape[1], activations.shape[2]])
    imgs[:, 0::2] = activations
    imgs[:, 1::2] = weights

    video_save = os.path.join(save_dir, 'output.mp4')
    writer = imageio.get_writer(video_save, fps=framerate)
    for idx in range(imgs.shape[0]):
        writer.append_data(imgs[idx])
    writer.close()

    # videoclip = moviepy.video.io.VideoFileClip.VideoFileClip(video_save, audio=False)
    # audioclip = moviepy.audio.io.AudioFileClip.AudioFileClip(audio_save, fps=FS)
    # videoclip.set_audio(audioclip)
    moviepy.video.io.ffmpeg_tools.ffmpeg_merge_video_audio(video=video_save, audio=audio_save, output=video_save)












