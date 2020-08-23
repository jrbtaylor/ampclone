import json
import os
import re
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import model
import losses
import utils

BASE_DIR = '/home/jtaylor/cloner/'  # 'C:/Users/Jason/Documents/cloner/'

if torch.cuda.is_available():
    DEVICE = torch.device('cuda:0')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    DEVICE = torch.device('cpu')


def get_dataloader(input_path, label_path, batchsize=1, shuffle=False):
    input_signal, signal_rate = utils.load_wav(input_path)
    label_signal, label_rate = utils.load_wav(label_path)
    assert (signal_rate == label_rate)
    # truncate label if it's somehow longer (reaper might've padded it)
    label_signal = label_signal[:input_signal.size]
    print(input_signal.shape)
    inputs = utils.split_signal(input_signal)
    labels = utils.split_signal(label_signal)
    print(inputs.shape)
    inputs = torch.Tensor(inputs)
    labels = torch.Tensor(labels)
    inputs = inputs.to(DEVICE)
    labels = labels.to(DEVICE)
    dataset = TensorDataset(inputs, labels)
    loader = DataLoader(dataset, batch_size=batchsize, shuffle=shuffle)
    return loader


def train_model(exp_name='model', modeltype='wavenet', model_kwargs=None,
                batchsize=1, n_epochs=1000, early_stopping=10,
                learnrate=1e-4, momentum=0.5, grad_clip=0.5,
                loss_type=None, loss_kwargs=None,
                train_signal='chirp'):
    val_signal = BASE_DIR + 'validation.wav'
    val_label = BASE_DIR + 'validation_2204_212.wav'
    if train_signal == 'chirp':
        train_signal = BASE_DIR + 'train.wav'
        train_label = BASE_DIR + 'train_2204_212.wav'
    elif train_signal == 'di':
        train_signal = BASE_DIR + 'train_alldi.wav'
        train_label = BASE_DIR + 'train_alldi_2204_212.wav'
    elif train_signal == 'all':
        train_signal = [BASE_DIR + 'train_alldi.wav', BASE_DIR + 'train.wav']
        train_label = [BASE_DIR + 'train_alldi_2204_212.wav', BASE_DIR + 'train_2204_212.wav']

    train_loader = get_dataloader(train_signal, train_label, batchsize=batchsize, shuffle=True)
    val_loader = get_dataloader(val_signal, val_label, batchsize=batchsize, shuffle=False)

    savedir = os.path.join(BASE_DIR, 'models', exp_name)
    if not os.path.isdir(savedir):
        os.makedirs(savedir)
    metrics_json = os.path.join(savedir, 'metrics.json')

    model_kwargs = {} if model_kwargs is None else model_kwargs
    if modeltype == 'wavenet':
        net = model.WaveNet(**model_kwargs)
    elif modeltype == 'fixedfilters':
        net = model.FixedFilters(**model_kwargs)
    net.to(DEVICE)

    criterion = losses.Loss(loss_type, loss_kwargs)
    # optimizer = optim.SGD(net.parameters(), lr=learnrate, momentum=momentum)
    optimizer = optim.Adam(net.parameters(), lr=learnrate)
    # optimizer = optim.Adam([
    #     {'params': net.layers[-1].parameters(), 'lr': 1e-2 * learnrate},
    #     {'params': [param for i in range(len(net.layers) - 1) for param in net.layers[i].parameters()], 'lr': learnrate}
    # ])

    train_losses = []
    val_losses = []
    best_val = [0, np.inf]
    for epoch in range(n_epochs):
        epoch_losses = []
        start_time = time.time()
        for i, data in enumerate(train_loader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
            optimizer.step()

            epoch_losses.append(loss.item())
            elapsed = time.time() - start_time
            utils.clear_print('training step %i    loss = %.4f    elapsed = %.2f minutes    ETA = %.2f minutes'
                              % (i, loss.item(), elapsed / 60, (len(train_loader) - i - 1) * elapsed / (i + 1) / 60))
            if not np.isfinite(loss.item()):
                print('NaN loss. Closing experiment.')
                return best_val[1]
        utils.clear_print('Epoch %i: training loss = %.4f' % (epoch, np.mean(epoch_losses)), end='\n')
        train_losses.append(np.mean(epoch_losses))

        epoch_losses = []
        start_time = time.time()
        val_outputs = []
        for i, data in enumerate(val_loader):
            inputs, labels = data
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            val_outputs.append(outputs.data.cpu().numpy())

            epoch_losses.append(loss.item())
            elapsed = time.time() - start_time
            utils.clear_print('validation step %i    loss = %.4f    elapsed = %.2f minutes    ETA = %.2f minutes'
                              % (i, loss.item(), elapsed / 60, (len(val_loader) - i - 1) * elapsed / (i + 1) / 60))
        utils.clear_print('Epoch %i: validation loss = %.4f' % (epoch, np.mean(epoch_losses)), end='\n')
        val_loss = np.mean(epoch_losses)
        val_losses.append(val_loss)

        with open(metrics_json, 'w') as f:
            json.dump({'train_losses': train_losses, 'val_losses': val_losses}, f)

        if val_loss < best_val[1]:
            best_val = [epoch, val_loss]
            saveto = os.path.join(savedir, exp_name + '.pth')
            torch.save(net.state_dict(), saveto)

            val_outputs = np.concatenate(val_outputs)
            utils.save_wav(val_outputs, os.path.join(savedir, 'val_output.wav'))
        elif epoch - best_val[0] >= early_stopping:
            break
    return best_val[1]


def hyperparam_search(skip_existing=True):
    """
    notes from first version:
    width and filter length don't make a huge difference
    validation loss for chirp is 50% lower than noise. stop using noise training signal
    depth makes a huge difference (tradeoff need for more gain stages with vanishing gradients from sigmoids)

    TODO: find a better loss function by training a bunch of networks on different losses and listening
    TODO: investigate other training signals, including DI files, longer training signals, etc.
    TODO: try multiplicative interactions between layers, and shortcuts (add or concat)
    """
    results_json = os.path.join(BASE_DIR, 'hyperparams_losses.json')
    if os.path.isfile(results_json) and skip_existing:
        with open(results_json, 'r') as f:
            results = json.load(f)
    else:
        results = {}
    widths = [4]  # [1] + list(range(4, 20, 4))
    depths = [5]  # range(3, 9, 2)
    filter_lengths = [1024]  # [128, 256, 512, 1024]
    signals = ['chirp']  # ['chirp', 'noise']  # validation loss for chirp is ~50% lower
    n_ffts = [2048, 4096, 4096*2, 4096*4, 4096*8]
    n_fft_kwargs = [{'n_ffts': n_fft} for n_fft in n_ffts]
    # loss_types = {'ssim_time': [{}],
    #               'ssim_mel': n_fft_kwargs,
    #               'l1_mel': n_fft_kwargs,
    #               'l2_mel': n_fft_kwargs,
    #               'l1_fft': n_fft_kwargs,
    #               'l2_fft': n_fft_kwargs}
    loss_types = {'ssim_mel': n_fft_kwargs,
                  'l1_mel': n_fft_kwargs,
                  'l2_mel': n_fft_kwargs,
                  'l1_fft': n_fft_kwargs,
                  'l2_fft': n_fft_kwargs}
    n_losses = sum([len(v) for v in loss_types.values()])
    n_experiments = len(widths) * len(depths) * len(filter_lengths) * len(signals) * n_losses
    n_done = 0
    start_time = time.time()
    for width in widths[::-1]:
        for depth in depths[::-1]:
            for filter_length in filter_lengths[::-1]:
                for signal in signals:
                    for loss_type, loss_kwargs in loss_types.items():
                        for loss_kwarg in loss_kwargs:
                            print(loss_kwarg)
                            exp_name = 'model_%s_width%i_depth%i_filterlength%i_%s_%s' % (
                                signal, width, depth, filter_length, loss_type, re.sub(r"[ :'{}]", '', str(loss_kwarg)))
                            if skip_existing and exp_name in results:
                                print('already run %s. skipping' % exp_name)
                                n_experiments -= 1
                                continue
                            print(exp_name)
                            val_loss = train_model(train_signal=signal, exp_name=exp_name, width=width,
                                                   depth=depth, filter_length=filter_length,
                                                   loss_type=loss_type, loss_kwargs=loss_kwarg)
                            results[exp_name] = val_loss
                            with open(results_json, 'w') as f:
                                json.dump(results, f)
                            n_done += 1
                            eta = (time.time() - start_time) / n_done * (n_experiments - n_done) / 60.
                            hours = int(eta // 60)
                            minutes = int(eta % 60)
                            print('--------- Completed experiment %i of %i. ETA: %i:%i'
                                  % (n_done, n_experiments, hours, minutes))

    print('Best result: %.4f validation loss in experiment %s' % (min(results.values()), min(results, key=results.get)))


def hyperparam_search_wavenet(skip_existing=True):
    """
    TODO: find a better loss function by training a bunch of networks on different losses and listening
    TODO: investigate other training signals, including DI files, longer training signals, etc.
    """
    results_json = os.path.join(BASE_DIR, 'hyperparams_losses.json')
    if os.path.isfile(results_json) and skip_existing:
        with open(results_json, 'r') as f:
            results = json.load(f)
    else:
        results = {}
    train_set = ['di', 'all']
    widths = [8, 16]
    filter_lengths = [3, 16]
    n_ffts = [2048, 4096, 4096*2, 4096*4]
    n_fft_kwargs = [{'n_ffts': n_fft} for n_fft in n_ffts]
    loss_types = {'l2': [{}],
                  'ssim_time': [{}],
                  'ssim_mel': n_fft_kwargs,
                  'l1_mel': n_fft_kwargs,
                  'l2_mel': n_fft_kwargs,
                  'l1_fft': n_fft_kwargs,
                  'l2_fft': n_fft_kwargs}
    n_losses = sum([len(v) for v in loss_types.values()])
    n_experiments = len(widths) * len(filter_lengths) * n_losses
    n_done = 0
    start_time = time.time()
    for signal in train_set:
        for width in widths:
            for filter_length in filter_lengths:
                for loss_type, loss_kwargs in loss_types.items():
                    for loss_kwarg in loss_kwargs:
                        exp_name = 'model_%s_width%i_filterlength%i_%s_%s' % (
                            signal, width, filter_length, loss_type, re.sub(r"[ :'{}]", '', str(loss_kwarg)))
                        if skip_existing and exp_name in results:
                            print('already run %s. skipping' % exp_name)
                            n_experiments -= 1
                            continue
                        print(exp_name)
                        val_loss = train_model(exp_name=exp_name, width=width, filter_length=filter_length,
                                               loss_type=loss_type, loss_kwargs=loss_kwarg, train_signal=signal)
                        results[exp_name] = val_loss
                        with open(results_json, 'w') as f:
                            json.dump(results, f)
                        n_done += 1
                        eta = (time.time() - start_time) / n_done * (n_experiments - n_done) / 60.
                        hours = int(eta // 60)
                        minutes = int(eta % 60)
                        print('--------- Completed experiment %i of %i. ETA: %i:%i'
                              % (n_done, n_experiments, hours, minutes))

    print('Best result: %.4f validation loss in experiment %s' % (min(results.values()), min(results, key=results.get)))


def hyperparam_search_melfilter(skip_existing=True):
    """
    try the fixed filter idea, compare with the wavenet results
    """
    results_json = os.path.join(BASE_DIR, 'hyperparams_losses_melfilter.json')
    if os.path.isfile(results_json) and skip_existing:
        with open(results_json, 'r') as f:
            results = json.load(f)
    else:
        results = {}
    widths = [10, 20]
    depths = [5, 7, 9]
    filter_lengths = [32, 64]  # 64 is better than 256
    signals = ['di']
    n_ffts = [8192]
    n_fft_kwargs = [{'n_ffts': n_fft} for n_fft in n_ffts]
    loss_types = {'l1_mel': n_fft_kwargs}
    n_losses = sum([len(v) for v in loss_types.values()])
    biases = [True]  # definitely need bias
    n_experiments = len(widths) * len(depths) * len(filter_lengths) * len(signals) * n_losses * len(biases)
    n_done = 0
    start_time = time.time()
    for width in widths[::-1]:
        for depth in depths[::-1]:
            for filter_length in filter_lengths[::-1]:
                for signal in signals:
                    for loss_type, loss_kwargs in loss_types.items():
                        for loss_kwarg in loss_kwargs:
                            for bias in biases:
                                print(loss_kwarg)
                                exp_name = 'melfilter_bias%s_%s_width%i_depth%i_filterlength%i_%s_%s' % (
                                    str(bias), signal, width, depth, filter_length, loss_type,
                                    re.sub(r"[ :'{}]", '', str(loss_kwarg)))
                                if skip_existing and exp_name in results:
                                    print('already run %s. skipping' % exp_name)
                                    n_experiments -= 1
                                    continue
                                print(exp_name)
                                model_kwargs = {'width': width, 'depth': depth, 'filter_length': filter_length,
                                                'bias': bias}
                                val_loss = train_model(exp_name=exp_name, modeltype='fixedfilters',
                                                       model_kwargs=model_kwargs, loss_type=loss_type,
                                                       loss_kwargs=loss_kwarg, train_signal=signal)
                                results[exp_name] = val_loss
                                with open(results_json, 'w') as f:
                                    json.dump(results, f)
                                n_done += 1
                                eta = (time.time() - start_time) / n_done * (n_experiments - n_done) / 60.
                                hours = int(eta // 60)
                                minutes = int(eta % 60)
                                print('--------- Completed experiment %i of %i. ETA: %i:%i'
                                      % (n_done, n_experiments, hours, minutes))

    print('Best result: %.4f validation loss in experiment %s' % (min(results.values()), min(results, key=results.get)))
