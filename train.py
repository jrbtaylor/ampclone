import json
import glob
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
IR_FILE = BASE_DIR+'IR/OH_212_ORNG_V30+M25_PROG-05_32bit_44100.wav'
FS = 44100

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


def get_dataloader_prefiltered(input_paths, label_path, batchsize=1, shuffle=False):
    label_signal, label_rate = utils.load_wav(label_path)
    input_signals = []
    for input_path in input_paths:
        input_signal, signal_rate = utils.load_wav(input_path)
        print((np.nanmax(input_signal), np.nanmin(input_signal), np.nanmean(input_signal), np.nanstd(input_signal)))
        assert (signal_rate == label_rate)
        assert np.all(np.isfinite(input_signal))
        assert np.all(np.less(np.abs(input_signal), 100))
        input_signals.append(input_signal)

    # truncate label if it's somehow longer (reaper might've padded it)
    label_signal = label_signal[:input_signals[0].size]

    labels = utils.split_signal(label_signal)
    input_signals = [utils.split_signal(signal) for signal in input_signals]
    input_signals = np.concatenate(input_signals, axis=1)  # batches x freq_bands x split_length
    print(input_signals.shape)

    inputs = torch.Tensor(input_signals)
    labels = torch.Tensor(labels)
    inputs = inputs.to(DEVICE)
    labels = labels.to(DEVICE)
    dataset = TensorDataset(inputs, labels)
    loader = DataLoader(dataset, batch_size=batchsize, shuffle=shuffle)
    return loader


def train_model(exp_name='model', amp='jcm800', modeltype='blender', model_kwargs=None,
                batchsize=1, n_epochs=1000, early_stopping=30, learnrate=1e-2, grad_clip=5.,
                loss_type=None, loss_kwargs=None, prefiltered=False, apply_ir=True):
    if amp not in exp_name:
        exp_name = amp + '_' + exp_name

    if apply_ir:
        ir, ir_fs = utils.load_wav(IR_FILE)
        assert ir_fs == FS
        print('IR shape: '+str(ir.shape))

    if prefiltered:
        data_dir = os.path.join(BASE_DIR, 'data_%i' % FS)
        train_signal = sorted(glob.glob(os.path.join(data_dir, 'prefilter', 'train_di*')))
        train_label = os.path.join(data_dir, 'train_di_' + amp + '.wav')
        val_signal = sorted(glob.glob(os.path.join(data_dir, 'prefilter', 'validation*')))
        val_label = os.path.join(data_dir, 'validation_' + amp + '.wav')
        train_loader = get_dataloader_prefiltered(train_signal, train_label, batchsize=batchsize, shuffle=True)
        val_loader = get_dataloader_prefiltered(val_signal, val_label, batchsize=batchsize, shuffle=False)
    else:
        data_dir = os.path.join(BASE_DIR, 'data_%i' % FS)
        train_signal = os.path.join(data_dir, 'train_di.wav')
        train_label = os.path.join(data_dir, 'train_di_' + amp + '.wav')
        val_signal = os.path.join(data_dir, 'validation.wav')
        val_label = os.path.join(data_dir, 'validation_' + amp + '.wav')
        train_loader = get_dataloader(train_signal, train_label, batchsize=batchsize, shuffle=True)
        val_loader = get_dataloader(val_signal, val_label, batchsize=batchsize, shuffle=False)

    savedir = os.path.join(BASE_DIR, 'models_%i' % FS, exp_name)
    if not os.path.isdir(savedir):
        os.makedirs(savedir)
    metrics_json = os.path.join(savedir, 'metrics.json')

    model_kwargs = {} if model_kwargs is None else model_kwargs
    net = model.get_model(modeltype, model_kwargs)
    net.to(DEVICE)

    criterion = losses.Loss(loss_type, loss_kwargs)
    optimizer = optim.Adam(net.parameters(), lr=learnrate)
    # TODO: debug filtering lists of parameters to train some with lower learning rates
    # if hasattr(net, 'slow_parameters'):
    #     regular_parameters = [param for param in net.parameters() if param not in net.slow_parameters]
    #     optimizer = optim.Adam([
    #         {'params': net.slow_parameters, 'lr': 1e-2 * learnrate},
    #         {'params': regular_parameters, 'lr': learnrate}
    #     ])
    # else:
    #     optimizer = optim.Adam(net.parameters(), lr=learnrate)

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.95)

    train_losses = []
    val_losses = []
    best_val = [0, np.inf, None]
    for epoch in range(n_epochs):

        print([(np.min(layer.weight.data.cpu().numpy()), np.max(layer.weight.data.cpu().numpy()))
               for layer in net.recombines])
        if hasattr(net, 'blends'):
            print(np.stack([b.data.cpu().numpy() for b in net.blends]))

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
            # print('\n' + str([(np.min(layer.weight.data.cpu().numpy()), np.max(layer.weight.data.cpu().numpy()))
            #                   for layer in net.recombines]))
            if not np.isfinite(loss.item()):
                utils.clear_print('NaN loss. Closing experiment.')
                if epoch < early_stopping:
                    return np.inf
                return best_val[1]
        utils.clear_print('Epoch %i: training loss = %.4f' % (epoch, np.mean(epoch_losses)), end='\n')
        train_losses.append(np.mean(epoch_losses))

        scheduler.step()

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
            val_outputs = np.concatenate(val_outputs).reshape([-1])
            best_val = [epoch, val_loss, val_outputs]
            saveto = os.path.join(savedir, exp_name + '.pth')
            torch.save(net.state_dict(), saveto)
        elif epoch - best_val[0] >= early_stopping:
            break

    if apply_ir:
        best_val[-1] = np.convolve(best_val[-1], ir)
    utils.save_wav(best_val[-1], os.path.join(savedir, 'val_output.wav'))
    return best_val[1]


def hyperparam_search(skip_existing=True):
    results_json = os.path.join(BASE_DIR, 'results_fir44100.json')
    if os.path.isfile(results_json) and skip_existing:
        with open(results_json, 'r') as f:
            results = json.load(f)
    else:
        results = {}
    models = ['blender']
    depths = [5, 7, 9]
    widths = [20, 30, 40]
    fmins = [40, 80]
    fmaxs = [16000, 18000]
    amps = ['5150', 'twin']  # ['5150', 'ac30', 'jcm800', 'plexi', 'recto', 'twin']
    n_ffts = [8192 * 2]  # [8192, 8192 * 2]
    time_weights = [0.1]
    loss_types = {'blend': [{'n_ffts': n_fft, 'time_weight': time_weight}
                            for n_fft in n_ffts for time_weight in time_weights]}
    n_losses = sum([len(v) for v in loss_types.values()])
    n_experiments = len(models) * len(depths) * len(amps) * n_losses * len(widths)
    n_done = 0
    start_time = time.time()
    for modeltype in models:
        for amp in amps:
            for depth in depths[::-1]:
                for width in widths[::-1]:
                    for fmin in fmins:
                        for fmax in fmaxs:
                            for loss_type, loss_kwargs in loss_types.items():
                                for loss_kwarg in loss_kwargs:
                                    exp_name = '%s_%s_d%i_w%i_fmin%i_fmax%ik_%s_%s' % (
                                        amp, modeltype, depth, width, fmin, fmax//1000, re.sub(r"[_]", '', loss_type),
                                        re.sub(r"[ :'{,_}]", '', str(loss_kwarg)))
                                    if skip_existing and exp_name in results:
                                        print('already run %s. skipping' % exp_name)
                                        n_experiments -= 1
                                        continue
                                    print(exp_name)
                                    model_kwargs = {'depth': depth, 'width': width, 'f_min': fmin, 'f_max': fmax,
                                                    'fs': FS}
                                    failed_experiments = 0
                                    lr = 1e-2
                                    while failed_experiments < 5:
                                        val_loss = train_model(amp=amp, exp_name=exp_name, modeltype=modeltype,
                                                               model_kwargs=model_kwargs, loss_type=loss_type,
                                                               loss_kwargs=loss_kwarg, learnrate=lr)
                                        if np.isfinite(val_loss):
                                            break
                                        else:
                                            print('NaN/Inf loss. Re-running with lower learning rate')
                                            failed_experiments += 1
                                            lr = lr / 10.
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
