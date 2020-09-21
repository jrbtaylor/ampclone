import json
import os
import re
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import model
import losses
import utils
from dataloader import AudioDataset
from config import BASE_DIR, IR_FILE, FS, DEVICE, SPLIT_LENGTH, VAL_SPLIT_LENGTH


def get_dataloader(input_path, label_path, batchsize=1, is_training=False):
    input_signal, signal_rate = utils.load_wav(input_path)
    label_signal, label_rate = utils.load_wav(label_path)

    assert (signal_rate == label_rate)
    # truncate label if it's somehow longer (reaper might've padded it)
    label_signal = label_signal[:input_signal.size]

    print(input_signal.size)
    inputs = torch.Tensor(input_signal)
    labels = torch.Tensor(label_signal)

    dataset = AudioDataset([inputs, labels], SPLIT_LENGTH if is_training else VAL_SPLIT_LENGTH, is_training=is_training)
    loader = DataLoader(dataset, batch_size=batchsize, shuffle=is_training)
    return loader


def train_model(exp_name='model', amp='jcm800', modeltype='blender', model_kwargs=None,
                batchsize=1, n_epochs=1000, early_stopping=30, learnrate=1e-2, grad_clip=5.,
                loss_type=None, loss_kwargs=None, apply_ir=True):
    """
    :param exp_name: experiment name (used for save directory)
    :param amp:
    :param modeltype:
    :param model_kwargs:
    :param batchsize:
    :param n_epochs:
    :param early_stopping:
    :param learnrate:
    :param grad_clip:
    :param loss_type:
    :param loss_kwargs:
    :param apply_ir: (bool) apply IR to validation output (use if IR is not applied in labels)
    :return:
    """
    if amp not in exp_name:
        exp_name = amp + '_' + exp_name

    if apply_ir:
        ir, ir_fs = utils.load_wav(IR_FILE)
        assert ir_fs == FS
        print('IR shape: ' + str(ir.shape))

    data_dir = os.path.join(BASE_DIR, 'data_%i' % FS)
    train_signal = os.path.join(data_dir, 'train_di.wav')
    train_label = os.path.join(data_dir, 'train_di_' + amp + '.wav')
    val_signal = os.path.join(data_dir, 'validation.wav')
    val_label = os.path.join(data_dir, 'validation_' + amp + '.wav')
    train_loader = get_dataloader(train_signal, train_label, batchsize=batchsize, is_training=True)
    val_loader = get_dataloader(val_signal, val_label, batchsize=batchsize, is_training=False)

    savedir = os.path.join(BASE_DIR, 'models_%i' % FS, exp_name)
    if not os.path.isdir(savedir):
        os.makedirs(savedir)
    metrics_json = os.path.join(savedir, 'metrics.json')

    model_kwargs = {} if model_kwargs is None else model_kwargs
    net = model.get_model(modeltype, model_kwargs)
    net.to(DEVICE)

    # Regularizers:
    phase_loss = False
    if 'learn_filters' in model_kwargs:
        if model_kwargs['learn_filters']:
            phase_loss = True

    criterion = losses.Loss(loss_type, loss_kwargs)
    if hasattr(net, 'slow_parameters'):
        regular_parameters = [param for param in net.parameters() if param not in set(net.slow_parameters)]
        optimizer = optim.Adam([
            {'params': net.slow_parameters, 'lr': 1e-4 * learnrate},
            {'params': regular_parameters, 'lr': learnrate}
        ])
        warmup_optimizer = optim.Adam(regular_parameters, lr=learnrate)
    else:
        optimizer = optim.Adam(net.parameters(), lr=learnrate)
        warmup_optimizer = optimizer

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.95)

    metrics_fn = losses.get_eq_metrics_fn()

    train_losses = []
    val_losses = []
    best_val = [0, np.inf, None]
    warmup_epochs = 10
    for epoch in range(n_epochs):

        # print([(np.min(layer.weight.data.cpu().numpy()), np.max(layer.weight.data.cpu().numpy()))
        #        for layer in net.recombines])
        # if hasattr(net, 'blends'):
        #     print(np.stack([b.data.cpu().numpy() for b in net.blends]))

        epoch_losses = []
        epoch_metrics = []
        start_time = time.time()
        optmzr = optimizer if epoch > warmup_epochs else warmup_optimizer
        for i, data in enumerate(train_loader):
            inputs, labels = data
            optmzr.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            metrics = metrics_fn(outputs, labels)
            if phase_loss:
                phs_lss = losses.phase_loss(net)
                # loss += 1e-7*phs_lss  # TODO: debug NaN issue with phase loss optimization -------------------------------
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
            optmzr.step()

            epoch_losses.append(loss.item())
            epoch_metrics.append(metrics.cpu().detach().numpy())
            elapsed = time.time() - start_time
            if phase_loss:
                utils.clear_print(
                    'training step %i    loss = %.4f    phase = %.4f    elapsed = %.2f minutes    ETA = %.2f minutes'
                    % (i, loss.item(), phs_lss.item(), elapsed / 60,
                       (len(train_loader) - i - 1) * elapsed / (i + 1) / 60))
            else:
                utils.clear_print('training step %i    loss = %.4f    elapsed = %.2f minutes    ETA = %.2f minutes'
                                  % (i, loss.item(), elapsed / 60,
                                     (len(train_loader) - i - 1) * elapsed / (i + 1) / 60))
            # print('\n' + str([(np.min(layer.weight.data.cpu().numpy()), np.max(layer.weight.data.cpu().numpy()))
            #                   for layer in net.recombines]))
            if not np.isfinite(loss.item()):
                utils.clear_print('NaN loss. Closing experiment.')
                if epoch < early_stopping:
                    return np.inf
                return best_val[1]
        if phase_loss:
            utils.clear_print('Epoch %i: training loss = %.4f     phase loss = %.4f'
                              % (epoch, np.mean(epoch_losses), phs_lss.item()), end='\n')
        else:
            utils.clear_print('Epoch %i: training loss = %.4f' % (epoch, np.mean(epoch_losses)), end='\n')
        utils.clear_print(np.mean(np.stack(epoch_metrics, axis=0), axis=0), end='\n')
        train_losses.append(np.mean(epoch_losses))

        if epoch > warmup_epochs:
            scheduler.step()

        epoch_losses = []
        epoch_metrics = []
        start_time = time.time()
        val_outputs = []
        for i, data in enumerate(val_loader):
            inputs, labels = data
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            metrics = metrics_fn(outputs, labels)
            val_outputs.append(outputs.data.cpu().numpy())

            epoch_losses.append(loss.item())
            epoch_metrics.append(metrics.cpu().detach().numpy())
            elapsed = time.time() - start_time
            utils.clear_print('validation step %i    loss = %.4f    elapsed = %.2f minutes    ETA = %.2f minutes'
                              % (i, loss.item(), elapsed / 60, (len(val_loader) - i - 1) * elapsed / (i + 1) / 60))
        utils.clear_print('Epoch %i: validation loss = %.4f' % (epoch, np.mean(epoch_losses)), end='\n')
        utils.clear_print(np.mean(np.stack(epoch_metrics, axis=0), axis=0), end='\n')
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
    results_json = os.path.join(BASE_DIR, 'results_fir44100_softsign.json')
    if os.path.isfile(results_json) and skip_existing:
        with open(results_json, 'r') as f:
            results = json.load(f)
    else:
        results = {}
    models = ['blender']
    depths = [3, 5, 7]  # [3, 5, 7, 9]
    widths = [20, 30, 40]  # [20, 30, 40]
    fmins = [40]
    fmaxs = [16000]
    offsets = [1000.]
    amps = ['twin', '5150', 'plexi', 'jcm800', 'ac30', 'recto']
    windows = ['flattop', 'blackmanharris', 'hann']
    n_ffts = [4096]  # RULE OF THUMB: MINIMUM FREQ = 5 * FS / N_FFT
    time_weights = [0.1]
    # loss_types = {'blend': [{'n_ffts': n_fft, 'time_weight': time_weight}
    #                         for n_fft in n_ffts for time_weight in time_weights],
    #               'l2_mel': [{'n_ffts': n_fft} for n_fft in n_ffts]}
    loss_types = {'blend': [{'n_ffts': n_fft, 'time_weight': time_weight}
                            for n_fft in n_ffts for time_weight in time_weights]}
    n_losses = sum([len(v) for v in loss_types.values()])
    n_experiments = len(models) * len(depths) * len(amps) * n_losses * len(widths) * len(fmins) * len(fmaxs) \
                    * len(windows) * len(offsets)
    n_done = 0
    start_time = time.time()
    for modeltype in models:
        for amp in amps:
            for depth in depths[::-1]:
                for width in widths[::-1]:
                    for fmin in fmins:
                        for fmax in fmaxs:
                            for window in windows:
                                for offset in offsets:
                                    for loss_type, loss_kwargs in loss_types.items():
                                        for loss_kwarg in loss_kwargs:
                                            exp_name = '%s_%s_d%i_w%i_%s_fmin%i_fmax%ik_offset%i_%s_%s_softsign' % (
                                                amp, modeltype, depth, width, window,
                                                fmin, fmax // 1000, offset,
                                                re.sub(r"[_]", '', loss_type),
                                                re.sub(r"[ :'{,_}]", '', str(loss_kwarg)))
                                            if skip_existing and exp_name in results:
                                                print('already run %s. skipping' % exp_name)
                                                n_experiments -= 1
                                                continue
                                            print(exp_name)
                                            model_kwargs = {'depth': depth, 'width': width, 'f_min': fmin,
                                                            'f_max': fmax, 'fs': FS, 'window': window,
                                                            'offset': offset, 'learn_filters': False,
                                                            'activation': 'softsign'}
                                            failed_experiments = 0
                                            lr = 1e-2  # was 1e-2 before using learned filters
                                            while failed_experiments < 5:
                                                val_loss = train_model(
                                                    amp=amp, exp_name=exp_name, modeltype=modeltype,
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
                                            eta = (time.time() - start_time) / n_done * (
                                                        n_experiments - n_done) / 60.
                                            hours = int(eta // 60)
                                            minutes = int(eta % 60)
                                            print('--------- Completed experiment %i of %i. ETA: %i:%i'
                                                  % (n_done, n_experiments, hours, minutes))

    print('Best result: %.4f validation loss in experiment %s' % (min(results.values()), min(results, key=results.get)))
