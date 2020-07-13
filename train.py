import json
import os
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


def train_model(train_signal='chirp',
                exp_name='model_chirp',
                batchsize=1, n_epochs=1000, early_stopping=10,
                width=16, depth=3, filter_length=512,
                learnrate=1e-4, momentum=0.5, grad_clip=0.5):
    if train_signal == 'chirp':
        train_signal = BASE_DIR + 'train2.wav'
        train_label = BASE_DIR + 'train2_2204_212.wav'
    elif train_signal == 'noise':
        train_signal = BASE_DIR + 'train.wav'
        train_label = BASE_DIR + 'train_2204_212.wav'
    else:
        raise ValueError
    val_signal = BASE_DIR + 'validation.wav'
    val_label = BASE_DIR + 'validation_2204_212.wav'

    train_loader = get_dataloader(train_signal, train_label, batchsize=batchsize, shuffle=True)
    val_loader = get_dataloader(val_signal, val_label, batchsize=batchsize, shuffle=False)

    savedir = os.path.join(BASE_DIR, 'models', exp_name)
    if not os.path.isdir(savedir):
        os.makedirs(savedir)
    metrics_json = os.path.join(savedir, 'metrics.json')

    net = model.SimpleCNN(width=width, depth=depth, filter_length=filter_length)
    net.to(DEVICE)

    criterion = losses.Loss()
    # optimizer = optim.SGD(net.parameters(), lr=learnrate, momentum=momentum)
    # optimizer = optim.Adam(net.parameters(), lr=learnrate)
    optimizer = optim.Adam([
        {'params': net.layers[-1].parameters(), 'lr': 1e-2 * learnrate},
        {'params': [param for i in range(len(net.layers) - 1) for param in net.layers[i].parameters()], 'lr': learnrate}
    ])

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
            val_outputs.append(outputs)

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

            val_outputs = np.concatenate([v.data.cpu().numpy() for v in val_outputs])
            utils.save_wav(val_outputs, os.path.join(savedir, 'val_output_%i.wav' % epoch))
        elif epoch - best_val[0] >= early_stopping:
            break
    return best_val[1]


def hyperparam_search():
    widths = [1] + list(range(4, 20, 4))
    depths = range(3, 9, 2)
    filter_lengths = [128, 256, 512, 1024]
    signals = ['chirp', 'noise']
    results = {}
    n_experiments = len(widths) * len(depths) * len(filter_lengths) * len(signals)
    n_done = 0
    start_time = time.time()
    for width in widths[::-1]:
        for depth in depths[::-1]:
            for filter_length in filter_lengths[::-1]:
                for signal in signals:
                    exp_name = 'model_%s_width%i_depth%i_filterlength%i' % (signal, width, depth, filter_length)
                    val_loss = train_model(train_signal=signal, exp_name=exp_name, width=width,
                                           depth=depth, filter_length=filter_length)
                    results[exp_name] = val_loss
                    with open(os.path.join(BASE_DIR, 'hyperparams.json'), 'w') as f:
                        json.dump(results, f)
                    n_done += 1
                    eta = (time.time() - start_time) / n_done * (n_experiments - n_done) / 60.
                    hours = int(eta // 60)
                    minutes = int(eta % 60)
                    print('--------- Completed experiment %i of %i. ETA: %i:%i'
                          % (n_done, n_experiments, hours, minutes))

    print('Best result: %.4f validation loss in experiment %s' % (min(results.values()), min(results, key=results.get)))
