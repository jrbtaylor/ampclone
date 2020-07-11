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

BASE_DIR = 'C:/Users/Jason/Documents/cloner/'


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
    dataset = TensorDataset(inputs, labels)
    loader = DataLoader(dataset, batch_size=batchsize, shuffle=shuffle)
    return loader


def train_model(train_signal=BASE_DIR + 'train2.wav',
                train_label=BASE_DIR + 'train2_2204_212.wav',
                val_signal=BASE_DIR + 'validation.wav',
                val_label=BASE_DIR + 'validation_2204_212.wav',
                exp_name='model_chirp',
                batchsize=1):
    train_loader = get_dataloader(train_signal, train_label, batchsize=batchsize, shuffle=True)
    val_loader = get_dataloader(val_signal, val_label, batchsize=batchsize, shuffle=False)

    savedir = os.path.join(BASE_DIR, exp_name)
    if not os.path.isdir(savedir):
        os.makedirs(savedir)
    metrics_json = os.path.join(savedir, 'metrics.json')

    net = model.SimpleCNN()

    criterion = losses.OffsetMSE()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    # TODO: add early stopping
    train_losses = []
    val_losses = []
    for epoch in range(50):
        epoch_losses = []
        start_time = time.time()
        for i, data in enumerate(train_loader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            elapsed = time.time() - start_time
            utils.clear_print('training step %i    loss = %.4f    elapsed = %.2f minutes    ETA = %.2f minutes'
                              % (i, loss.item(), elapsed / 60, (len(train_loader) - i - 1) * elapsed / (i + 1) / 60))
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
        val_losses.append(np.mean(epoch_losses))

        with open(metrics_json, 'w') as f:
            json.dump({'train_losses': train_losses, 'val_losses': val_losses}, f)

        val_outputs = np.concatenate([v.data.numpy() for v in val_outputs])
        utils.save_wav(val_outputs, os.path.join(savedir, 'val_output_%i.wav' % epoch))

        saveto = os.path.join(savedir, exp_name + '_%i.pth' % epoch)
        torch.save(net.state_dict(), saveto)
