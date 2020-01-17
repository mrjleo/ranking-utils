import os
import csv
import itertools

import torch
from tqdm import tqdm

from qa_utils.misc import Logger
from qa_utils.io import batches_to_device
from qa_utils.evaluation import get_checkpoints


def save_args(args_file, args):
    """Save all arguments in a file.

    Arguments:
        args_file {str} -- The .csv file to save
        args {argparse.Namespace} -- Command line arguments
    """
    print('writing {}...'.format(args_file))
    with open(args_file, 'w') as fp:
        writer = csv.writer(fp)
        for arg in vars(args):
            writer.writerow([arg, getattr(args, arg)])


def train_model_bce(model, train_dl, optimizer, args, device, has_multiple_inputs=False,
                    continue_training=True):
    """Train a model using binary cross entropy. Save the model after each epoch and log the loss in
    a file.

    Arguments:
        model {torch.nn.Module} -- The model to train
        train_dl {torch.utils.data.DataLoader} -- Train dataloader
        optimizer {torch.optim.Optimizer} -- Optimizer
        args {argparse.Namespace} -- All command line arguments
        device {torch.device} -- Device to train on

    Keyword Arguments:
        has_multiple_inputs {bool} -- Whether the input is a a list of tensors (default: {False})
        continue_training {bool} -- Load the last existing checkpoint (default: {True})
    """
    ckpt_dir = os.path.join(args.working_dir, 'ckpt')
    log_file = os.path.join(args.working_dir, 'train.csv')
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = Logger(log_file, ['epoch', 'loss'], new_file=not continue_training)

    args_file = os.path.join(args.working_dir, 'args.csv')
    save_args(args_file, args)

    checkpoints = get_checkpoints(ckpt_dir, r'weights_(\d+).pt')
    if len(checkpoints) > 0 and continue_training:
        checkpoint = torch.load(checkpoints[-1])
        model.module.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        last_epoch = checkpoint['epoch']

    criterion = torch.nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(last_epoch, last_epoch + args.epochs):
        loss_sum = 0
        optimizer.zero_grad()
        for i, (b_x, b_y) in enumerate(tqdm(train_dl, desc='epoch {}'.format(epoch + 1))):
            if has_multiple_inputs:
                inputs = batches_to_device(b_x, device)
                out = model(*inputs)
            else:
                out = model(b_x.to(device))
            loss = criterion(out, b_y.to(device)) / args.accumulate_batches
            loss.backward()
            if (i + 1) % args.accumulate_batches == 0:
                optimizer.step()
                optimizer.zero_grad()
            loss_sum += loss.item()

        epoch_loss = loss_sum / len(train_dl)
        logger.log([epoch + 1, epoch_loss])

        state = {'epoch': epoch + 1, 'state_dict': model.module.state_dict(),
                 'optimizer': optimizer.state_dict()}
        fname = os.path.join(ckpt_dir, 'weights_{:03d}.pt'.format(epoch + 1))
        print('saving {}...'.format(fname))
        torch.save(state, fname)


def train_model_bce_batches(model, train_dl, optimizer, args, device, has_multiple_inputs=False,
                            continue_training=True):
    """Train a model using binary cross entropy. Save the model after a number of batches and log
    the loss in a file.

    Arguments:
        model {torch.nn.Module} -- The model to train
        train_dl {torch.utils.data.DataLoader} -- Train dataloader
        optimizer {torch.optim.Optimizer} -- Optimizer
        args {argparse.Namespace} -- All command line arguments
        device {torch.device} -- Device to train on

    Keyword Arguments:
        has_multiple_inputs {bool} -- Whether the input is a a list of tensors (default: {False})
        continue_training {bool} -- Load the last existing checkpoint (default: {True})
    """
    ckpt_dir = os.path.join(args.working_dir, 'ckpt')
    log_file = os.path.join(args.working_dir, 'train.csv')
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = Logger(log_file, ['epoch', 'loss'], new_file=not continue_training)

    args_file = os.path.join(args.working_dir, 'args.csv')
    save_args(args_file, args)

    checkpoints = get_checkpoints(ckpt_dir, r'weights_(\d+).pt')
    if len(checkpoints) > 0 and continue_training:
        checkpoint = torch.load(checkpoints[-1])
        model.module.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        last_epoch = checkpoint['epoch']

    criterion = torch.nn.BCEWithLogitsLoss()
    train_inf = itertools.cycle(train_dl)
    model.train()
    for epoch in range(last_epoch, last_epoch + args.epochs):
        loss_sum = 0
        optimizer.zero_grad()
        for i in tqdm(range(args.save_after)):
            b_x, b_y = next(train_inf)
            if has_multiple_inputs:
                inputs = batches_to_device(b_x, device)
                out = model(*inputs)
            else:
                out = model(b_x.to(device))

            loss = criterion(out, b_y.to(device)) / args.accumulate_batches
            loss.backward()

            if (i + 1) % args.accumulate_batches == 0:
                optimizer.step()
                optimizer.zero_grad()
            loss_sum += loss.item()

        epoch_loss = loss_sum / args.save_after
        logger.log([epoch + 1, epoch_loss])

        state = {'epoch': epoch + 1, 'state_dict': model.module.state_dict(),
                 'optimizer': optimizer.state_dict()}
        fname = os.path.join(ckpt_dir, 'weights_{:03d}.pt'.format(epoch + 1))
        print('saving {}...'.format(fname))
        torch.save(state, fname)


def train_model_multi_bce(model, train_dl, optimizers, args, device):
    """Train a model with multiple outputs using binary cross entropy. Save the model after each
    epoch and log the loss in a file.

    Arguments:
        model {torch.nn.Module} -- The model to train
        train_dl {torch.utils.data.DataLoader} -- Train dataloader
        optimizers {list[torch.optim.Optimizer]} -- Optimizers, one for each output
        args {argparse.Namespace} -- All command line arguments
        device {torch.device} -- Device to train on
    """
    ckpt_dir = os.path.join(args.working_dir, 'ckpt')
    log_file = os.path.join(args.working_dir, 'train.csv')
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = Logger(log_file, ['epoch', 'loss'])

    args_file = os.path.join(args.working_dir, 'args.csv')
    save_args(args_file, args)

    # load BERT weights
    if args.bert_weights is not None:
        state_dict_new = model.module.state_dict()
        state = torch.load(args.bert_weights)
        # we need to re-write all the keys as they are from a different model
        for k, v in state['state_dict'].items():
            if 'bert' in k:
                k_new = k[k.index('bert'):]
                state_dict_new[k_new] = v
        model.module.load_state_dict(state_dict_new)

    criterion = torch.nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(args.epochs):
        loss_sum = 0

        for optimizer in optimizers:
            optimizer.zero_grad()

        for i, (b_x, b_y) in enumerate(tqdm(train_dl, desc='epoch {}'.format(epoch + 1))):
            out = model(b_x.to(device))

            losses = [criterion(b_o, b_y.to(device)) / args.accumulate_batches for b_o in out]
            for loss in losses:
                loss.backward()

            if (i + 1) % args.accumulate_batches == 0:
                for optimizer in optimizers:
                    optimizer.step()
                    optimizer.zero_grad()

            for loss in losses:
                loss_sum += loss.item()

        epoch_loss = loss_sum / len(train_dl)
        print('epoch {} -- loss: {}'.format(epoch + 1, epoch_loss))
        logger.log([epoch + 1, epoch_loss])

        state = {'epoch': epoch + 1, 'state_dict': model.module.state_dict()}
        for i, optimizer in enumerate(optimizers):
            state['optimizer_{}'.format(i)] = optimizer.state_dict()
        fname = os.path.join(ckpt_dir, 'weights_{:03d}.pt'.format(epoch + 1))
        print('saving {}...'.format(fname))
        torch.save(state, fname)
