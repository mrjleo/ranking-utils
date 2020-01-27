import os
import csv
import itertools

import torch
from tqdm import tqdm

from qa_utils.misc import Logger
from qa_utils.io import batches_to_device


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


def train_model_bce(model, train_dl, optimizer, args, device, has_multiple_inputs=False):
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
    """
    ckpt_dir = os.path.join(args.working_dir, 'ckpt')
    log_file = os.path.join(args.working_dir, 'train.csv')
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = Logger(log_file, ['epoch', 'loss'])

    args_file = os.path.join(args.working_dir, 'args.csv')
    save_args(args_file, args)

    criterion = torch.nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        loss_sum = 0
        for i, (b_x, b_y) in enumerate(tqdm(train_dl, desc='epoch {}'.format(epoch))):
            if has_multiple_inputs:
                inputs = batches_to_device(b_x, device)
                out = model(*inputs)
            else:
                out = model(b_x.to(device))
            loss = criterion(out, b_y.to(device)) / args.accumulate_batches
            loss_sum += loss.item()
            loss.backward()
            if (i + 1) % args.accumulate_batches == 0:
                optimizer.step()
                optimizer.zero_grad()

        epoch_loss = loss_sum / len(train_dl)
        logger.log([epoch, epoch_loss])

        state = {'epoch': epoch, 'batch': i, 'state_dict': model.module.state_dict(),
                 'optimizer': optimizer.state_dict()}
        fname = os.path.join(ckpt_dir, 'weights_{:03d}.pt'.format(epoch))
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

        for i, (b_x, b_y) in enumerate(tqdm(train_dl, desc='epoch {}'.format(epoch))):
            for b_o in model(b_x.to(device)):
                loss = criterion(b_o, b_y.to(device)) / args.accumulate_batches
                loss_sum += loss.item()
                loss.backward()

            if (i + 1) % args.accumulate_batches == 0:
                for optimizer in optimizers:
                    optimizer.step()
                    optimizer.zero_grad()

        epoch_loss = loss_sum / len(train_dl)
        logger.log([epoch, epoch_loss])

        state = {'epoch': epoch, 'batch': i, 'state_dict': model.module.state_dict()}
        for i, optimizer in enumerate(optimizers):
            state['optimizer_{}'.format(i)] = optimizer.state_dict()
        fname = os.path.join(ckpt_dir, 'weights_{:03d}.pt'.format(epoch))
        torch.save(state, fname)
