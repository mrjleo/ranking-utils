import os
import csv

import torch
from tqdm import tqdm

from qa_utils.misc import Logger


def train_model_bce(model, train_dl, optimizer, args, device):
    """Train a model using binary cross entropy. Save the model after each epoch and log the loss in
    a file.

    Arguments:
        model {torch.nn.Module} -- The model to train
        train_dl {torch.utils.data.DataLoader} -- Train dataloader
        optimizer {torch.optim.Optimizer} -- Optimizer
        args {argparse.Namespace} -- All command line arguments
        device {torch.device} -- Device to train on
    """
    ckpt_dir = os.path.join(args.working_dir, 'ckpt')
    log_file = os.path.join(args.working_dir, 'train.csv')
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = Logger(log_file, ['epoch', 'loss'])

    # save all args in a file
    args_file = os.path.join(args.working_dir, 'args.csv')
    print('writing {}...'.format(args_file))
    with open(args_file, 'w') as fp:
        writer = csv.writer(fp)
        for arg in vars(args):
            writer.writerow([arg, getattr(args, arg)])

    criterion = torch.nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(args.epochs):
        loss_sum = 0
        optimizer.zero_grad()
        for i, (b_x, b_y) in enumerate(tqdm(train_dl, desc='epoch {}'.format(epoch + 1))):
            out = model(b_x.to(device))
            loss = criterion(out, b_y.to(device)) / args.accumulate_batches
            loss.backward()
            if (i + 1) % args.accumulate_batches == 0:
                optimizer.step()
                optimizer.zero_grad()
            loss_sum += loss.item()

        epoch_loss = loss_sum / len(train_dl)
        print('epoch {} -- loss: {}'.format(epoch + 1, epoch_loss))
        logger.log([epoch + 1, epoch_loss])

        state = {'epoch': epoch + 1, 'state_dict': model.module.state_dict(),
                 'optimizer': optimizer.state_dict()}
        fname = os.path.join(ckpt_dir, 'weights_{:03d}.pt'.format(epoch + 1))
        print('saving {}...'.format(fname))
        torch.save(state, fname)


def train_model_multi_bce(model, train_dl, optimizer, args, device):
    """Train a model with multiple outputs using binary cross entropy. Save the model after each
    epoch and log the loss in a file.

    Arguments:
        model {torch.nn.Module} -- The model to train
        train_dl {torch.utils.data.DataLoader} -- Train dataloader
        optimizer {torch.optim.Optimizer} -- Optimizer
        args {argparse.Namespace} -- All command line arguments
        device {torch.device} -- Device to train on
    """
    ckpt_dir = os.path.join(args.working_dir, 'ckpt')
    log_file = os.path.join(args.working_dir, 'train.csv')
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = Logger(log_file, ['epoch', 'loss'])

    # save all args in a file
    args_file = os.path.join(args.working_dir, 'args.csv')
    print('writing {}...'.format(args_file))
    with open(args_file, 'w') as fp:
        writer = csv.writer(fp)
        for arg in vars(args):
            writer.writerow([arg, getattr(args, arg)])

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

        optimizer.zero_grad()
        for i, (b_x, b_y) in enumerate(tqdm(train_dl, desc='epoch {}'.format(epoch + 1))):
            out = model(b_x.to(device))
            loss = sum(criterion(b_o, b_y.to(device)) for b_o in out) / args.accumulate_batches
            loss.backward()
            if (i + 1) % args.accumulate_batches == 0:
                optimizer.step()
                optimizer.zero_grad()
            loss_sum += loss.item()

        epoch_loss = loss_sum / len(train_dl)
        print('epoch {} -- loss: {}'.format(epoch + 1, epoch_loss))
        logger.log([epoch + 1, epoch_loss])

        state = {'epoch': epoch + 1, 'state_dict': model.module.state_dict(),
                 'optimizer': optimizer.state_dict()}
        fname = os.path.join(ckpt_dir, 'weights_{:03d}.pt'.format(epoch + 1))
        print('saving {}...'.format(fname))
        torch.save(state, fname)
