import csv
import os

import torch
from tqdm import tqdm

from qa_utils.io import batches_to_device
from qa_utils.misc import Logger


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


def save_checkpoint(state, epoch, ckpt_dir):
    """Takes a dict containing training state and saves it to a checkpoint file in ckpt_dir.

    Args:
        state {dict} -- dict containing checkpoint information e.g. pytorch state dicts of model and optimizer.
        epoch {int} -- epoch at which the state was taken.
        ckpt_dir {str} -- path to checkpoint directory.
    """
    fname = os.path.join(ckpt_dir, 'weights_{:03d}.pt'.format(epoch))
    torch.save(state, fname)


def prepare_logging(args):
    """Creates the checkpoint directory, a logger and exports the training arguments.

    Args:
        args {argparse.Namespace} -- All command line arguments

    Returns:
        logger {misc.Logger} -- Logger for working directory

    """
    ckpt_dir = os.path.join(args.working_dir, 'ckpt')
    log_file = os.path.join(args.working_dir, 'train.csv')
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = Logger(log_file, ['epoch', 'loss'])

    args_file = os.path.join(args.working_dir, 'args.csv')
    save_args(args_file, args)

    return logger, ckpt_dir


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
    logger, ckpt_dir = prepare_logging(args)

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
        save_checkpoint(state, epoch, ckpt_dir)


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
    logger, ckpt_dir = prepare_logging(args)

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
        save_checkpoint(state, epoch, ckpt_dir)


def train_model_pairwise(model, criterion, train_dl, optimizer, args, device):
    """Trains a model in pairwise fashion by sampling negative examples with the highest loss from a list of sampled
    negative examples.

    Args:
        model {torch.nn.Module} -- the model to train
        criterion {function} -- a pairwise loss function. It is expected to handle reduction, e.g. average over the
        batch.
        train_dl {torch.utils.data.DataLoader} -- Train dataloader that yields a positive and a list of
        negative inputs with each input being a batch of examples.
        optimizer {list[torch.optim.Optimizer]} -- Optimizer
        args {argparse.Namespace} -- All command line arguments
        device {torch.device} -- Device to train on
    """
    logger, ckpt_dir = prepare_logging(args)

    args_file = os.path.join(args.working_dir, 'args.csv')
    save_args(args_file, args)

    model.train()
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        loss_sum = 0
        for i, batch in enumerate(tqdm(train_dl, desc='epoch {}'.format(epoch))):
            pos_inputs, neg_inputs = batch

            pos_inputs = pos_inputs.to(device)
            neg_inputs = batches_to_device(neg_inputs, device)

            max_neg_inputs = _sample_max_loss_neg_batch(model, criterion, pos_inputs, neg_inputs, args.pred_batch_size)
            pos_scores = model(pos_inputs)
            neg_scores = model(max_neg_inputs)

            loss = criterion(pos_scores, neg_scores)
            loss = loss / args.accumulate_batches
            loss_sum += loss.item()

            loss.backward()
            if (i + 1) % args.accumulate_batches == 0:
                optimizer.step()
                optimizer.zero_grad()

        epoch_loss = loss_sum / len(train_dl)
        logger.log([epoch, epoch_loss])

        state = {'epoch': epoch, 'batch': i, 'state_dict': model.module.state_dict(),
                 'optimizer': optimizer.state_dict()}
        save_checkpoint(state, epoch, ckpt_dir)


def _sample_max_loss_neg_batch(model, criterion, pos_inputs, neg_inputs, pred_batch_size=None):
    """Helper function for sampling negative examples with highest pairwise loss from a list of batches with negative
    input examples.

    Args:
        model {torch.nn.Module} -- model for computing the pairwise loss based on its predictions.
        criterion {} -- a pairwise loss function accepting the models predictions on `pos_inputs` and `neg_inputs` as
        inputs.
        pos_inputs {torch.Tensor} -- batch of positive examples to feed into `model`.
        neg_inputs {list(torch.Tensor)} -- list of batches containing negative examples to feed into the model.
        predict_batch_size {int} -- the maximum number of examples to run through the model in parallel when predicting.
        defaults to batch_size * len(neg_inputs) which might not fit on GPU depending on model size.

    Returns:
        neg_batch {list(torch.Tensor)} -- batch containing the negative examples with maximum loss along dimension 1 of
        `neg_inputs`.

    """
    with torch.no_grad():
        n_negs = len(neg_inputs)
        batch_size = pos_inputs.shape[0]
        pred_batch_size = n_negs * batch_size if pred_batch_size is None else pred_batch_size

        pos_preds = model(pos_inputs)
        # to be able to compute loss for multiple negative docs and the positive in parallel we expand the pos scores
        # since they're the same for each negative doc
        pos_preds = pos_preds.unsqueeze(0).expand((n_negs,) + pos_preds.shape)
        pos_preds = pos_preds.reshape((batch_size * n_negs, 1))

        # we put all negative examples into one large batch for parallel prediction
        all_negs = torch.cat(neg_inputs, dim=0)

        # in cases where not all negative documents fit into memory at the same time we need to predict on
        # slightly smaller chunks
        neg_pred_chunks = []
        for i in range(0, n_negs * batch_size, pred_batch_size):
            neg_pred_chunk = model(all_negs[i:i + pred_batch_size])
            neg_pred_chunks.append(neg_pred_chunk)

        neg_preds = torch.cat(neg_pred_chunks)
        losses = criterion(pos_preds, neg_preds)

        # to get the per batch maximum loss we need to split back into batches of the original batch size
        losses = torch.split(losses, batch_size)
        losses = torch.stack(losses, dim=1)
        # index of the highest loss negative inputs for each row in the batch
        max_loss_ids = torch.argmax(losses, dim=1)
        max_loss_batch = []
        for j, idx in enumerate(max_loss_ids):
            max_input = neg_inputs[idx][j]
            max_loss_batch.append(max_input)

        return torch.stack(max_loss_batch)
