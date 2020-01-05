import csv
import os
from datetime import datetime

import torch
from tqdm import tqdm


class Logger(object):
    """Simple .csv file logger that supports appending a timestamp.

    Arguments:
        filename {str} -- Log file path
        header {list[str]} -- Column names

    Keyword Arguments:
        add_timestamp {bool} -- Whether to add timestamps to each row (default: {True})
    """
    def __init__(self, filename, header, add_timestamp=True):
        self.add_timestamp = add_timestamp
        self._fp = open(filename, 'w', encoding='utf-8')
        self._writer = csv.writer(self._fp)
        if add_timestamp:
            self._writer.writerow(header + ['time'])
        else:
            self._writer.writerow(header)
        self._fp.flush()

    def log(self, item):
        """Log a single item.

        Arguments:
            item {list[str]} -- The row to log
        """
        if self.add_timestamp:
            self._writer.writerow(item + [datetime.now()])
        else:
            self._writer.writerow(item)
        self._fp.flush()

    def __del__(self):
        self._fp.close()


def train_model_bce(model, train_dl, epochs, optimizer, accumulate_batches, device, working_dir):
    """Train a model using binary cross entropy. Save the model after each epoch and log the loss in
    a file.
    
    Arguments:
        model {torch.nn.Module} -- The model to tran
        train_dl {torch.utils.data.DataLoader} -- Train dataloader
        epochs {int} -- Number of epochs
        optimizer {torch.optim.Optimizer} -- Optimizer
        accumulate_batches {int} -- Update the parameters after this many batches
        device {str} -- Device to train on
        working_dir {str} -- Working directory to save all files in
    """
    ckpt_dir = os.path.join(working_dir, 'ckpt')
    log_file = os.path.join(working_dir, 'train.csv')
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = Logger(log_file, ['epoch', 'loss'])

    criterion = torch.nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(epochs):
        loss_sum = 0
        optimizer.zero_grad()
        for i, (b_x, b_y) in enumerate(tqdm(train_dl, desc='epoch {}'.format(epoch + 1))):
            out = model(b_x)
            loss = criterion(out, b_y.to(device)) / accumulate_batches
            loss.backward()
            if (i + 1) % accumulate_batches == 0:
                optimizer.step()
                optimizer.zero_grad()
            loss_sum += loss.item()

        epoch_loss = loss_sum / len(train_dl)
        print('epoch {} -- loss: {}'.format(epoch + 1, epoch_loss))
        logger.log([epoch + 1, epoch_loss])

        state = {'epoch': epoch + 1, 'state_dict': model.module.state_dict(), 'optimizer': optimizer.state_dict()}
        fname = os.path.join(ckpt_dir, 'weights_{:03d}.pt'.format(epoch + 1))
        print('saving {}...'.format(fname))
        torch.save(state, fname)


def save_args(args, working_dir):
    """Save all arguments in a csv file.
    
    Arguments:
        args {argparse.Namespace} -- The arguments to save
        working_dir {str} -- The directory where the file will be saved
    """
    os.makedirs(working_dir, exist_ok=True)
    params_file = os.path.join(working_dir, 'params.csv')
    print('writing {}...'.format(params_file))
    with open(params_file, 'w') as fp:
        writer = csv.writer(fp)
        for arg in vars(args):
            writer.writerow([arg, getattr(args, arg)])
