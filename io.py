import pickle

import torch


def load_pkl_file(filepath):
    """Load the contents of a pickle file.

        Args:
            filepath {str} -- path to the pickle file to load.

        Returns:
            object -- the object stored in the pickle file.
        """
    with open(filepath, 'rb') as pkl_fp:
        return pickle.load(pkl_fp)


def dump_pkl_file(obj, filepath):
    """ Dumps an object into a pickle file.

    Args:
        obj {object} -- object to pickle
        filepath {str} -- destination path for the pickle file.
    """
    with open(filepath, 'wb') as fp:
        pickle.dump(obj, fp)


def batch_to_device(batch, device):
    """Take a multi input batch and send it to a pytorch device.

    Args:
        batch {Iterable(torch.Tensor)} -- A batch of multiple inputs i.e. a list of single input batches.
        device {torch.device} -- a pytorch device to send the batch to.

    Returns:

    """
    return [y.to(device) for y in batch]


def get_cuda_device():
    """Get the pytorch cuda device if available.

    Returns: A cuda device if available, cpu device else.

    """
    if torch.cuda.is_available():
        # cuda:0 will still use all GPUs
        device = torch.device('cuda:0')
        dev_name = torch.cuda.get_device_name(torch.cuda.current_device())
        print('using {} device(s): "{}"'.format(torch.cuda.device_count(), dev_name))
    else:
        device = torch.device('cpu')

    return device
