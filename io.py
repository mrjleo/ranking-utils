import pickle

import torch


def load_pkl_file(filepath):
    """Load the contents of a pickle file.

        Arguments:
            filepath {str} -- path to the pickle file to load

        Returns:
            object -- the object stored in the pickle file
        """
    with open(filepath, 'rb') as pkl_fp:
        return pickle.load(pkl_fp)


def dump_pkl_file(obj, filepath):
    """Dump an object into a pickle file.

    Arguments:
        obj {object} -- object to pickle
        filepath {str} -- destination path for the pickle file
    """
    with open(filepath, 'wb') as fp:
        pickle.dump(obj, fp)


def list_to(device, x):
    """Take a multi input batch and send it to a pytorch device.

    Arguments:
        device {torch.device} -- a pytorch device to send the list to
        x {list(torch.Tensor)} -- A list of single input batches

    Returns:
        list[torch.Tensor] -- The tensors
    """
    return [y.to(device) for y in x if y is not None]


def list_or_tensor_to(device, x):
    """Sends a tensor or a list of tensors to `device`

    Args:
        Arguments:
        device {torch.device} -- a pytorch device to send the list to
        x {list(torch.Tensor) or torch.Tensor} -- A list of single input batches

    Returns:

    """
    if isinstance(x, list):
        return list_to(device, x)
    elif isinstance(x, torch.Tensor):
        return x.to(device)
    raise TypeError(f'{type(x)} is not supported for sending to a pytorch device.')


def get_cuda_device():
    """Get the pytorch cuda device if available.

    Returns:
        {torch.device} -- A cuda device if available, otherwise cpu
    """
    if torch.cuda.is_available():
        # cuda:0 will still use all GPUs
        device = torch.device('cuda:0')
        dev_name = torch.cuda.get_device_name(torch.cuda.current_device())
        print('using {} device(s): "{}"'.format(torch.cuda.device_count(), dev_name))
    else:
        device = torch.device('cpu')
    return device
