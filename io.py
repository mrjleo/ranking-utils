import pickle


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
