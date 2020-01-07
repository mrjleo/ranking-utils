import gzip


def count_lines(file_path):
    """Count the number of lines in a text file. Supports .gzip files.

    Arguments:
        file_path {str} -- The path to a file

    Returns:
        int -- The number of lines
    """
    if file_path.endswith('.gz'):
        with gzip.open(file_path) as fp:
            for i, _ in enumerate(fp):
                pass
    else:
        with open(file_path, encoding='utf-8') as fp:
            for i, _ in enumerate(fp):
                pass
    return i + 1
