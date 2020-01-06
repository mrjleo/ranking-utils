import csv
from datetime import datetime


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