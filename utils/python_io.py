import sys


class Logger(object):
    """
    Logger class to print output to terminal and to file
    """

    def __init__(self, filename):
        self.terminal = sys.stdout
        self.logfile = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def start_logging(filename):
    """Start logger, appending print output to given filename"""
    sys.stdout = Logger(filename)


def stop_logging():
    """Stop logger and return print functionality to normal"""
    sys.stdout.logfile.close()
    sys.stdout = sys.stdout.terminal
