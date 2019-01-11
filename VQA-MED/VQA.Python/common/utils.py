import time
from contextlib import contextmanager
from datetime import timedelta

import os

import sys


class Timer:
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start

    def __str__(self):
        return str(timedelta(seconds=self.interval))


class VerboseTimer(Timer):
    def __init__(self, title):
        """"""
        super().__init__()
        self.title = title

    def __exit__(self, *args):
        super().__exit__()
        print("{0}: {1}".format(self.title, str(self)))



def timeit(func):
    def timed(*args, **kw):
        with Timer() as t:
            result = func(*args, **kw)
        print('Function {0}; Time: {1}'.format(func.__name__, 'time:', str(t)))
        return result
    return timed


def get_word_match(word, string):
    import re
    k= re.compile(r'\b%s\b' % word, re.I)
    try:
        res = k.search(string)
    except TypeError as ex:
        res = None
    return res

def has_word(word, string):
    return get_word_match(word, string) is not None


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def suppress_func_stdout(func):
    def silent_func(*args, **kw):
        with suppress_stdout():
            result = func(*args, **kw)
        return result

    return silent_func

