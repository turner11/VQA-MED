import sys
from io import StringIO


class SupressPrint(object):

    def __init__(self):
        pass

    def __enter__(self):
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()

        import warnings
        warnings.filterwarnings('ignore')

    def __exit__(self, *args):
        sys.stdout = self.stdout
        sys.stderr = self.stderr


def supress_print(func):
    def supressed(*args, **kw):
        with SupressPrint():
            result = func(*args, **kw)
        return result
    return supressed
