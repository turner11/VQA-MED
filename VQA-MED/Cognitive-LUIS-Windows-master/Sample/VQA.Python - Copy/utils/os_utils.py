import os, errno
import pickle


class File(object):
    def __init__(self, ):
        """"""

    @staticmethod
    def validate_dir_exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    @staticmethod
    def dump_pickle(obj, fn):
        with open(fn, 'wb') as f:
            pickle.dump(obj.history, f)

    @staticmethod
    def load_pickle(fn):
        with open(fn, 'rb') as f:
            return pickle.load(f)

    @classmethod
    def write_text(cls, fn, txt):
        with open(fn, "w") as text_file:
            text_file.write(txt)

    @classmethod
    def write_lines(cls, fn, txt):
        with open(fn, "w") as text_file:
            text_file.writelines(txt)

    @classmethod
    def read_text(cls, fn, txt):
        with open(fn, "r") as text_file:
            return text_file.read(txt)

    @classmethod
    def read_lines(cls, fn, txt):
        with open(fn, "r") as text_file:
            return text_file.readlines(txt)