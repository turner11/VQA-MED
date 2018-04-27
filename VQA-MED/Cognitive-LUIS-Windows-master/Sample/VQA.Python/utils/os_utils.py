import json
import os, errno
import pickle

import sys


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
    def _load_object(fn, load_module, read_mode='rb'):
        with open(fn, read_mode) as f:
            return load_module.load(f)

    @staticmethod
    def _dump_object(obj, fn, dump_module, write_mode='wb'):
        with open(fn, write_mode) as f:
            dump_module.dump(obj, f)

    @staticmethod
    def dump_pickle(obj, fn, write_mode='wb'):
        return File._dump_object(obj, fn, pickle, write_mode)


    @staticmethod
    def load_pickle(fn, read_mode='rb'):
        return File._load_object(fn, pickle, read_mode)

    @staticmethod
    def dump_json(obj, fn, write_mode='w'):
        return File._dump_object(obj, fn, json, write_mode)

    @staticmethod
    def load_json(fn,read_mode='r'):
        return File._load_object(fn, json, read_mode)


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

    @classmethod
    def file_len(cls, fname, encoding=None):
        idx = 0
        with open(fname,  encoding=encoding) as f:
            for i, _ in enumerate(f):
                idx = i
        return idx + 1


def print_progress(count, total, suffix=''):
    bar_len = 60

    percent = float(count) / total
    hashes = '=' * int(round(percent * bar_len))
    spaces = '-' * (bar_len- len(hashes))
    sys.stdout.write("\rPercent: [{0}] {1}%".format(hashes + spaces, int(round(percent * 100))))
    sys.stdout.flush()

    return
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush()