import os, errno

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