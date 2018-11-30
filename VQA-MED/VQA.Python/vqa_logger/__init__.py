import logging
import os

import sys

# import coloredlogs
import datetime
import time

from common.os_utils import File

log_level = logging.DEBUG
# format = '%(asctime)s,%(msecs)d %(name)s %(levelname)s ## %(message)s'
format = '[%(asctime)s][%(levelname)s] %(message)s'

now = time.time()
ts = datetime.datetime.fromtimestamp(now).strftime('%Y%m%d')
file_name = os.path.join(os.getcwd(),'logs',f"{ts}_vqa.log")
folder, _ = os.path.split(file_name)
File.validate_dir_exists(folder)

logging.basicConfig(filemode='a',
                    format=format,
                    datefmt='%H:%M:%S',
                    level=log_level,
                    # stream=sys.stdout,
                    filename=file_name)


# coloredlogs.install()
logger = logging.getLogger('pythonVQA')

#
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(log_level)
formatter = logging.Formatter(format)
ch.setFormatter(formatter)
if not logger.handlers:
    logger.removeHandler(ch)
    logger.addHandler(ch)


# LOGNAME = "vqa"
# logger = logging.getLogger(LOGNAME)
# logger.level = logging.DEBUG
#
# fileHandler = logging.FileHandler("{0}.log".format(LOGNAME))
# consoleHandler = logging.StreamHandler()
# logger.addHandler(fileHandler)
# logger.addHandler(consoleHandler)