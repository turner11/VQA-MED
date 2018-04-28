import logging

import sys

import coloredlogs

log_level = logging.DEBUG
format = '%(asctime)s,%(msecs)d %(name)s %(levelname)s ## %(message)s'
logname = "vqa.log"

logging.basicConfig(filemode='a',
                    format=format,
                    datefmt='%H:%M:%S',
                    level=log_level)
coloredlogs.install()
logger = logging.getLogger('pythonVQA')
#
# ch = logging.StreamHandler(sys.stdout)
# ch.setLevel(log_level)
# # formatter = logging.Formatter(format)
# # ch.setFormatter(formatter)
# if not logger.handlers:
#     logger.removeHandler(ch)
#     logger.addHandler(ch)


# LOGNAME = "vqa"
# logger = logging.getLogger(LOGNAME)
# logger.level = logging.DEBUG
#
# fileHandler = logging.FileHandler("{0}.log".format(LOGNAME))
# consoleHandler = logging.StreamHandler()
# logger.addHandler(fileHandler)
# logger.addHandler(consoleHandler)