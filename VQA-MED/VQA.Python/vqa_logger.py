import os
import sys
import datetime
import time
import coloredlogs, logging
from common.os_utils import File

def init_log(name=None):
    name = name or __name__
    # Create a logger object.
    logger = logging.getLogger(name)
    log_level = logging.DEBUG
    # format = '%(asctime)s,%(msecs)d %(name)s %(levelname)s ## %(message)s'
    format = '[%(asctime)s][%(levelname)s] %(message)s'

    # By default the install() function installs a file_handler on the root logger,
    # this means that log messages from your code and log messages from the
    # libraries that you use will all show up on the terminal.
    coloredlogs.install(level='DEBUG', fmt=format)



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
                        # filename=file_name
                        )

    formatter = logging.Formatter(format)
    file_handler = logging.FileHandler(file_name)
    file_handler.setFormatter(formatter)
    logger.setLevel(logging.NOTSET)
    logger.addHandler(file_handler)
    return  logger


def test_log():
    l = logging.getLogger(__name__)
    l.debug("this is a debugging message")
    l.info("this is an informational message")
    l.warning("this is a warning message")
    l.error("this is an error message")
    l.critical("this is a critical message")

init_log()
# test_log()