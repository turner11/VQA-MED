import datetime
import os
import sys
import time

import coloredlogs
import logging



from common.os_utils import File

__is_initialized = False


def init_log():
    global __is_initialized
    if __is_initialized:
        return

    __is_initialized = True
    root_logger = logging.getLogger()
    # This for avoiding streams to log to root's stderr, which prints in red in jupyter
    for handler in root_logger.handlers:
        # continue
        root_logger.removeHandler(handler)

    log_format = '[%(asctime)s][%(name)s][%(levelname)s] %(message)s'
    formatter = logging.Formatter(log_format)

    std_out_log_level = logging.DEBUG

    # By default the install() function installs a file_handler on the root root_logger,
    # this means that log messages from your code and log messages from the
    # libraries that you use will all show up on the terminal.
    coloredlogs.install(level=std_out_log_level, fmt=log_format, stream=sys.stdout)

    now = time.time()
    ts = datetime.datetime.fromtimestamp(now).strftime('%Y%m%d')
    file_name = os.path.abspath(os.path.join(os.getcwd(), '..', 'logs', f"{ts}_vqa.log"))
    folder, _ = os.path.split(file_name)
    File.validate_dir_exists(folder)

    std_out = logging.StreamHandler(sys.stdout)
    std_out.setFormatter(formatter)
    std_out.setLevel(std_out_log_level)

    file_handler = logging.FileHandler(file_name)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    logging.basicConfig(filemode='a',
                        format=log_format,
                        datefmt='%H:%M:%S',
                        level=logging.ERROR,
                        stream=std_out,
                        # filename=file_handler
                        )


    root_logger.addHandler(file_handler)
    str()
    # root_logger.addHandler(std_out)
    # print (str(root_logger.handlers))


def test_log():
    # init_log()
    local_logger = logging.getLogger(__name__)
    print('This is just a print')
    local_logger.debug("this is a debugging message")
    local_logger.info("this is an informational message")
    local_logger.warning("this is a warning message")
    local_logger.error("this is an error message")
    local_logger.critical("this is a critical message")


init_log()
# test_log()
