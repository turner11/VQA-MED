import logging

import sys

log_level = logging.DEBUG
format = '%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s'
logname = "vqa.log"

logging.basicConfig(filename=logname,
                    filemode='a',
                    format=format,
                    datefmt='%H:%M:%S',
                    level=log_level)

logger = logging.getLogger('pythonVQA')

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(log_level)
# formatter = logging.Formatter(format)
# ch.setFormatter(formatter)
logger.addHandler(ch)