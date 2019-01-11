import os
import pandas as pd
from parsers.VQA18 import Vqa18Base
# from pre_processing.known_find_and_replace_items import DataLocations
import logging
logger = logging.getLogger(__name__)




class DAL(object):
    """"""

    def __init__(self):
        """"""
        super(DAL, self).__init__()

    def __repr__(self):
        return super(DAL, self).__repr__()

    @staticmethod
    def get_df(data_arg):
        logger.debug('Getting data for argument: {0}'.format(data_arg))
        from pre_processing.known_find_and_replace_items import DataLocations
        if isinstance(data_arg, pd.DataFrame):
            df = data_arg
        elif isinstance(data_arg, DataLocations):
            inst = Vqa18Base.get_instance(data_arg.processed_xls)
            df = inst.data
        elif os.path.isfile(str(data_arg)):
            inst = Vqa18Base.get_instance(data_arg)
            df = inst.data
        else:
            raise Exception('Unrecognized data arg: {0}'.format(data_arg))
        return df
