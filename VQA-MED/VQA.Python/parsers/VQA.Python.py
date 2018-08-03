import json
import argparse
import itertools
import sys
import os
# from collections import OrderedDict
# from .vqa_logger import logger

import pandas as pd

# from common.utils import VerboseTimer

ERROR_KEY = "error"
df_path = 'C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\data\\model_input.h5'
# print(f'Loading data from \n"{df_path}"\n\n')
# with VerboseTimer('Loading Data'):
#     with pd.HDFStore(df_path) as store:
#         df_data = store['data']


def get_image_data(image_name, dataframe_path=df_path):
    try:
        df_ret = pd.read_hdf(dataframe_path, key='light', where=[f'image_name="{image_name}"'])

        # df_ret = df_data[df_data.image_name == image_name]
        df_ret = df_ret[['image_name','question', 'answer', 'imaging_device']]
        json_ret = df_ret.to_json(orient='columns')


    except Exception as ex:
        import traceback
        tb = traceback.format_exc()
        json_ret = json.dumps({ERROR_KEY: "Got an unexpected error:\n{0}\n\nTraceback:\n{1}".format(ex,tb)})
    finally: 
        pass
    return json_ret

if __name__ == "__main__":
    # image = '0392-100X-31-109-g001.jpg'
    # ret_val = get_image_data(image)
    # print(ret_val)
    if True:
        parser = argparse.ArgumentParser(description='Extracts caption for a COCO image.')
        parser.add_argument('-p', dest='df_path', help='path of data frame',default=None)
        parser.add_argument('-n', dest='image_name', help='name_of_image')
        # parser.add_argument('-q', dest='query', help='query to look for',default=None)
        # parser.add_argument('-a', dest='question', help='question to ask',default=None)

        args = parser.parse_args()
        ret_val = get_image_data(image_name=args.image_name, dataframe_path=args.df_path)
        json_string = ret_val
        # json_string = json.dumps(ret_val)
        print(json_string)
        #raw_input()

