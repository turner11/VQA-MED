import json
import argparse
import itertools
import sys
import os
# from collections import OrderedDict
# from .vqa_logger import logger

import pandas as pd

from common.utils import VerboseTimer

ERROR_KEY = "error"
df_path = 'C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\data\\model_input.h5'
print(f'Loading data from \n"{df_path}"')
with VerboseTimer('Loading Data'):
    with pd.HDFStore(df_path) as store:
        df_data = store['data']


def get_image_data(image_name):
    try:
        # pd.read_hdf(df_path,key='data', where=[f'image_name=image_name'])

        df_ret = df_data[df_data.image_name == image_name]
        df_ret = df_ret[['image_name','question', 'answer', 'imaging_device']]
        json_ret = df_ret.to_json()


    except Exception as ex:
        import traceback
        tb = traceback.format_exc()
        json_ret = json.dumps({ERROR_KEY: "Got an unexpected error:\n{0}\n\nTraceback:\n{1}".format(ex,tb)})
    finally: 
        pass
    return json_ret

if __name__ == "__main__":
    image = '0392-100X-31-109-g001.jpg'
    ret_val = get_image_data(image)
    print(ret_val)
    if False:


        parser = argparse.ArgumentParser(description='Extracts caption for a COCO image.')
        parser.add_argument('-p', dest='df_path', help='path of data frame')
        parser.add_argument('-n', dest='image_name', help='name_of_image',default=None)
        # parser.add_argument('-q', dest='query', help='query to look for',default=None)
        # parser.add_argument('-a', dest='question', help='question to ask',default=None)

        args = parser.parse_args()


        #pix_p = "C:\\Users\\Public\\Documents\\Data\\2017\\annotations\\stuff_val2017_pixelmaps"
        #imgs_p = "C:\\Users\\Public\\Documents\\Data\\2017\\val2017"

        #def clean_names(path):
        #    return sorted([int(fn.split(".")[0]) for fn in os.listdir(path)])
        #pix = clean_names(pix_p)
        #img = clean_names(imgs_p)
        #inter = set(pix).intersection(set(img))
        #print(inter)



        #args.path = "C:\\Users\\Public\\Documents\\Data\\2017\\annotations\\stuff_val2017.json"
        #args.imag_name= "C:\\Users\\Public\\Documents\\Data\\2017\\val2017\\000000000285.jpg"#"C:\\Users\\Public\\Documents\\Data\\2017\\val2017\\000000000139.jpg"#"C:\\Users\\Public\\Documents\\Data\\2017\\val2017\\000000001584.jpg"

        #args.path = "C:\\Users\\Public\\Documents\\Data\\2017\\annotations\\stuff_val2017.json"
        #args.imag_name= None
        #args.query = "dishes"


        ret_val = main(**args.__dict__)
        json_string = json.dumps(ret_val)
        print(json_string)
            #raw_input()

        #sys.exit(int(main(file_name) or 0))

        #python C:\Users\avitu\Documents\GitHub\VQA-MED\VQA-MED\Cognitive-LUIS-Windows-master\Sample\VQA.Python\VQA.Python.py -p "C:\Users\Public\Documents\Data\2014 Train\annotations\captions_train2014.json" -n "C:\Users\Public\Documents\Data\2014 Train\train2014\COCO_train2014_000000000061.jpg"
        #"C:\Program Files (x86)\Microsoft Visual Studio\Shared\Anaconda3_64\python.exe" "C:\Users\avitu\Documents\GitHub\VQA-MED\VQA-MED\Cognitive-LUIS-Windows-master\Sample\VQA.Python\VQA.Python.py" -p "C:\Users\Public\Documents\Data\2014 Train\annotations\captions_train2014.json" -n "C:\Users\Public\Documents\Data\2014 Train\train2014\COCO_train2014_000000000061.jpg"