import json
from utils import Timer
import itertools
import sys
import argparse
import os
import logging 
from collections import OrderedDict
#import pandas as pd
import logging
logger = logging.getLogger(__name__)

ERROR_KEY = "error"
logname = "vqa.log"


QUESTIONS_INFO = 'questions info'
class Vqa14_mult_DataParser(object):
    def __init__(self, json_path, **kwargs):
        if not json_path or not os.path.isfile(json_path):
            raise Exception("Got a non valid path: {0}".format(json_path))
        
        self.json_path = json_path
        with open(json_path) as f:
            self.json_data = json.load(f)
        super().__init__(**kwargs)
    
    def __repr__(self, **kwargs):
        return "{0}({1})".format(self.__class__.__name__,self.json_path)



    def get_all_data(self):
        
        t = self.json_data
        #t.keys()
        questions = t["questions"]        
        return questions

    def get_image_data(self, image_path):    
        questions = self.get_all_data() 
    
        caption = ""
        image_info = {}
        try:
            image_id = int(os.path.basename(image_path).split("_")[-1].split(".")[0])
            image_info = {q['question']:q['multiple_choices'] for q in questions if q['image_id'] == image_id}
        except Exception as ex:
            logger.warn("@@@@@@@@@@@@@ Error @@@@@@@@@:\n{0}".format(ex))
            caption = caption or "Failed to get caption"
            image_info = image_info or {}
            image_info[ERROR_KEY] = str(ex)

        #image_info.update({QUESTIONS_INFO:questions_info})    
        return image_info


    def query_data(self, query):    
        questions = self.get_all_data()     
        
        q_results = {}
        try:                
            ql = query.strip().lower()
            q_results = { q['image_id'] :q['question'] for q in questions if ql in str( q['image_id'] ).lower() 
                                                                            or ql in q['question'].lower() 
                                                                           # or any(ql in c.lower() for c in q['multiple_choices'])
                                                                           }
        except Exception as ex:        
            q_results = q_results or {}
            q_results[ERROR_KEY] = str(ex)

        return q_results
        

def main(args):
    try:
        
        file_name = args.path
        image_name = args.imag_name

        parser = Vqa14_mult_DataParser(file_name)
        query = args.query       

        
        if args.imag_name:
            image_info = parser.get_image_data(image_name)        
            ret_val = image_info
        elif query:
            ret_val = parser.query_data(query)
            #ret_val = {"Testing": "query was: {0}".format(query)}
        elif args.question:
            ret_val = {ERROR_KEY: "Asking a question was not implemented yet..."}
        else:
             ret_val = {ERROR_KEY: "Could not figure out what you want me to do..."}
    except Exception as ex:
         ret_val = {ERROR_KEY: "Got a non expected error: {0}".format(ex)}
    finally: 
        pass
    return ret_val

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extracts caption for a COCO image.')    
    parser.add_argument('-p', dest='path', help='path of annotation file')
    parser.add_argument('-n', dest='imag_name', help='name_of_image',default=None)
    parser.add_argument('-q', dest='query', help='query to look for',default=None)
    parser.add_argument('-a', dest='question', help='question to ask',default=None)

    args = parser.parse_args()  

    #args.path = "D:\\GitHub\\VQA-Keras-Visual-Question-Answering\\data\\Questions_Train_mscoco\\MultipleChoice_mscoco_train2014_questions.json"
    #args.imag_name = "COCO_train2014_000000487025"
    #args.query = "polo"
    #args.imag_name = ""

    with Timer() as t:
        ret_val = main(args)
    ret_val['time elapsed'] = str(t)
    #print(str(t))
    json_string = json.dumps(ret_val)
    print(json_string)
        #raw_input()

    #sys.exit(int(main(file_name) or 0))

    #python C:\Users\avitu\Documents\GitHub\VQA-MED\VQA-MED\Cognitive-LUIS-Windows-master\Sample\VQA.Python\VQA.Python.py -p "C:\Users\Public\Documents\Data\2014 Train\annotations\captions_train2014.json" -n "C:\Users\Public\Documents\Data\2014 Train\train2014\COCO_train2014_000000000061.jpg"
    #"C:\Program Files (x86)\Microsoft Visual Studio\Shared\Anaconda3_64\python.exe" "C:\Users\avitu\Documents\GitHub\VQA-MED\VQA-MED\Cognitive-LUIS-Windows-master\Sample\VQA.Python\VQA.Python.py" -p "C:\Users\Public\Documents\Data\2014 Train\annotations\captions_train2014.json" -n "C:\Users\Public\Documents\Data\2014 Train\train2014\COCO_train2014_000000000061.jpg"
