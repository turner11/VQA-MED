import json
from utils import Timer
import itertools
import sys
import argparse
import os
import logging
from collections import OrderedDict
import pandas as pd

ERROR_KEY = "error"
logname = "vqa.log"
logging.basicConfig(filename=logname,
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

logger = logging.getLogger('pythonVQA')

QUESTIONS_INFO = 'questions info'


class Vqa18(object):
    COL_ROW_ID = 'row_id'
    COL_IMAGE_NAME = "image_name"
    COL_QUESTION = "question"
    COL_ANSWER = "answer"

    ALL_COLLS = [COL_ROW_ID, COL_IMAGE_NAME, COL_QUESTION, COL_ANSWER]

    def __init__(self, csv_path, **kwargs):
        if not csv_path or not os.path.isfile(csv_path):
            raise Exception("Got a non valid path: {0}".format(csv_path))

        self.csv_path = csv_path
        # self.csv_data = df = pd.read_csv(csv_path)
        self.csv_data = pd.read_csv(csv_path, sep='\t', header=None,
                                    names=self.ALL_COLLS)
        self.csv_data.set_index(self.COL_ROW_ID)
        super().__init__(**kwargs)

    def __repr__(self, **kwargs):
        return "{0}({1})".format(self.__class__.__name__, self.csv_path)

    def get_all_data(self):
        return self.csv_data

    def get_image_data(self, image_path):
        data = self.get_all_data()
        _, fn = os.path.split(image_path)
        clean_image_name, ext = os.path.splitext(fn)
        # clean_image_name = os.path.basename(image_path).split('\\')[-1].replace(".j")

        i_question_rows = data[self.COL_IMAGE_NAME] == clean_image_name
        image_data = data[i_question_rows]

        ret = image_data.to_json()
        j = json.loads(ret)
        return j



    def query_data(self, query):
        df = self.get_all_data()

        #self.plot_data_info(df)
        match = None
        for col in self.ALL_COLLS:
            try:
                curr_match = df[col].str.contains(query)
                if match is None:
                    match = curr_match
                match = match | curr_match
            except:
                pass
        filtered = df[match]

        match_names = filtered[self.COL_IMAGE_NAME].values
        ret = {n:n for n in match_names}
        return ret
        # ret = filtered.to_json()
        # j = json.loads(ret)
        # return j

    def plot_data_info(self, data):
        import matplotlib.pyplot as plt
        # df[self.COL_ANSWER].value_counts().plot(kind='bar')
        cols = [self.COL_QUESTION,self.COL_ANSWER]
        for col in cols:
            df = data[col].value_counts()
            plt.figure(col)
            f = df[df > 9]
            ax = plt.barh(range(len(f.index)), f.values)
            plt.yticks(range(len(f.index)), f.index.values)
            plt.gca().invert_yaxis()
            plt.show()


def main(args):
    try:

        file_name = args.path
        image_name = args.imag_name

        parser = Vqa18(file_name)
        query = args.query

        if args.imag_name:
            image_info = parser.get_image_data(image_name)
            ret_val = image_info
        elif query:
            ret_val = parser.query_data(query)
            # ret_val = {"Testing": "query was: {0}".format(query)}
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
    parser.add_argument('-n', dest='imag_name', help='name_of_image', default=None)
    parser.add_argument('-q', dest='query', help='query to look for', default=None)
    parser.add_argument('-a', dest='question', help='question to ask', default=None)

    args = parser.parse_args()

    # args.path = "D:\\GitHub\\VQA-Keras-Visual-Question-Answering\\data\\Questions_Train_mscoco\\MultipleChoice_mscoco_train2014_questions.json"
    # args.imag_name = "COCO_train2014_000000487025"
    # args.query = "polo"
    # args.imag_name = ""

    with Timer() as t:
        ret_val = main(args)
    ret_val['time elapsed'] = str(t)
    # print(str(t))
    json_string = json.dumps(ret_val)
    print(json_string)
    # raw_input()

    # sys.exit(int(main(file_name) or 0))

    # python C:\Users\avitu\Documents\GitHub\VQA-MED\VQA-MED\Cognitive-LUIS-Windows-master\Sample\VQA.Python\VQA.Python.py -p "C:\Users\Public\Documents\Data\2014 Train\annotations\captions_train2014.json" -n "C:\Users\Public\Documents\Data\2014 Train\train2014\COCO_train2014_000000000061.jpg"
    # "C:\Program Files (x86)\Microsoft Visual Studio\Shared\Anaconda3_64\python.exe" "C:\Users\avitu\Documents\GitHub\VQA-MED\VQA-MED\Cognitive-LUIS-Windows-master\Sample\VQA.Python\VQA.Python.py" -p "C:\Users\Public\Documents\Data\2014 Train\annotations\captions_train2014.json" -n "C:\Users\Public\Documents\Data\2014 Train\train2014\COCO_train2014_000000000061.jpg"
