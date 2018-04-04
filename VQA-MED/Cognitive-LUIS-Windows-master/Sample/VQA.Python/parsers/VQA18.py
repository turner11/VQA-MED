import json
from parsers.utils import Timer
import argparse
import os

import pandas as pd

from vqa_logger import logger
from pre_processing.known_find_and_replace_items import all_tags

ERROR_KEY = "error"
QUESTIONS_INFO = 'questions info'
TOKENIZED_COL_PREFIX = 'tokenized_'
class Vqa18Base(object):

    COL_ROW_ID = 'row_id'
    COL_IMAGE_NAME = "image_name"
    COL_QUESTION = "question"
    COL_ANSWER = "answer"
    COL_TOK_Q = TOKENIZED_COL_PREFIX + COL_QUESTION
    COL_TOK_A = TOKENIZED_COL_PREFIX + COL_ANSWER

    @property
    def ALL_RAW_COLS(self):
        return [self.COL_ROW_ID, self.COL_IMAGE_NAME, self.COL_QUESTION, self.COL_ANSWER]


    def __init__(self, data_path, **kwargs):
        super().__init__()
        if not data_path or not os.path.isfile(data_path):
            raise Exception("Got a non valid path: {0}".format(data_path))

        self.data_path = data_path
        self.data = self._read_data(data_path)
        assert self.data is not None, "Got a None data set"
        self.data.set_index(self.COL_ROW_ID)


    @classmethod
    def get_instance(cls, data_path=None):
        ctors = iter([
            lambda excel_path=data_path: Vqa18_from_processed_excel(excel_path),
            lambda excel_path=data_path: Vqa18_from_excel(excel_path),
            lambda: Vqa18_from_excel(Vqa18_from_raw_csv.csv_path_2_excel_path(data_path)),
            lambda csv_path=data_path: Vqa18_from_raw_csv(csv_path),
                      ])
        instance = None
        while not instance:
            try:
                ctor = next(ctors)
                instance = ctor()
            except StopIteration as ex:
                raise
            except AssertionError as ex:
                raise
            except Exception as ex:
                pass
        return instance



    def __repr__(self, **kwargs):
        return "{0}({1})".format(self.__class__.__name__, self.data_path)

    def get_all_data(self):
        return self.data

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
        for col in self.ALL_RAW_COLS:
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

    def _read_data(self, data_path):
        raise NotImplementedError()



class Vqa18_from_raw_csv(Vqa18Base):
    def __init__(self, csv_path, **kwargs):
        super().__init__(csv_path,**kwargs)

        # excel_path = self.csv_path_2_excel_path(csv_path)
        #
        # self.add_tokenized_column(self.data, self.COL_QUESTION, self.COL_TOK_Q)
        # self.add_tokenized_column(self.data, self.COL_ANSWER, self.COL_TOK_A)
        # self.dump_to_excel(self.data, excel_path)
        # process_excel(excel_path, excel_path)
        # excel_data = pd.read_excel(excel_path)

    def _read_data(self, data_path):
        # self.data = df = pd.read_csv(data_path)
        return pd.read_csv(data_path, sep='\t', header=None, names=self.ALL_RAW_COLS)

    @classmethod
    def csv_path_2_excel_path(self, csv_path):
        path = os.path.splitext(csv_path)[0] + '_DUMPED.xlsx'
        return path


class Vqa18_from_excel(Vqa18Base):
    @property
    def ALL_RAW_COLS(self):
        return list(set(super().ALL_RAW_COLS + [self.COL_TOK_Q, self.COL_TOK_A]))
    def __init__(self, excel_path, **kwargs):
        super().__init__(excel_path,**kwargs)

    def _read_data(self, data_path):
        # df =  pd.read_excel(data_path, names=self.ALL_RAW_COLS)
        df = pd.read_excel(data_path)
        cols = [c for c in self.ALL_RAW_COLS if c in df.columns]
        df = df[cols]
        return df

class Vqa18_from_processed_excel(Vqa18_from_excel):
    @property
    def ALL_RAW_COLS(self):
        return list(set(super().ALL_RAW_COLS + all_tags))
    def __init__(self, excel_path, **kwargs):
        super().__init__(excel_path,**kwargs)




def main(args):
    try:

        file_name = args.path
        image_name = args.image_name

        parser = Vqa18Base.get_instance(file_name)
        query = args.query

        if args.image_name:
            image_info = parser.get_image_data(image_name)
            ret_val = image_info

            # import matplotlib.pyplot as plt
            # ax = parser.data[all_tags].plot(kind='bar', title="V comp", figsize=(15, 10), legend=True, fontsize=12)
            # ax.set_xlabel("Hour", fontsize=12)
            # ax.set_ylabel("V", fontsize=12)
            # plt.show()

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
    parser.add_argument('-n', dest='image_name', help='name_of_image', default=None)
    parser.add_argument('-q', dest='query', help='query to look for', default=None)
    parser.add_argument('-a', dest='question', help='question to ask', default=None)

    args = parser.parse_args()
    # args.path = 'C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\Cognitive-LUIS-Windows-master\\Sample\\VQA.Python\\dumped_data\\vqa_data.xlsx'
    # args.path = "D:\\GitHub\\VQA-Keras-Visual-Question-Answering\\data\\Questions_Train_mscoco\\MultipleChoice_mscoco_train2014_questions.json"
    # args.image_name = "COCO_train2014_000000487025"
    # args.query = "polo"
    # args.image_name = ""

    with Timer() as t:
        ret_val = main(args)
    ret_val['time elapsed'] = str(t)
    # logger.debug(str(t))
    json_string = json.dumps(ret_val)
    logger.info(json_string)
    # raw_input()

    # sys.exit(int(main(file_name) or 0))

    # python C:\Users\avitu\Documents\GitHub\VQA-MED\VQA-MED\Cognitive-LUIS-Windows-master\Sample\VQA.Python\VQA.Python.py -p "C:\Users\Public\Documents\Data\2014 Train\annotations\captions_train2014.json" -n "C:\Users\Public\Documents\Data\2014 Train\train2014\COCO_train2014_000000000061.jpg"
    # "C:\Program Files (x86)\Microsoft Visual Studio\Shared\Anaconda3_64\python.exe" "C:\Users\avitu\Documents\GitHub\VQA-MED\VQA-MED\Cognitive-LUIS-Windows-master\Sample\VQA.Python\VQA.Python.py" -p "C:\Users\Public\Documents\Data\2014 Train\annotations\captions_train2014.json" -n "C:\Users\Public\Documents\Data\2014 Train\train2014\COCO_train2014_000000000061.jpg"
