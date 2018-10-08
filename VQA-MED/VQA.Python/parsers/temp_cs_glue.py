import argparse
import os
from common.supress_print import supress_print
import json

model = None

def error_to_json(func):
    def wrapper(*args, **kw):
        try:
            result = func(*args, **kw)
        except Exception as ex:
            result = json.dumps({'error': str(ex)})

        return result
    return wrapper


@error_to_json
def get_models():
    from common import DAL

    models = DAL.get_models_data_frame()
    #'columns' # 'records'#'values'#'table'#'index'#'split'#
    # j = models.to_json(orient='columns')

    cols = [c for c in models.columns if c.lower() != 'models']
    j = models[cols].to_json(orient='columns')
    # m = models[['models']]
    return j


@supress_print
def set_model(model_id, cpu=True):
    if cpu:
        set_cpu()
    from common import DAL
    global model


    try:
        from classes.vqa_model_predictor import VqaModelPredictor
        model_dal = DAL.get_model_by_id(model_id)
        mp = VqaModelPredictor(model_dal)
        model = mp
        success = True
    except:
        success = False
        raise
    return success




@error_to_json
@supress_print
def predict(question, image_path):
    import pandas as pd
    from common.functions import pre_process_raw_data

    #,get_highlited_function_code, get_image, get_size

    data = {'question':  [question],
            'path':      [image_path],
            'image_name':[os.path.split(image_path)[-1]]}
    df = pd.DataFrame(data)
    pre_processed = pre_process_raw_data(df)
    # df['question']
    # df['path']
    # df['image_name']
    ps = model.predict(pre_processed)

    cols_to_ommit = ['answer']
    clean_df = ps[[col for col in ps.columns if col not in cols_to_ommit]]
    # j = clean_df.iloc[0].to_json()
    j = clean_df.to_json(orient='columns')
    return j






def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-m', dest='model_id', help='the model to set', default=34)
    parser.add_argument('-c', '--cpu', dest='cpu',help='forces usage of CPU', default=False,action='store_true')

    args = parser.parse_args()
    if True or args.cpu:
        set_cpu()

    # set_model(args.model_id)
    # question = 'what does CT show?'
    # curr_image_path = r'C:\Users\Public\Documents\Data\2018\VQAMed2018Train\VQAMed2018Train-images\0392-100X-30-209-g003.jpg'
    # predict(question=question,curr_image_path=curr_image_path)
    # ms =  get_models()


def set_cpu():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


if __name__ == '__main__':
    main()

