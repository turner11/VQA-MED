import argparse
import os
from pathlib import Path

from common.supress_print import supress_print
import json
from classes.vqa_model_predictor import DefaultVqaModelPredictor

default_model_id = 5
model: DefaultVqaModelPredictor = None

def error_to_json(func):
    def wrapper(*args, **kw):
        try:
            result = func(*args, **kw)
        except Exception as ex:
            result = json.dumps({'error': str(ex)})
            # import traceback as tb
            # print(tb.format_exc())

        return result
    return wrapper


@error_to_json
def get_models():
    from common import DAL
    from data_access.model_folder import ModelFolder

    models = DAL.get_models_data_frame()
    #'columns' # 'records'#'values'#'table'#'index'#'split'#
    # j = models.to_json(orient='columns')

    models['image_path'] = models.model_location.apply(lambda location: str(ModelFolder(Path(location).parent).image_file_path))
    models['summary'] = models.model_location.apply(
        lambda location: str(ModelFolder(Path(location).parent).summary))
    cols = [c for c in models.columns if c.lower() != 'models']
    j = models[cols].to_json(orient='columns')
    # m = models[['models']]
    return j


@supress_print
def set_model(model_id=default_model_id, cpu=True):
    global model
    if cpu:
        set_cpu()

    try:
        mp = DefaultVqaModelPredictor(model_id)
        model = mp
        success = True
    except:
        raise
    return success




@error_to_json
# @supress_print
def predict(question, image_path):
    import pandas as pd
    from pre_processing.prepare_data import pre_process_raw_data
    if model is None:
        set_model()


    #,get_highlited_function_code, get_image, get_size


    data = {
            'image_name': [Path(image_path).name],
            'question':   [question],
            'path':       [image_path],
    }
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

    # set_model(5)
    # a = predict(question='what type of imaging modality is shown?',
    #             image_path=r'C:\\Users\\Public\\Documents\\Data\\2019\\validation\\Val_images\\synpic100545.jpg')
    # #
    # str()
    # ms =  get_models()
    # str()


def set_cpu():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ""



if __name__ == '__main__':
    main()

