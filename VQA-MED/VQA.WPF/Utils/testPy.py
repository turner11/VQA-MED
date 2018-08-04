import os, sys
sys.path.insert(0, os.getcwd())


print("\n".join([p for p in sys.path]))
print("\n\nCurrent Dir:\n")
print(os.getcwd())


def add(a,b):
    print(str(a+b))


if True:

    from keras.models import load_model
    from common.constatns import images_path_test
    from common.utils import VerboseTimer
    from common.DAL import get_models_data_frame, get_model
    from parsers.VQA18 import Vqa18Base
    from common.functions import get_size, get_highlited_function_code, normalize_data_strucrture
    from vqa_logger import logger
    from common.os_utils import File 
    from pandas import HDFStore
    import numpy as np


    vqa_specs_location = 'C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\data\\vqa_specs.pkl'


    model_id = 2
    model_dal = get_model(model_id)

    model_location = model_dal.model_location

    with VerboseTimer("Loading Model"):
        model = load_model(model_location)


    vqa_specs = File.load_pickle(vqa_specs_location)
    data_location = vqa_specs.data_location



    logger.debug(f"Loading test data from {data_location}")
    with VerboseTimer("Loading Test Data"):
        with HDFStore(data_location) as store:        
            df_data = store['test']


    def concate_row(df, col):
        return np.concatenate(df[col], axis=0)

    def get_features_and_labels(df):
        image_features = np.asarray([np.array(im) for im in df['image']])
        # np.concatenate(image_features['question_embedding'], axis=0).shape
        question_features = concate_row(df, 'question_embedding') 

        reshaped_q = np.array([a.reshape(a.shape + (1,)) for a in question_features])
    
        features = ([f for f in [reshaped_q, image_features]])    
    
        return features




    def predcit(idx):
        row = df_data.ix[idx]
        features = get_features_and_labels(row)
        p = model.predict(features)
        print(p)