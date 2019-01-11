import itertools
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer, OneHotEncoder

from common.functions import get_features
from keras.models import load_model
from classes.vqa_model_builder import VqaModelBuilder
from classes.vqa_model_predictor import VqaModelPredictor
from common.DAL import get_model_by_id


def reset_model(model):
    from keras.initializers import glorot_uniform  # Or your initializer of choice
    import keras.backend as K

    initial_weights = model.get_weights()

    backend_name = K.backend()
    if backend_name == 'tensorflow':
        k_eval = lambda placeholder: placeholder.eval(session=K.get_session())
    elif backend_name == 'theano':
        k_eval = lambda placeholder: placeholder.eval()
    else:
        raise ValueError("Unsupported backend")

    new_weights = [k_eval(glorot_uniform()(w.shape)) for w in initial_weights]

    model.set_weights(new_weights)


def retrain_model():
    # from classes.vqa_model_predictor import VqaModelPredictor


    # data_path = 'C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\data\\model_input.h5'
    # with pd.HDFStore(data_path) as store:
    #     train_data = store['data']
    #
    # model_dal = get_model_by_id(80)
    # mp = VqaModelPredictor(model_dal)
    #
    # # mp = VqaModelPredictor(None)
    # model = mp.image_device_classifier
    # reset_model(model)
    # vqa_data, imaging_data = mp.split_data_to_vqa_and_imaging(train_data)
    #
    #
    meta_path = 'C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\data\\meta_data.h5'
    with pd.HDFStore(meta_path) as store:
        imaging_devices   = store['imaging_devices'] #ct , mri


    #
    devices = list(itertools.chain.from_iterable(imaging_devices.values))
    # train_data_raw = imaging_data.drop_duplicates(subset=['image_name'])
    # train_data_raw = train_data_raw[train_data_raw.imaging_device.isin(devices)]

    # with pd.HDFStore('D:\\Users\\avitu\\Downloads\\temp.hdf') as store:
    #     store['train_data_raw'] =train_data_raw
    #     store['devices'] = devices


    with pd.HDFStore('D:\\Users\\avitu\\Downloads\\temp.hdf') as store:
        train_data_raw = store['train_data_raw']

    df_train, df_test = train_test_split(train_data_raw, test_size=0.25)
    print(len(df_train),len(df_test))


    features_list = []
    labels_list = []
    for df in [df_train, df_test ]:

        features = get_features(df)
        text = features[0]
        for t in text:
            t[:]=0

        # labeler = LabelBinarizer()
        labeler = OneHotEncoder(sparse=False)
        labels = df.imaging_device.apply(lambda v: devices.index(v))
        labels = labels.reshape(-1, 1)
        y_train_vector = labeler.fit_transform(labels)
        y_train_vector = [list(l) for l in y_train_vector] #list(itertools.chain.from_iterable(y_train_vector))

        features_list.append(features)
        labels_list.append(np.asarray(y_train_vector))

    features_t, labels_t = features_list[0], labels_list[0]
    features_val, labels_val = features_list[1], labels_list[1]

    pp = 'C:\\Users\\Public\\Documents\\Data\\2018\\vqa_models\\20180815_0137_53\\vqa_model_ClassifyStrategies.CATEGORIAL_trained.h5'
    from keras.models import load_model
    model = load_model(pp)
    reset_model(model)

    # from
    # >>> train_data_raw[['imaging_device']].describe()
    factor =1.653276955602537
    class_weight = {0: factor,
                    1: 1.,
                    }

    history = model.fit(features_t, labels_t,
                        epochs=5,
                        batch_size=64,
                        validation_data=(features_val, labels_val),
                        shuffle=True,
                        class_weight=class_weight)

    str(model)


    

    return None

def main():
    retrain_model()


    data_set = ''


if __name__ == '__main__':
    main()
