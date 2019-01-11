import os
import itertools
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from common.constatns import vqa_models_folder
from common.functions import get_features
from keras import Model
from keras.layers import Dense, BatchNormalization, Activation

from common.model_utils import save_model
from evaluate.statistical import f1_score, recall_score, precision_score
from classes.vqa_model_builder import VqaModelBuilder

logger = logging.getLogger(__name__)


def get_device_predictor_model(number_of_devices):
    metrics = [f1_score, recall_score, precision_score, 'accuracy']
    model_output_num_units = number_of_devices

    loss_function = 'categorical_crossentropy'
    optimizer ='Adam'
    output_activation_function = 'softmax'  # 'relu'
    try:
        image_input_tensor, image_model = VqaModelBuilder.get_image_model()

        logger.debug("merging final model")
        fc_tensors = Dense(units=256)(image_model)
        fc_tensors = BatchNormalization()(fc_tensors)
        fc_tensors = Activation('relu')(fc_tensors)

        fc_tensors = Dense(units=model_output_num_units, activation=output_activation_function,
                           name=f'model_output_{output_activation_function}_dense')(fc_tensors)

        fc_model = Model(inputs=image_input_tensor, output=fc_tensors)
        fc_model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)
    except Exception as ex:
        logger.error(f"Got an error while building vqa model:\n{ex}")
        raise

    return fc_model


def get_data(quick_load=True):
    meta_path = 'C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\data\\meta_data.h5'
    with pd.HDFStore(meta_path) as store:
        imaging_devices = store['imaging_devices']  # ct , mri

    devices = list(itertools.chain.from_iterable(imaging_devices.values))
    if quick_load:
        with pd.HDFStore('D:\\Users\\avitu\\Downloads\\temp.hdf') as store:
            train_data_raw = store['train_data_raw']

    else:
        from classes.vqa_model_predictor import VqaModelPredictor
        from common.DAL import get_model_by_id

        data_path = 'C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\data\\model_input.h5'
        with pd.HDFStore(data_path) as store:
            train_data = store['data']

        model_dal = get_model_by_id(80)
        mp = VqaModelPredictor(model_dal)
        vqa_data, imaging_data = mp.split_data_to_vqa_and_imaging(train_data)
        train_data_raw = imaging_data.drop_duplicates(subset=['image_name'])
        train_data_raw = train_data_raw[train_data_raw.imaging_device.isin(devices)]

        with pd.HDFStore('D:\\Users\\avitu\\Downloads\\temp.hdf') as store:
            store['train_data_raw'] =train_data_raw
            store['devices'] = devices

    df_train, df_test = train_test_split(train_data_raw, test_size=0.25)
    print(len(df_train), len(df_test))

    return df_train, df_test, devices




def retrain_model():
    df_train, df_test, devices = get_data()
    model = get_device_predictor_model(len(devices))

    X_train, X_test, y_train, y_test = extract_features(df_test, df_train, devices)

    # from:
    # >>> train_data_raw[['imaging_device']].describe()
    factor = 1.653276955602537
    class_weight = {0: factor,1: 1.}

    history = model.fit(X_train, y_train,
                        epochs=5,
                        batch_size=64,
                        validation_data=(X_test, y_test),
                        shuffle=True,
                        class_weight=class_weight)


    suffix = 'imaging_device_classifier'
    folder = os.path.join(vqa_models_folder, suffix)
    save_model(model=model, base_folder=folder, name_suffix=suffix, history=history)
    str(model)



def extract_features(df_test, df_train, devices):
    features_list = []
    labels_list = []
    for df in [df_train, df_test]:
        features = get_features(df)
        images_features = features[1]
        # text = features[0]
        # for t in text:
        #     t[:] = 0

        # labeler = LabelBinarizer()
        labeler = OneHotEncoder(sparse=False)
        labels = df.imaging_device.apply(lambda v: devices.index(v))
        labels = labels.reshape(-1, 1)

        y_train_vector = labeler.fit_transform(labels)
        y_train_vector = np.asarray(
            [list(l) for l in y_train_vector])  # list(itertools.chain.from_iterable(y_train_vector))

        features_list.append(images_features)
        labels_list.append(y_train_vector)
    X_train, y_train = features_list[0], labels_list[0]
    X_test, y_test = features_list[1], labels_list[1]
    return X_train, X_test, y_train, y_test


def main():
    retrain_model()

    data_set = ''


if __name__ == '__main__':
    main()
