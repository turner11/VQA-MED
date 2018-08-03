

import os
import numpy as np
from enum import Enum
import time
import datetime
import keras.layers as keras_layers
from vqa_logger import logger
from common.os_utils import File
from common.settings import classify_strategy, embedded_sentence_length, get_stratagy_str
from common.classes import ClassifyStrategies, VqaSpecs
from common.model_utils import save_model
from common.constatns import vqa_models_folder, vqa_specs_location
from keras import backend as keras_backend
from keras.applications.vgg19 import VGG19
from keras.layers import Dense, GlobalAveragePooling2D#, Input, Dropout

DEFAULT_IMAGE_WIEGHTS = 'imagenet'
image_size_by_base_models = {'imagenet': (224, 224)}
merge_strategy = keras_layers.concatenate

vqa_specs = File.load_pickle(vqa_specs_location)
meta_data = vqa_specs.meta_data






def get_image_model(base_model_weights=DEFAULT_IMAGE_WIEGHTS, out_put_dim=1024):
    base_model_weights = base_model_weights

    # base_model = VGG19(weights=base_model_weights,include_top=False)
    base_model = VGG19(weights=base_model_weights, include_top=False)
    base_model.trainable = False
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    # add a global spatial average pooling layer
    x = GlobalAveragePooling2D(name="image_model_average_pool")(x)
    # let's add a fully-connected layer
    x = Dense(out_put_dim, activation='relu',name="image_model_dense")(x)
    # and a logistic layer -- let's say we have 200 classes
    # predictions = Dense(200, activation='softmax')(x)
    model = x
    
    return base_model.input , model


def word_2_vec_model(input_tensor):
    # print(dir(input_tensor))
    print('---------------------------------------------')
    print(input_tensor.get_shape())
    print('---------------------------------------------')
    print(input_tensor.shape)
    print('---------------------------------------------')
    print(embedded_sentence_length)
    print('---------------------------------------------')
    # return
    # notes:
    # num works: scalar represents size of original corpus
    # embedding_dim : dim reduction. every input string will be encoded in a binary fashion using a vector of this length
    # embedding_matrix (AKA embedding_initializers): represents a pre trained network



    'ValueError: Input 0 is incompatible with layer embbeding_LSTM_1: expected ndim=3, found ndim=2'


    LSTM_UNITS = 512
    DENSE_UNITS = 1024
    DENSE_ACTIVATION = 'relu'

    logger.debug("Creating Embedding model")
    x = input_tensor  # Since using spacy

    # x = Embedding(num_words, embedding_dim, weights=[embedding_matrix], input_length=seq_length,trainable=False)(input_tensor)
    # x = LSTM(units=LSTM_UNITS, return_sequences=True, input_shape=(seq_length, embedding_dim))(x)
    # x = LSTM(units=LSTM_UNITS, return_sequences=True, name='embbeding_LSTM_1',input_shape=(1,embedded_sentence_length))(x)
    x = LSTM(units=LSTM_UNITS, return_sequences=True)(x)
    x = BatchNormalization(name='embbeding_batch_normalization_1')(x)
    x = LSTM(units=LSTM_UNITS, return_sequences=False, name='embbeding_LSTM_1')(x)
    x = BatchNormalization(name='embbeding_batch_normalization_1')(x)

    x = Dense(units=DENSE_UNITS, activation=DENSE_ACTIVATION)(x)
    model = x
    logger.debug("Done Creating Embedding model")
    return model




from keras import Model, models, Input, callbacks
from keras.utils import plot_model, to_categorical
from keras.layers import Dense, Embedding, LSTM, BatchNormalization#, GlobalAveragePooling2D, Merge, Flatten

def get_vqa_model(meta):
#     import tensorflow as tf
#     g = tf.Graph()
#     with g.as_default():
    DENSE_UNITS = 1000
    DENSE_ACTIVATION = 'relu'

    OPTIMIZER = 'rmsprop'
    LOSS = 'categorical_crossentropy'
    METRICS = 'accuracy'
    num_classes = len(meta['ix_to_ans'].keys())
    image_model, lstm_model, fc_model = None, None, None
    try:
        keras_backend.clear_session()
        # ATTN:
        lstm_input_tensor = Input(shape=(embedded_sentence_length,1), name='embedding_input')
        #lstm_input_tensor = Input(shape=(embedding_dim,), name='embedding_input')

        logger.debug("Getting embedding (lstm model)")
        # ----------------------------------------------------------------------------------------------
        LSTM_UNITS = 512
        x = lstm_input_tensor  # Since using spacy

        # x = Embedding(num_words, embedding_dim, weights=[embedding_matrix], input_length=seq_length,trainable=False)(input_tensor)
        # x = LSTM(units=LSTM_UNITS, return_sequences=True, input_shape=(seq_length, embedding_dim))(x)
        x = LSTM(units=LSTM_UNITS, return_sequences=True, name='embbeding_LSTM_1')(x)
        x = LSTM(units=LSTM_UNITS, return_sequences=True)(x)
        # ----------------------------------------------------------------------------------------------


        lstm_model = word_2_vec_model(input_tensor=lstm_input_tensor)

        logger.debug("Getting image model")
        out_put_dim = lstm_model.shape[-1].value
        image_input_tensor, image_model = get_image_model(out_put_dim=out_put_dim)


        logger.debug("merging final model")
        fc_tensors = merge_strategy(inputs=[image_model, lstm_model])
        fc_tensors = BatchNormalization()(fc_tensors)
        fc_tensors = Dense(units=DENSE_UNITS, activation=DENSE_ACTIVATION)(fc_tensors)
        fc_tensors = BatchNormalization()(fc_tensors)

        #ATTN:
        fc_tensors = Dense(units=embedded_sentence_length, activation='softmax', name='model_output_sofmax_dense')(fc_tensors)
        #fc_tensors = Dense(units=num_classes, activation='softmax', name='model_output_sofmax_dense')(fc_tensors)

        fc_model = Model(inputs=[lstm_input_tensor, image_input_tensor], output=fc_tensors)
        fc_model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=[METRICS])
    except Exception as ex:
        logger.error("Got an error while building vqa model:\n{0}".format(ex))
        models = [(image_model, 'image_model'), (lstm_model, 'lstm_model'), (fc_model, 'lstm_model')]
        for m, name in models:
            if m is not None:
                logger.error("######################### {0} model details: ######################### ".format(name))
                try:
                    m.summary(print_fn=logger.error)
                except Exception as ex2:
                    logger.warning("Failed to print summary for {0}:\n{1}".format(name, ex2))
        raise

    return fc_model




def main():
    keras_backend.clear_session()
    strat = get_stratagy_str()
    if classify_strategy == ClassifyStrategies.CATEGORIAL:
        model_output_num_units = len(list(meta_data['ix_to_ans'].keys()))
    elif classify_strategy == ClassifyStrategies.NLP:
        model_output_num_units = embedded_sentence_length
    else:
        raise Exception(f'Unfamilier strategy: {strat}')

    logger.debug(f'Model will have {model_output_num_units} output units (Strategy: {classify_strategy})')

    model = get_vqa_model(meta_data)
    model.describe()


if __name__ == '__main__':
    main()
