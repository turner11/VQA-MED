# from common.constatns import _DB_FILE_LOCATION
from copy import deepcopy

import keras.layers as keras_layers
from keras import backend as keras_backend
from keras.layers import GlobalAveragePooling2D  # , Dense, Input, Dropout
from keras.applications.vgg19 import VGG19
# from keras.applications.resnet50 import ResNet50
from keras import Model, Input  # ,models, callbacks
from keras.layers import Dense, LSTM, BatchNormalization, Activation# GlobalAveragePooling2D, Merge, Flatten, Embedding

from vqa_logger import logger
from common.os_utils import File
from common.settings import embedded_sentence_length
from common.model_utils import save_model, get_trainable_params_distribution
from common.constatns import vqa_models_folder, vqa_specs_location
from evaluate.statistical import f1_score, recall_score, precision_score
import pandas as pd
import numpy as np

DEFAULT_IMAGE_WIEGHTS = 'imagenet'
#  Since VGG was trained as a image of 224x224, every new image
# is required to go through the same transformation
image_size_by_base_models = {'imagenet': (224, 224)}

LSTM_UNITS = 64
POST_CONCAT_DENSE_UNITS = 64#256
DENSE_ACTIVATION = 'relu'
OPTIMIZER = 'rmsprop'

class VqaModelBuilder(object):
    """"""

    CATEGORIAL_DATA_FRAME = 'words'

    def __init__(self, loss_function, output_activation_function,
                 lstm_units=LSTM_UNITS,
                 post_concat_dense_units=POST_CONCAT_DENSE_UNITS,
                 dense_activation=DENSE_ACTIVATION,
                 optimizer=OPTIMIZER
    ):
        """"""
        super(VqaModelBuilder, self).__init__()
        self.loss_function = loss_function
        self.output_activation_function = output_activation_function

        vqa_specs = File.load_pickle(vqa_specs_location)
        self.meta_data_location = vqa_specs.meta_data_location

        df_meta_answers, df_meta_words, df_meta_imaging_devices = self.__get_data_frames(self.meta_data_location)
        self.df_meta_answers = df_meta_answers
        self.df_meta_words = df_meta_words
        self.df_meta_imaging_devices = df_meta_imaging_devices

        self.lstm_units = lstm_units
        self.post_concat_dense_units = post_concat_dense_units
        self.dense_activation = dense_activation
        self.optimizer = optimizer

        self.model_location = ''

    @staticmethod
    def __get_data_frames(meta_data_location):
        with pd.HDFStore(meta_data_location, 'r') as hdf:
            keys = list(hdf.keys())
            print(f"meta Keys: {keys}")
        df_meta_answers = pd.read_hdf(meta_data_location, 'answers')
        df_meta_words = pd.read_hdf(meta_data_location, 'words')
        df_meta_imaging_devices = pd.read_hdf(meta_data_location, 'imaging_devices')

        return df_meta_answers, df_meta_words, df_meta_imaging_devices

    @staticmethod
    def word_2_vec_model(input_tensor,lstm_units):
        # print(dir(input_tensor))
        #         print('---------------------------------------------')
        #         print('Tensor shape: {0}'.format(input_tensor.get_shape()))
        #         print('---------------------------------------------')
        #         print(input_tensor.shape)
        #         print('---------------------------------------------')
        #         print('embedded_sentence_length: {0}'.format(embedded_sentence_length))
        #         print('---------------------------------------------')
        #         return


        logger.debug("Creating Embedding model")
        x = input_tensor  # Since using spacy

        x = LSTM(units=lstm_units, return_sequences=False, name='embbeding_LSTM',
                 input_shape=(1, embedded_sentence_length))(x)
        x = BatchNormalization(name='embbeding_batch_normalization')(x)

        #         x = LSTM(units=LSTM_UNITS, return_sequences=True, name='embbeding_LSTM_1',  input_shape=(1,embedded_sentence_length))(x)
        #         x = BatchNormalization(name='embbeding_batch_normalization_1')(x)
        #         x = LSTM(units=LSTM_UNITS, return_sequences=False, name='embbeding_LSTM_2')(x)
        #         x = BatchNormalization(name='embbeding_batch_normalization_2')(x)

        #         x = Dense(units=DENSE_UNITS, activation=DENSE_ACTIVATION)(x)
        model = x
        logger.debug("Done Creating Embedding model")
        return model

    @staticmethod
    def get_image_model(base_model_weights=DEFAULT_IMAGE_WIEGHTS):
        base_model_weights = base_model_weights

        base_model = VGG19(weights=base_model_weights, include_top=False)
        #     base_model = ResNet50(weights=base_model_weights, include_top=False)
        base_model.trainable = False
        for layer in base_model.layers:
            layer.trainable = False

        x = base_model.output
        # add a global spatial average pooling layer
        x = GlobalAveragePooling2D(name="image_model_average_pool")(x)

        model = x
        return base_model.input, model

    def get_vqa_model(self):
        metrics = [f1_score, recall_score, precision_score, 'accuracy']

        model_output_num_units = len(pd.read_hdf(self.meta_data_location, self.CATEGORIAL_DATA_FRAME))

        image_model, lstm_model, fc_model = None, None, None
        try:
            # ATTN:
            lstm_input_tensor = Input(shape=(embedded_sentence_length, 1), name='embedding_input')

            logger.debug("Getting embedding (lstm model)")
            lstm_model = self.word_2_vec_model(input_tensor=lstm_input_tensor, lstm_units=self.lstm_units)

            logger.debug("Getting image model")

            image_input_tensor, image_model = self.get_image_model()

            logger.debug("merging final model")
            # Available merge strategies: keras_layers.multiply, keras_layers.add, keras_layers.concatenate,
            # keras_layers.average, keras_layers.co, keras_layers.dot, keras_layers.maximum
            fc_tensors = keras_layers.concatenate([image_model, lstm_model])
            #         fc_tensors = BatchNormalization()(fc_tensors)
            fc_tensors = Dense(units=self.post_concat_dense_units)(fc_tensors)
            fc_tensors = BatchNormalization()(fc_tensors)
            fc_tensors = Activation(self.dense_activation)(fc_tensors)

            fc_tensors = Dense(units=model_output_num_units, activation=self.output_activation_function
                               , name=f'model_output_{self.output_activation_function}_dense')(fc_tensors)

            fc_model = Model(inputs=[lstm_input_tensor, image_input_tensor], output=fc_tensors)
            fc_model.compile(optimizer=self.optimizer, loss=self.loss_function, metrics=metrics)
        except Exception as ex:
            logger.error("Got an error while building vqa model:\n{0}".format(ex))
            models = [(image_model, 'image_model'), (lstm_model, 'lstm_model'), (fc_model, 'fc_model')]
            for m, name in models:
                if m is not None and hasattr(m, 'summary'):
                    logger.error("######################### {0} model details: ######################### ".format(name))
                    try:
                        m.summary(print_fn=logger.error)
                    except Exception as ex2:
                        logger.warning("Failed to print summary for {0}:\n{1}".format(name, ex2))
            raise

        return fc_model

    @staticmethod
    def save_model(model):
        model_fn, summary_fn, fn_image, _ = save_model(model, vqa_models_folder)

        msg = f"Summary: {summary_fn}\n"
        msg += f"Image: {fn_image}\n"
        location_message = f"model_location = '{model_fn}'"

        print(msg)
        print(location_message)
        return model_fn, summary_fn, fn_image

    @staticmethod
    def get_trainable_params_distribution(model, params_threshold=1000):
        return get_trainable_params_distribution(model, params_threshold)

    def __getstate__(self):
        state = {k:v for k, v in self.__dict__.items()}
        state['df_meta_answers'] = None
        state['df_meta_imaging_devices'] = None
        state['df_meta_words'] = None

    def __setstate__(self, state):
        self.__dict__.update(state)

        df_meta_answers, df_meta_words, df_meta_imaging_devices = self.__get_data_frames(self.meta_data_location)
        self.df_meta_answers = df_meta_answers
        self.df_meta_words = df_meta_words
        self.df_meta_imaging_devices = df_meta_imaging_devices



def main():
    # good for a model to predict multiple mutually-exclusive classes:
    # loss, activation = 'categorical_crossentropy', 'softmax'

    # loss, activation = 'binary_crossentropy', 'sigmoid'
    loss, activation = 'categorical_crossentropy', 'sigmoid'
    mb = VqaModelBuilder(loss, activation)
    model = mb.get_vqa_model()

    top_params = VqaModelBuilder.get_trainable_params_distribution(model)
    str(top_params)
    # model


if __name__ == '__main__':
    main()
