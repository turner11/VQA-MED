import os
import warnings

import pandas as pd
from pandas import HDFStore
from vqa_logger import logger
from keras.models import load_model

from keras import callbacks as K_callbacks #, backend as keras_backend,
from common.functions import get_features, _concat_row, sentences_to_hot_vector, hot_vector_to_words
from common.constatns import data_location as default_data_location, vqa_models_folder, vqa_specs_location  # train_data, validation_data,
from common.utils import VerboseTimer
from common.classes import EarlyStoppingByAccuracy
from common.model_utils import save_model
from common.os_utils import File
from evaluate.statistical import f1_score, recall_score, precision_score
import numpy as np






class VqaModelTrainer(object):
    """"""

    @property
    def model(self):
        return self._model


    @property
    def class_df(self):
        return self.df_meta_words

    def __init__(self, model_location, epochs, batch_size, data_location=default_data_location):
        super(VqaModelTrainer, self).__init__()
        self.epochs = epochs
        self.batch_size = batch_size
        self.data_location = data_location
        self.model_location = model_location

        # ---- Getting model ----
        with VerboseTimer("Loading Model"):
            self._model = load_model(model_location,
                                     custom_objects={'f1_score': f1_score,
                                                     'recall_score': recall_score,
                                                      'precision_score': precision_score})

        # ---- Getting meta ----
        with VerboseTimer("Loading Meta"):
            vqa_specs = File.load_pickle(vqa_specs_location)
            meta_data_location = vqa_specs.meta_data_location
            self.df_meta_answers = pd.read_hdf(meta_data_location, 'answers')
            self.df_meta_words = pd.read_hdf(meta_data_location, 'words')
            self.df_meta_imaging_devices = pd.read_hdf(meta_data_location, 'imaging_devices')


            self.class_count = len(self.class_df)

        # ---- Getting Data ----
        logger.debug(f"Loading the data from {self.data_location}")
        with VerboseTimer("Loading Data"):
            with HDFStore(data_location) as store:
                df_data = store['data']


        self.data_train = df_data[df_data.group == 'train'].copy().reset_index()
        self.data_val = df_data[df_data.group == 'validation'].copy().reset_index()


    def get_labels(self, df: pd.DataFrame) -> iter:
        return sentences_to_hot_vector(df.answer, words_df=self.df_meta_words.word)


    def print_shape_sanity(self, features_t, labels_t, features_val, labels_val):
        print(f'Expectedt shape: {self.model.input_shape}')
        print('---------------------------------------------------------------------------')
        print(f'Actual training shape:{features_t[0].shape, features_t[1].shape}')
        print(f'Train Labels shape:{labels_t.shape}')
        print('---------------------------------------------------------------------------')
        print(f'Actual Validation shape:{features_val[0].shape, features_val[1].shape}')
        print(f'Validation Labels shape:{labels_val.shape}')

    def train(self):
        with VerboseTimer('Getting train features'):
            features_t = get_features(self.data_train)
        with VerboseTimer('Getting train labels'):
            labels_t = self.get_labels(self.data_train)
        with VerboseTimer('Getting train features'):
            features_val = get_features(self.data_val)
        with VerboseTimer('Getting validation labels'):
            labels_val = self.get_labels(self.data_val)

        self.print_shape_sanity(features_t, labels_t, features_val, labels_val)

        validation_input = (features_val, labels_val)

        model = self.model

        # train_features = image_name_question
        # validation_input = (validation_features, categorial_validation_labels)

        ## construct the image generator for data augmentation
        # aug = image.ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
        #                                height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
        #                                horizontal_flip=True, fill_mode="nearest")
        # train_generator = aug.flow(train_features, categorial_train_labels)

        # stop_callback = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=3, verbose=1,mode='auto')
        try:
            #     history = model.fit_generator(train_generator,
            #                                   validation_data=validation_input,
            #                                   steps_per_epoch=len(train_features) // self.batch_size,
            #                                   epochs=self.epochs,
            #                                   verbose=1,
            #                                   callbacks=[stop_callback],
            #                                   class_weight=class_weight
            #                                   )
            # verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.

            # import tensorflow as tf
            # import keras.backend.tensorflow_backend as ktf

            # def get_session(gpu_fraction=0.333):
            #     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction, allow_growth=True)
            #     return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

            stop_callback = K_callbacks.EarlyStopping(monitor='val_loss', min_delta=0.02, patience=0, verbose=1,
                                                    mode='auto')
            acc_early_stop = EarlyStoppingByAccuracy(monitor='accuracy', value=0.98, verbose=1)

            tensor_log_dir = os.path.abspath(os.path.join('.', 'tensor_board_logd'))
            File.validate_dir_exists(tensor_log_dir)
            tensor_board_callback = K_callbacks.TensorBoard(log_dir=tensor_log_dir)
            callbacks = [stop_callback, acc_early_stop, tensor_board_callback]

            with VerboseTimer("Training Model"):
                #         with get_session() as sess:
                #             ktf.set_session(sess)
                #             sess.run(tf.global_variables_initializer())

                history = model.fit(features_t, labels_t,
                                    epochs=self.epochs,
                                    batch_size=self.batch_size,
                                    validation_data=validation_input,
                                    shuffle=True,
                                    callbacks=callbacks)
        #             sess.close()

        except Exception as ex:
            logger.error("Got an error training model: {0}".format(ex))
            raise

        return history

    @staticmethod
    def save(model, history=None, notes=None):
        with VerboseTimer("Saving trained Model"):
            model_fn, summary_fn, fn_image, fn_history = save_model(model, vqa_models_folder,history=history)

        msg = f"Summary: {summary_fn}\n"
        msg += f"Image: {fn_image}\n"
        msg += f'History: {fn_history or "NONE"}\n'
        location_message = f"model_location = '{model_fn}'"

        print(msg)
        print(location_message)

        win_file_name = location_message.replace('\\', '\\\\')
        print(win_file_name)


        try:
            notes = f'{notes or ""}\n\n{location_message}'
            VqaModelTrainer.model_2_db(model, model_fn, fn_history, notes=notes)
        except Exception as ex:
            warnings.warn(f'Failed to insert model to DB:\n{ex}')
        return model_fn, summary_fn, fn_image, fn_history


    @staticmethod
    def model_2_db(model, model_fn, fn_history='', notes=''):
        from keras import backend as K
        from common import DAL
        from common.DAL import Model as DalModel


        h = model.history.history
        loss = model.loss
        out_put_layer = model.layers[-1]
        activation = out_put_layer.activation.__name__

        trainable_params = sum([np.prod(K.get_value(w).shape) for w in model.trainable_weights])
        all_params = sum([np.prod(K.get_value(w).shape) for w in model.weights])

        get_val = lambda key: next(iter(h.get(key, [])), None)

        dal_model = DalModel(
            model_location=model_fn,
            history_location=fn_history,
            image_base_net='vgg19',  # TODO: this is now used as constant
            loss=get_val('loss'),
            val_loss=get_val('val_loss'),
            accuracy=get_val('acc'),
            val_accuracy=get_val('val_acc'),
            notes=notes,
            parameter_count=int(all_params),
            trainable_parameter_count=int(trainable_params),
            f1_score=get_val('f1_score'),
            f1_score_val=get_val('val_f1_score'),
            recall=get_val('recall_score'),
            recall_val=get_val('val_recall_score'),
            precsision=get_val('precision_score'),
            precsision_val=get_val('val_precision_score'),
            loss_function=loss,
            activation=activation)

        DAL.insert_dal(dal_model)


def main():
    # epochs=25
    # batch_size = 20
    from keras import backend as keras_backend
    import numpy as np
    from common.model_utils import save_model, get_trainable_params_distribution
    keras_backend.clear_session()
    epochs = 1
    batch_size = 75

    model_location = 'C:\\Users\\Public\\Documents\\Data\\2018\\vqa_models\\20180831_1244_55\\vqa_model_.h5'
    mt = VqaModelTrainer(model_location, epochs=epochs, batch_size=batch_size)
    history = mt.train()
    with VerboseTimer("Saving trained Model"):
        model_fn, summary_fn, fn_image, fn_history = VqaModelTrainer.save(mt.model, history)
    print(model_fn)







if __name__ == '__main__':
    main()