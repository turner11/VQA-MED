import os
import numpy as np
from sklearn.utils import class_weight as sklearn_class_weight
from keras.callbacks import History
from typing import Union

from classes.DataGenerator import DataGenerator
import logging
from data_access.api import DataAccess, SpecificDataAccess
from data_access.model_folder import ModelFolder

from keras import callbacks as K_callbacks, Model  # , backend as keras_backend,
from common.constatns import vqa_models_folder  # train_data, validation_data,
from common.utils import VerboseTimer
from common.model_utils import save_model, EarlyStoppingByAccuracy
from common.os_utils import File



from pre_processing.known_find_and_replace_items import diagnosis

logger = logging.getLogger(__name__)


class VqaModelTrainer(object):
    """"""

    @property
    def model(self):
        return self._model

    @property
    def class_df(self):
        return self.model_folder.prediction_vector

    @property
    def epochs(self):
        return self._epochs  # 1  # 20 if self.use_augmentation else 1

    def __init__(self, model_folder: ModelFolder, augmentations: int, batch_size: int,
                 data_access: DataAccess, epochs: int = 1, question_category: str = None,
                 use_class_weight: bool = True) -> None:
        super().__init__()

        self._epochs = epochs
        self.augmentations = augmentations

        self.batch_size = batch_size
        self.data_access = data_access

        self.model_folder = model_folder
        self._model = model_folder.load_model()
        self.model_location = str(model_folder.model_path)
        self.question_category = question_category

        # self.class_weight = self.get_diagnosis_class_weight()
        self.use_class_weight = use_class_weight

        # # ---- Getting Data ----
        # data_train: DataFrame = data_access.load_processed_data(group='train').reset_index()
        # data_val: DataFrame = data_access.load_processed_data(group='validation').reset_index()
        # # if question_category:
        # #     existing_categories = data_train.question_category.drop_duplicates().values
        # #     assert question_category in existing_categories , \
        # #         f'got a non existing question category ("{question_category}" not in {existing_categories})'
        # #     data_train = data_train[data_train.question_category == question_category ]
        # #     data_val = data_val[data_val.question_category == question_category]
        #
        # self.data_train = data_train
        # self.data_val = data_val

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(model_folder={self.model_folder}, augmentations={self.augmentations}, ' \
            f'batch_size={self.batch_size},data_access={self.data_access}, question_category={self.question_category})'

    def __str__(self) -> str:
        return f'VqaModelTrainer{"" if self.question_category is not None else ": " + str(self.question_category)}'

    def print_shape_sanity(self, features_t, labels_t, features_val, labels_val):
        logger.debug('===========================================================================')
        logger.debug(f'Expected shape: {self.model.input_shape}')
        logger.debug('---------------------------------------------------------------------------')
        logger.debug(f'Actual training shape:{features_t[0].shape, features_t[1].shape}')
        logger.debug(f'Actual Validation shape:{features_val[0].shape, features_val[1].shape}')
        logger.debug('---------------------------------------------------------------------------')
        logger.debug(f'Train Labels shape:{labels_t.shape}')
        logger.debug(f'Validation Labels shape:{labels_val.shape}')
        logger.debug('===========================================================================')

    def train(self):
        data_access_train = SpecificDataAccess.factory(self.data_access, group='train')
        data_access_val = SpecificDataAccess.factory(self.data_access, group='validation')

        prediction_vector = self.model_folder.prediction_vector

        dg = DataGenerator(data_access_train, prediction_vector=prediction_vector,
                           batch_size=self.batch_size,
                           augmentations=self.augmentations,
                           )

        data_val = data_access_val.load_processed_data()
        features_val, labels_val = DataGenerator._generate_data(data_val, prediction_vector)
        validation_input = (features_val, labels_val)

        model = self.model

        try:

            callbacks = self._get_call_backs()

            with VerboseTimer("Training Model"):
                features_t, labels_t = dg[0]
                self.print_shape_sanity(features_t, labels_t, features_val, labels_val)

                class_weight = None
                if self.use_class_weight:
                    df_data_prediction = self. data_access.load_processed_data(columns=['processed_answer', 'question_category', 'answer'])
                    df_data_prediction = df_data_prediction[df_data_prediction.group!= 'test']
                    # df_data_prediction = df_data_prediction[df_data_prediction.question_category == self.question_category]
                    train_classes = df_data_prediction.processed_answer.values
                    class_weight = self.get_auto_class_weight(prediction_vector=prediction_vector, train_classes=train_classes)

                history = model.fit_generator(generator=dg,
                                              validation_data=validation_input,
                                              epochs=self.epochs,
                                              callbacks=callbacks,
                                              use_multiprocessing=True,
                                              workers=3,
                                              class_weight=class_weight)

        #             sess.close()

        except Exception as ex:
            logger.exception('Got an error training model')
            raise
        return history

    def get_diagnosis_class_weight(self, diagnosis_class_weight=50):
        prediction_vector = self.model_folder.prediction_vector
        class_weight = self._get_diagnosis_class_weight(prediction_vector, diagnosis_class_weight)
        return class_weight

    @staticmethod
    def _get_diagnosis_class_weight(prediction_vector, diagnosis_class_weight=50):
        diagnosis_in_prediction_vector = set(prediction_vector.values).intersection(set(diagnosis))
        has_diagnosis = len(diagnosis_in_prediction_vector) > 0
        if has_diagnosis:
            class_weight = {i: 1 if label not in diagnosis else diagnosis_class_weight
                            for i, label
                            in enumerate(prediction_vector.values)}
        else:
            class_weight = None
        return class_weight

    @staticmethod
    def get_auto_class_weight(prediction_vector, train_classes):

        class_weights = sklearn_class_weight .compute_class_weight(
            'balanced',
            prediction_vector, #np.unique(prediction_vector),
            train_classes)

        ret = {i: weight for i, weight in enumerate(class_weights)}
        return ret


    def _get_call_backs(self):
        stop_callback = K_callbacks.EarlyStopping(monitor='val_loss', min_delta=0.02, patience=0, verbose=1,
                                                  mode='auto')
        acc_early_stop = EarlyStoppingByAccuracy(monitor='accuracy', value=0.98, verbose=1)
        tensor_log_dir = os.path.abspath(os.path.join('.', 'tensor_board_logd'))
        File.validate_dir_exists(tensor_log_dir)
        tensor_board_callback = None  # K_callbacks.TensorBoard(log_dir=tensor_log_dir)

        callbacks = [stop_callback, acc_early_stop, tensor_board_callback]
        callbacks = [c for c in callbacks if c is not None]


        model_check_point_file_path = str(self.model_folder.folder / 'model_check_point.{epoch: 02d}-{val_acc: .2f}.hdf5')

        save_model_call_back = K_callbacks.ModelCheckpoint(model_check_point_file_path, monitor='val_acc', verbose=0,
                                                           save_best_only=False,
                                                           save_weights_only=False, mode='auto', period=1)
        callbacks = [save_model_call_back]
        return callbacks

    @staticmethod
    def save(model: Model,
             base_model_folder: ModelFolder,
             history: History = None,
             notes: str = None,
             folder_suffix :str='') -> ModelFolder:
        with VerboseTimer("Saving trained Model"):
            model_folder: ModelFolder = save_model(model, vqa_models_folder,
                                                   base_model_folder.additional_info,
                                                   base_model_folder.meta_data_path,
                                                   history=history,
                                                   folder_suffix=folder_suffix)

        msg = f"Summary: {model_folder.summary_path}\n"
        msg += f"Image: {model_folder.image_file_path}\n"
        msg += f'History: {model_folder.history_path or "NONE"}\n'
        location_message = f"model_location = '{model_folder.model_path}'"

        logger.info(msg)
        logger.info(location_message)

        win_file_name = location_message.replace('\\', '\\\\')
        logger.info(win_file_name)

        try:
            notes = f'{notes or ""}\n\n{location_message}'
            VqaModelTrainer.model_2_db(model_folder, notes=notes)
        except Exception:
            logger.exception(f'Failed to insert model to DBץ Model folder is at: {model_folder}')

        return model_folder

    @staticmethod
    def model_2_db(model_folder: ModelFolder, notes=''):
        from keras import backend as K
        from common import DAL
        from common.DAL import Model as DalModel

        model = model_folder.load_model()

        try:
            h = model.history.history
        except:
            h = model_folder.history

        loss = model.loss
        out_put_layer = model.layers[-1]
        activation = out_put_layer.activation.__name__

        trainable_params = sum([np.prod(K.get_value(w).shape) for w in model.trainable_weights])
        all_params = sum([np.prod(K.get_value(w).shape) for w in model.weights])

        get_val = lambda key: next(iter(h.get(key, [])), None)

        dal_model = DalModel(
            model_location=str(model_folder.model_path),
            history_location=str(model_folder.history_path),
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
            activation=activation,
            class_strategy=model_folder.additional_info.get('prediction_data', None))

        DAL.insert_dal(dal_model)


def main():
    # from classes.vqa_model_predictor import DefaultVqaModelPredictor
    from keras import backend as keras_backend
    from common.settings import data_access as common_data_access
    keras_backend.clear_session()

    best_model_id = 5
    best_model_location = 'C:\\Users\\Public\\Documents\\Data\\2019\\models\\20190315_1614_49\\'
    model_folder = ModelFolder(best_model_location)

    # model_folder_path = 'C:\\Users\\Public\\Documents\\Data\\2019\\models\\20190219_0120_04'
    # model_folder = ModelFolder(model_folder_path)
    # model = model_folder.load_model()

    batch_size = 64
    epochs = 10
    augmentations = 20
    question_category = 'Abnormality'

    data_access = SpecificDataAccess.factory(common_data_access, question_category=model_folder.question_category)
    mt = VqaModelTrainer(model_folder, augmentations=augmentations, batch_size=batch_size, epochs=epochs,
                         data_access=data_access)
    history = mt.train()

    with VerboseTimer("Saving trained Model"):
        model_folder = VqaModelTrainer.save(mt.model, model_folder, history=history,
                                            notes='Abnormality model\n20 augmentations\nbased on model 5')

    str()


if __name__ == '__main__':
    main()
