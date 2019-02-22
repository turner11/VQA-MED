import logging
import keras.layers as keras_layers
from keras.layers import GlobalAveragePooling2D, Flatten  # , Dense, Input, Dropout
from keras.applications.vgg19 import VGG19
# from keras.applications.resnet50 import ResNet50
from keras import Model, Input  # ,models, callbacks
from keras.layers import Dense, LSTM, BatchNormalization, Activation  # GlobalAveragePooling2D, Merge, Embedding

from data_access.model_folder import ModelFolder
from common.settings import embedded_sentence_length, data_access
from common.model_utils import save_model, get_trainable_params_distribution
from common.constatns import vqa_models_folder
from evaluate.statistical import f1_score, recall_score, precision_score

DEFAULT_IMAGE_WEIGHTS = 'imagenet'
#  Since VGG was trained as a image of 224x224, every new image
# is required to go through the same transformation
image_size_by_base_models = {'imagenet': (224, 224)}

LSTM_UNITS = 64
POST_CONCAT_DENSE_UNITS = 16  # 64  # 256
DENSE_ACTIVATION = 'relu'
OPTIMIZER = 'rmsprop'

logger = logging.getLogger(__name__)


class VqaModelBuilder(object):
    """"""

    def __init__(self, loss_function: str, output_activation_function: str,
                 lstm_units: int = LSTM_UNITS,
                 post_concat_dense_units: int = POST_CONCAT_DENSE_UNITS,
                 optimizer: str = OPTIMIZER, categorical_data_frame_name: str = 'words'
                 ) -> None:
        """"""
        super(VqaModelBuilder, self).__init__()
        self.loss_function = loss_function
        self.output_activation_function = output_activation_function

        self.meta_dicts = data_access.load_meta()
        self.categorical_data_frame_name = categorical_data_frame_name
        self.categorical_data_frame = self.meta_dicts[self.categorical_data_frame_name]

        self.lstm_units = lstm_units
        self.post_concat_dense_units = post_concat_dense_units
        self.optimizer = optimizer

    @staticmethod
    def word_2_vec_model(input_tensor: object, lstm_units: int) -> object:
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
        if lstm_units > 0:
            x = LSTM(units=lstm_units
                     , return_sequences=False
                     , name='embedding_LSTM'
                     , input_shape=(1, embedded_sentence_length))(x)
        else:
            x = Flatten(name='embedding_Flattening')(x)

        x = BatchNormalization(name='embedding_batch_normalization')(x)
        model = x
        logger.debug("Done Creating Embedding model")
        return model

    @staticmethod
    def get_image_model(base_model_weights=DEFAULT_IMAGE_WEIGHTS):
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

        out_put_vals = self.categorical_data_frame
        model_output_num_units = len(out_put_vals)

        image_model, lstm_model, fc_model = None, None, None
        try:
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
            fc_tensors = Activation(DENSE_ACTIVATION)(fc_tensors)

            fc_tensors = Dense(units=model_output_num_units
                               , activation=self.output_activation_function
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
    def save_model(model: Model, prediction_df_name: str) -> ModelFolder:
        additional_info = {'prediction_data': prediction_df_name}
        model_folder: ModelFolder = save_model(model, vqa_models_folder, additional_info, data_access.fn_meta)
        # Copy meta data to local folder

        msg = f"Summary: {model_folder.summary_path}\n"
        msg += f"Image: {model_folder.image_file_path}\n"
        location_message = f"model_location = '{model_folder.model_path}'"

        logger.info(msg)
        logger.info(location_message)
        return model_folder

    @staticmethod
    def get_trainable_params_distribution(model, params_threshold=1000):
        return get_trainable_params_distribution(model, params_threshold)


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
