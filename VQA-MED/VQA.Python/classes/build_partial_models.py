import logging
from collections import namedtuple
from common.settings import data_access as common_data_access
from classes.vqa_model_builder import VqaModelBuilder
from classes.vqa_model_trainer import VqaModelTrainer
from common.utils import VerboseTimer
from data_access.api import SpecificDataAccess
from keras import backend as keras_backend

logger = logging.getLogger(__name__)


def train_partial_models():
    ModelCreationArgs = namedtuple('ModelCreationArgs', ['post_concat_dense_units', 'optimizer', 'loss', 'activation',
                                                         'prediction_vector_name', 'lstm_units', 'question_category'])

    args_recipe = [('answers', category) for category in ['Abnormality', 'Modality', 'Organ', 'Plane']]
    args_recipe.append(('words', 'Abnormality'))

    for recipe in args_recipe:
        # Building the model
        keras_backend.clear_session()
        prediction_vector_name = recipe[0]
        question_category = recipe[1]
        arg = ModelCreationArgs(post_concat_dense_units=8,
                                optimizer='RMSprop',
                                loss='cosine_proximity',
                                activation='sigmoid',
                                prediction_vector_name=prediction_vector_name,
                                lstm_units=0,
                                question_category=question_category)

        mb = VqaModelBuilder(arg.loss, arg.activation,
                             lstm_units=arg.lstm_units,
                             optimizer=arg.optimizer,
                             post_concat_dense_units=arg.post_concat_dense_units,
                             prediction_vector_name=arg.prediction_vector_name,
                             question_category=arg.question_category)

        model = mb.get_vqa_model()
        model_folder = VqaModelBuilder.save_model(model, mb.prediction_vector_name, question_category)

        logger.info(f'saved at {model_folder}')
        keras_backend.clear_session()

        # Training the model
        augmentations = 20
        epochs = 10
        batch_size = 64

        data_access = SpecificDataAccess.factory(common_data_access, question_category=model_folder.question_category)
        mt = VqaModelTrainer(model_folder, augmentations=augmentations, batch_size=batch_size, epochs=epochs,
                             data_access=data_access)
        meta = data_access.load_meta()




        history = mt.train()

        with VerboseTimer("Saving trained Model"):
            notes = f'{arg}\n' \
                f'augmentations: {augmentations} \n' \
                f'epochs: {epochs}\n' \
                f'batch size: {batch_size}\n' \
                f'based on model 5'

            logger.debug(notes)
            model_folder = VqaModelTrainer.save(mt.model, model_folder, history=history, notes=notes)

            logger.info(f'saved trained model to: {model_folder}')
            logger.debug(f'Model args:\n{arg}')


def main():
    train_partial_models()


if __name__ == '__main__':
    main()
