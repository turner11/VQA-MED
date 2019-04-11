import logging
from collections import OrderedDict
from pathlib import Path

from tqdm import tqdm

from common import DAL
from common.DAL import ModelScore, ModelPartialScore
from common.utils import VerboseTimer
from data_access.model_folder import ModelFolder

logger = logging.getLogger(__name__)


def _train_model(activation, prediction_vector_name, epochs, loss_function, lstm_units, optimizer,
                 post_concat_dense_units,
                 question_category,
                 batch_size=75,
                 augmentations=20,
                 notes_suffix=''):
    # Doing all of this here in order to not import tensor flow for other functions
    from classes.vqa_model_trainer import VqaModelTrainer
    from classes.vqa_model_builder import VqaModelBuilder
    from common.settings import data_access
    from keras import backend as keras_backend
    from classes.vqa_model_predictor import DefaultVqaModelPredictor
    from evaluate.VqaMedEvaluatorBase import VqaMedEvaluatorBase

    keras_backend.clear_session()
    mb = VqaModelBuilder(loss_function, activation,
                         post_concat_dense_units=post_concat_dense_units,
                         optimizer=optimizer,
                         lstm_units=lstm_units,
                         prediction_vector_name=prediction_vector_name,
                         question_category=question_category)
    model = mb.get_vqa_model()
    model_folder = VqaModelBuilder.save_model(model, prediction_vector_name, question_category)
    # Train ------------------------------------------------------------------------
    keras_backend.clear_session()
    mt = VqaModelTrainer(model_folder,
                         augmentations=augmentations,
                         batch_size=batch_size,
                         data_access=data_access,
                         epochs=epochs,
                         question_category=question_category)
    history = mt.train()
    # Train ------------------------------------------------------------------------
    with VerboseTimer("Saving trained Model"):
        notes = f'post_concat_dense_units: {post_concat_dense_units};\n' \
            f'Optimizer: {optimizer}\n' \
            f'loss: {loss_function}\n' \
            f'activation: {activation}\n' \
            f'prediction vector: {prediction_vector_name}\n' \
            f'lstm_units: {lstm_units}\n' \
            f'batch_size: {batch_size}\n' \
            f'{notes_suffix}'

        model_folder = mt.save(mt.model, mt.model_folder, history, notes=notes)
    logger.debug(f'model_folder: {model_folder}')

    # Evaluate ------------------------------------------------------------------------
    keras_backend.clear_session()

    model_id_in_db = None  # latest...

    mp = DefaultVqaModelPredictor(model=model_id_in_db)
    validation_prediction = mp.predict(mp.df_validation)
    predictions = validation_prediction.prediction.values
    ground_truth = validation_prediction.answer.values

    max_length = max([len(s) for s in predictions])
    if max_length < 100:
        results = VqaMedEvaluatorBase.get_all_evaluation(predictions=predictions, ground_truth=ground_truth)
    else:
        results = {'bleu': -1, 'wbss': -1}

    bleu = results['bleu']
    wbss = results['wbss']

    model_db_id = mp.model_idx_in_db
    model_score = ModelScore(model_db_id, bleu=bleu, wbss=wbss)
    DAL.insert_dal(model_score)
    insert_partial_scores(model_predicate=lambda m: m.id > 72)
    logger.info('----------------------------------------------------------------------------------------')
    logger.info(f'@@@For:\tLoss: {loss_function}\tActivation: {activation}: Got results of {results}@@@')
    logger.info('----------------------------------------------------------------------------------------')


def train_model(base_model_id,
                optimizer,
                post_concat_dense_units,
                lstm_units=0,
                question_category='Abnormality',
                epochs=20,
                batch_size=75,
                notes_suffix=''):
    # Get------------------------------------------------------------------------
    model_dal = DAL.get_model_by_id(model_id=base_model_id)
    loss_function = model_dal.loss_function
    activation = model_dal.activation
    prediction_vector_name = model_dal.class_strategy

    _train_model(activation,
                 prediction_vector_name,
                 epochs,
                 loss_function,
                 lstm_units,
                 optimizer,
                 post_concat_dense_units,
                 question_category,
                 batch_size=batch_size,
                 notes_suffix=notes_suffix)


# noinspection PyBroadException
def insert_partial_scores(model_predicate=None):
    from common.settings import data_access as data_access_api
    from data_access.api import SpecificDataAccess
    from classes.vqa_model_predictor import DefaultVqaModelPredictor
    from evaluate.VqaMedEvaluatorBase import VqaMedEvaluatorBase
    from keras import backend as keras_backend

    all_models = DAL.get_models()
    all_models = [m for m in all_models if model_predicate is None or model_predicate(m)]
    partial_scores: [ModelPartialScore] = DAL.get_partial_scores()

    pbar = tqdm(all_models)
    for model in pbar:
        model_id = model.id
        pbar.set_description(f'Working on model {model_id}')

        model_folder_location = Path(model.model_location).parent

        if not model_folder_location.is_dir():
            continue

        model_folder = ModelFolder(model_folder_location)
        if model_folder.prediction_data_name != 'answers':
            # logger.warning(
            #     f'Skipping model {model_id}. The prediction vector was "{model_folder.prediction_data_name}"')
            # continue
            logger.warning(
                f'for model {model_id}m prediction vector was "{model_folder.prediction_data_name}". '
                f'This might take a while')

        keras_backend.clear_session()
        categories = OrderedDict(
            {2: 'Plane', 3: 'Organ', 1: 'Modality', 4: 'Abnormality'}
        )
        evaluations = {'wbss': 1, 'bleu': 2}

        for category_id, question_category in categories.items():
            data_access = SpecificDataAccess(data_access_api.folder, question_category=question_category, group=None)

            # ps: ModelPartialScore
            existing_evaluations = [ps for ps in partial_scores
                                    if ps.model_id == model_id
                                    and ps.question_category_id == category_id]

            if len(existing_evaluations) > 0:
                logger.debug(f'Model {model_id} had evaluations for "{question_category} , '
                             f'got {len(existing_evaluations)} partial results. Continuing')
                continue

            try:
                mp = DefaultVqaModelPredictor(model_folder, data_access=data_access)

                df_to_predict = mp.df_validation
                df_predictions = mp.predict(df_to_predict)
            except Exception:
                logger.exception(f'Failed to predict (model {model_id})')
                continue

            predictions = df_predictions.prediction.values
            ground_truth = df_predictions.answer.values
            results = VqaMedEvaluatorBase.get_all_evaluation(predictions=predictions, ground_truth=ground_truth)

            logger.debug(f'For {question_category} (model id: {model_id}), Got results of\n{results}')

            for evaluation_name, score in results.items():
                evaluation_id = evaluations[evaluation_name]
                try:
                    ps = ModelPartialScore(model_id, evaluation_id, category_id, score)
                    DAL.insert_dal(ps)
                except Exception:
                    logger.exception(f'Failed to insert partial score to db (model: {model_id})')
            # for evaluation_id, evaluation_type in evaluations.items():d
