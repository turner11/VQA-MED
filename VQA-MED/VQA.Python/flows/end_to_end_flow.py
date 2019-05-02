import logging
import os
import shutil
from collections import OrderedDict, namedtuple
from pathlib import Path
from uuid import uuid4

from sqlalchemy.exc import IntegrityError
from tqdm import tqdm
import pandas as pd

from common import DAL
from common.DAL import ModelPartialScore
from common.utils import VerboseTimer
from data_access.api import SpecificDataAccess
from data_access.model_folder import ModelFolder

logger = logging.getLogger(__name__)


def _post_training_prediction(model_folder):
    from classes.vqa_model_predictor import DefaultVqaModelPredictor
    from evaluate.VqaMedEvaluatorBase import VqaMedEvaluatorBase

    model_dal = DAL.get_model(lambda dal: Path(dal.model_location).parent == model_folder.folder)
    model_id = model_dal.id
    mp = DefaultVqaModelPredictor(model_folder)

    data_sets = {'test': mp.df_test, 'validation': mp.df_validation}
    if model_folder.question_category:
        for name, df in data_sets.items():
            data_sets[name] = df[df.question_category == model_folder.question_category]

    predictions = {}
    for name, df in data_sets.items():
        with VerboseTimer(f"Predictions for VQA contender {name}"):
            df_predictions = mp.predict(df)
            predictions[name] = df_predictions

    outputs = {}
    for name, df_predictions in predictions.items():
        curr_predictions = df_predictions.prediction.values
        df_predicted = data_sets[name]
        df_output = df_predicted.copy()
        df_output['image_id'] = df_output.path.apply(lambda p: p.rsplit(os.sep)[-1].rsplit('.', 1)[0])
        df_output['prediction'] = curr_predictions

        columns_to_remove = ['path', 'answer_embedding', 'question_embedding', 'group', 'diagnosis', 'processed_answer']
        for col in columns_to_remove:
            del df_output[col]

        sort_columns = sorted(df_output.columns, key=lambda c: c not in ['question', 'prediction', 'answer'])
        df_output = df_output[sort_columns]
        outputs[name] = df_output

    df_output_test = outputs['test']
    df_output_validation = outputs['validation']

    def get_str(df_arg):
        # strs = []
        # debug_output_rows = df_arg.apply(lambda row: row.image_id + '|' + row.question + '|' + row.prediction, axis=1)
        output_rows = df_arg.apply(lambda row: row.image_id + '|' + row.prediction + '|' + row.answer, axis=1)
        output_rows = output_rows.str.strip('|')
        rows = output_rows.values
        res_value = '\n'.join(rows)
        return res_value

    res = get_str(df_output_test)
    res_val = get_str(df_output_validation)

    # Get evaluation per category:
    evaluations = {}
    pbar = tqdm(df_output_validation.groupby('question_category'))

    for question_category, df in pbar:
        pbar.set_description(f'evaluating {len(df)} for {question_category} items')
        curr_predictions = df.prediction.values
        curr_ground_truth = df.answer.values
        curr_evaluations = VqaMedEvaluatorBase.get_all_evaluation(predictions=curr_predictions,
                                                                  ground_truth=curr_ground_truth)
        evaluations[question_category] = curr_evaluations

    total_evaluations = VqaMedEvaluatorBase.get_all_evaluation(predictions=df_output_validation.prediction.values,
                                                               ground_truth=df_output_validation.answer.values)
    evaluations['Total'] = total_evaluations

    df_evaluations = pd.DataFrame(evaluations).T  # .sort_values(by=('bleu'))
    df_evaluations['sort'] = df_evaluations.index == 'Total'
    df_evaluations = df_evaluations.sort_values(by=['sort', 'wbss'])
    del df_evaluations['sort']

    # Getting string
    model_repr = repr(mp)
    sub_models = {category: folder for category, (model, folder) in mp.model_by_question_category.items()}
    sub_models_str = '\n'.join(
        [str(f'{category}: {folder} ({folder.prediction_data_name})') for category, folder in sub_models.items() if
         folder is not None])

    model_description_copy = df_evaluations.copy()

    def get_prediction_vector(category):
        sub_model = sub_models.get(category)
        if sub_model is not None:
            return sub_model.prediction_data_name
        else:
            return '--'

    model_description_copy['prediction_vector'] = model_description_copy.index.map(get_prediction_vector)

    model_description = f'''
    ==Model==
    {model_repr}

    ==Sub models==
    {sub_models_str}

    ==validation evaluation==
    {model_description_copy.to_string()}
    '''

    logger.debug(model_description)

    # Saving predictions
    submission_folder = model_folder.folder / 'submissions'
    if submission_folder.exists():
        shutil.copy(str(submission_folder), str(submission_folder) + '_' + str(uuid4()))

    submission_folder.mkdir()

    txt_path = submission_folder / f'submission.txt'
    txt_path.write_text(res)

    txt_path_val = submission_folder / f'submission_validation.txt'
    txt_path_val.write_text(res_val)

    model_description_path = submission_folder / f'model_description.txt'
    model_description_path.write_text(model_description)

    with pd.HDFStore(str(submission_folder / 'predictions.hdf')) as store:
        for name, df_predictions in predictions.items():
            store[name] = df_predictions

    logger.debug(f'For model {model_id}, Got results of\n{evaluations}')
    evaluations_types = {'wbss': 1, 'bleu': 2, 'strict_accuracy': 3}
    categories = OrderedDict({5: 'Abnormality_yes_no', 2: 'Plane', 3: 'Organ', 1: 'Modality', 4: 'Abnormality'})

    partial_scores: [ModelPartialScore] = DAL.get_partial_scores()

    for category_id, question_category in categories.items():
        evaluations_dict = evaluations.get(question_category)
        if not evaluations_dict:
            continue

        existing_evaluations = [ps for ps in partial_scores
                                if ps.model_id == model_id
                                and ps.question_category_id == category_id]

        for evaluation_name, score in evaluations_dict.items():
            evaluation_id = evaluations_types[evaluation_name]

            ps = ModelPartialScore(model_id, evaluation_id, category_id, score)

            existing_partials = [ev
                                 for ev in existing_evaluations
                                 if ev.evaluation_type == evaluation_id
                                 and ev.question_category_id == category_id]

            if len(existing_partials) != 0:
                logger.debug(f'for {ps}, already had a partial score. Continuing...')
                continue

            try:
                DAL.insert_dal(ps)
            except IntegrityError:
                logger.debug(f'for {ps}, value already existed')
            except Exception as ex:
                logger.exception(f'Failed to insert partial score to db (model: {model_id})')
                print(type(ex))

    return categories
    # insert_partial_scores(model_predicate=lambda m: m.id == model_db_id)


def generate_multi_configuration():
    BuildConfig = namedtuple('BuildConfig',
                             ['dense_units', 'lstm_units', 'use_text_inputs_attention', 'use_class_weight'])
    lstm_units = 128
    dense_units_collection = [
        (8,),
        (8, 7, 6),
        (7, 8, 6),(6, 9, 7),
        (6, 9),(8, 6),
        (7, 8, 9), (6, 8, 9), (8, 6, 7), (8, 9, 7), (8, 6, 9),
    ]
    use_text_inputs_attention = False  # True if i % 2 == 0 else True
    use_class_weight = False
    configs = [BuildConfig(dense_units=ds,
                           lstm_units=lstm_units,
                           use_text_inputs_attention=use_text_inputs_attention,
                           use_class_weight=use_class_weight)
               if not isinstance(ds, (BuildConfig,)) else ds
               for i, ds in enumerate(dense_units_collection)]
    configs = [BuildConfig(dense_units=(8, 7, 6), lstm_units=lstm_units, use_text_inputs_attention=True,
                           use_class_weight=True)] + configs
    question_category = 'Abnormality'
    for i, config in enumerate(configs):
        logger.info(f'Training : {config} ({i + 1} / {len(dense_units_collection)})')
        dense_units = config.dense_units
        use_text_inputs_attention = config.use_text_inputs_attention
        use_class_weight = config.use_class_weight

        curr_lstm_units = config.lstm_units
        epochs = 3  # 8 if len(dense_units) > 2 else 12

        folder_suffix = get_folder_suffix(question_category, dense_units, curr_lstm_units, use_class_weight,
                                          use_text_inputs_attention)

        _train_model(activation='softmax',
                     prediction_vector_name='answers',
                     epochs=epochs,
                     loss_function='categorical_crossentropy',
                     lstm_units=curr_lstm_units,
                     optimizer='RMSprop',
                     post_concat_dense_units=dense_units,
                     use_text_inputs_attention=use_text_inputs_attention,
                     question_category=question_category,
                     batch_size=32,
                     augmentations=20,
                     notes_suffix=f'For Category: {question_category}',
                     folder_suffix=folder_suffix,
                     use_class_weight=use_class_weight)


def get_folder_suffix(question_category, dense_units, lstm_units, use_class_weight, use_text_inputs_attention):

    folder_suffix = f'{question_category}_dense_{"_".join(str(v) for v in dense_units)}'
    if lstm_units:
        folder_suffix += f'_lstm_{lstm_units}'
    if use_text_inputs_attention:
        folder_suffix += f'_attention'
    if use_class_weight:
        folder_suffix += f'_weighted_class'
    return folder_suffix


def _train_model(activation, prediction_vector_name, epochs, loss_function, lstm_units, optimizer,
                 post_concat_dense_units,
                 use_text_inputs_attention,
                 question_category,
                 batch_size=75,
                 augmentations=20,
                 notes_suffix='',
                 folder_suffix='',
                 use_class_weight=False):
    # Doing all of this here in order to not import tensor flow for other functions
    from classes.vqa_model_trainer import VqaModelTrainer
    from classes.vqa_model_builder import VqaModelBuilder
    from common.settings import data_access as data_access_api
    from keras import backend as keras_backend
    # from classes.vqa_model_predictor import DefaultVqaModelPredictor
    # from evaluate.VqaMedEvaluatorBase import VqaMedEvaluatorBase

    keras_backend.clear_session()
    mb = VqaModelBuilder(loss_function, activation,
                         post_concat_dense_units=post_concat_dense_units,
                         use_text_inputs_attention=use_text_inputs_attention,
                         optimizer=optimizer,
                         lstm_units=lstm_units,
                         prediction_vector_name=prediction_vector_name,
                         question_category=question_category)
    model = mb.get_vqa_model()
    model_folder = VqaModelBuilder.save_model(model, prediction_vector_name, question_category, folder_suffix)
    # Train ------------------------------------------------------------------------

    keras_backend.clear_session()
    data_access = SpecificDataAccess(data_access_api.folder, question_category=question_category, group=None)
    mt = VqaModelTrainer(model_folder,
                         augmentations=augmentations,
                         batch_size=batch_size,
                         data_access=data_access,
                         epochs=epochs,
                         question_category=question_category,
                         use_class_weight=use_class_weight)
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
            f'epochs: {epochs}\n' \
            f'class weights: {use_class_weight}\n' \
            f'Inputs Attention: {use_text_inputs_attention}\n' \
            f'{notes_suffix}'

        trained_suffix = f'{folder_suffix}_trained'
        model_folder = mt.save(mt.model, mt.model_folder, history, notes=notes, folder_suffix=trained_suffix)
    logger.debug(f'model_folder: {model_folder}')

    # Evaluate ------------------------------------------------------------------------
    keras_backend.clear_session()

    # model_id_in_db = None  # latest...
    #
    # mp = DefaultVqaModelPredictor(model=model_id_in_db)
    # validation_prediction = mp.predict(mp.df_validation)
    # predictions = validation_prediction.prediction.values
    # ground_truth = validation_prediction.answer.values
    #
    # max_length = max([len(s) for s in predictions])
    # if max_length < 100:
    #     # results = VqaMedEvaluatorBase.get_all_evaluation(predictions=predictions, ground_truth=ground_truth)
    #     from evaluate.BleuEvaluator import BleuEvaluator
    #     ins = BleuEvaluator(predictions, ground_truth)
    #     results = {}
    #     results['bleu'] = ins.evaluate()
    #     results['wbss'] = -2
    # else:
    #     results = {'bleu': -1, 'wbss': -1}
    #
    # bleu = results['bleu']
    # wbss = results['wbss']
    #
    # model_db_id = mp.model_idx_in_db
    # model_score = ModelScore(model_db_id, bleu=bleu, wbss=wbss)
    # DAL.insert_dal(model_score)

    results = _post_training_prediction(model_folder)

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
                notes_suffix='',
                folder_suffix='',
                use_text_inputs_attention=False,
                use_class_weight=False):
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
                 use_text_inputs_attention=use_text_inputs_attention,
                 batch_size=batch_size,
                 notes_suffix=notes_suffix,
                 folder_suffix=folder_suffix,
                 question_category=question_category,
                 use_class_weight=use_class_weight)


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
                f'for model {model_id} prediction vector was "{model_folder.prediction_data_name}". '
                f'This might take a while')

        keras_backend.clear_session()
        categories = OrderedDict(
            {5: 'Abnormality_yes_no', 2: 'Plane', 3: 'Organ', 1: 'Modality', 4: 'Abnormality'}
        )
        evaluations = {'wbss': 1, 'bleu': 2, 'strict_accuracy': 3}

        for category_id, question_category in categories.items():
            data_access = SpecificDataAccess(data_access_api.folder, question_category=question_category, group=None)

            # ps: ModelPartialScore
            existing_evaluations = [ps for ps in partial_scores
                                    if ps.model_id == model_id
                                    and ps.question_category_id == category_id]

            if len(existing_evaluations) == len(evaluations):
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
                ps = ModelPartialScore(model_id, evaluation_id, category_id, score)

                existing_partials = [ev
                                     for ev in existing_evaluations
                                     if ev.evaluation_type == evaluation_id and ev.question_category_id == category_id]
                if len(existing_partials) != 0:
                    logger.debug(f'for {ps}, already had a partial score. Continuing...')
                    continue

                try:
                    DAL.insert_dal(ps)
                except IntegrityError:
                    logger.debug(f'for {ps}, value already existed')
                except Exception as ex:
                    logger.exception(f'Failed to insert partial score to db (model: {model_id})')
                    print(type(ex))
            # for evaluation_id, evaluation_type in evaluations.items():d
