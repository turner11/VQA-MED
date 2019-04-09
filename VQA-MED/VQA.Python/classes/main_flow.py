import itertools
import logging
import os, sys
from collections import OrderedDict, namedtuple
from pathlib import Path

from tqdm import tqdm
from common import DAL
from common.DAL import ModelScore, ModelPartialScore
from common.utils import VerboseTimer
from data_access.model_folder import ModelFolder
import vqa_logger

sys.path.append('C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\')

logger = logging.getLogger(__name__)


def debug():
    from exported_notebooks import aaaa
    aaaa.do()
    return
    # from exported_notebooks import concise_train
    # concise_train.do()
    # return
    train_all()
    return


def remove_folders():
    existing_folders = Path('C:\\Users\\Public\\Documents\\Data\\2019\\models')
    folders = {d for d in existing_folders.iterdir() if str(d.name).replace('_', '').isdigit()}
    model_folder = {Path(m.model_location).parent for m in DAL.get_models()}
    diff = folders - model_folder
    a = sorted(list(diff), key=lambda p: len(str(p)))
    import shutil
    pbar = tqdm(a)
    for p in pbar:
        shutil.rmtree(str(p))


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
            logger.warning(f'for model {model_id}. The prediction vector was "{model_folder.prediction_data_name}". This might take a while')

        keras_backend.clear_session()
        categories = OrderedDict(
            {2: 'Plane', 3: 'Organ', 1: 'Modality', 4: 'Abnormality'}
        )
        evaluations = {'wbss': 1, 'bleu': 2}

        for category_id, question_category in categories.items():
            data_access = SpecificDataAccess(data_access_api.folder, question_category=question_category, group=None)

            ps: ModelPartialScore
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
            except:
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
                except:
                    logger.exception(f'Failed to insert partial score to db (model: {model_id})')
            # for evaluation_id, evaluation_type in evaluations.items():d


def evaluate_models(models):
    from keras import backend as keras_backend
    from data_access.api import SpecificDataAccess
    from classes.vqa_model_predictor import DefaultVqaModelPredictor
    from common.settings import data_access as data_access_api
    from evaluate.VqaMedEvaluatorBase import VqaMedEvaluatorBase

    ModelResults = namedtuple('ModelResults', ['model_id', 'category', 'evaluation', 'score', 'notes'])

    model_results = []
    pbar = tqdm(models)
    for model_id in pbar:
        pbar.set_description(f'working on model {model_id}')
        model_dal = DAL.get_model_by_id(model_id)

        assert model_id == model_dal.id
        model_folder_location = Path(model_dal.model_location).parent
        assert model_folder_location.is_dir()

        keras_backend.clear_session()
        model_folder = ModelFolder(model_folder_location)
        categories = OrderedDict({2: 'Plane', 3: 'Organ', 1: 'Modality', 4: 'Abnormality'})
        evaluations = {'wbss': 1, 'bleu': 2}

        for category_id, question_category in categories.items():

            data_access = SpecificDataAccess(data_access_api.folder, question_category=question_category,
                                             group=None)

            try:
                mp = DefaultVqaModelPredictor(model_folder, data_access=data_access)

                df_to_predict = mp.df_validation
                df_predictions = mp.predict(df_to_predict)
            except:
                logger.exception(f'Failed to predict (model {model_id})')
                continue
            predictions = df_predictions.prediction.values
            ground_truth = df_predictions.answer.values
            with VerboseTimer(f"Evaluation for model {model_id}, category: {question_category}"):
                results = VqaMedEvaluatorBase.get_all_evaluation(predictions=predictions, ground_truth=ground_truth)

            logger.debug(f'For {question_category} (model id: {model_id}), Got results of\n{results}')

            for evaluation_name, score in results.items():
                evaluation_id = evaluations[evaluation_name]

                mr = ModelResults(model_id, question_category, evaluation_name, score, model_dal.notes)
                model_results.append(mr)

            # for evaluation_id, evaluation_type in evaluations.items():d
    import pandas as pd
    df = pd.DataFrame(model_results)

    location_by_id = {mid: Path(DAL.get_model_by_id(mid).model_location) for mid in df.model_id.values}
    category_by_id = {mid: ModelFolder(model_location.parent).question_category for mid, model_location in
                      location_by_id.items() if model_location.exists()}
    full_categories = {mid: category_by_id.get(mid, 'NO FOLDER') for mid in category_by_id.keys()}

    df = df.sort_values(by=['category', 'score'], ascending=(True, False))
    df['model_category'] = df.model_id.apply(lambda mid: full_categories.get(mid, 'Failed'))
    df = df[sorted(df.columns, key=lambda c: c == 'notes')]
    str(df)

    pp = 'D:\\Users\\avitu\\Downloads\\evaluations.h5'
    with pd.HDFStore(pp) as store:
        store['data'] = df


def evaluate_contender():
    from classes.vqa_model_predictor import DefaultVqaModelPredictor
    from evaluate.VqaMedEvaluatorBase import VqaMedEvaluatorBase

    mp = DefaultVqaModelPredictor.get_contender()

    df_to_predict = mp.df_validation
    with VerboseTimer(f"Predictions for VQA contender"):
        df_predictions = mp.predict(df_to_predict)

    predictions = df_predictions.prediction.values
    ground_truth = df_predictions.answer.values
    with VerboseTimer(f"Evaluation for VQA contender"):
        results = VqaMedEvaluatorBase.get_all_evaluation(predictions=predictions, ground_truth=ground_truth)

    # folder = Path('D:\\Users\\avitu\\Downloads')
    # import pandas as pd
    # dd = pd.DataFrame({'predictions':predictions, 'ground_truth':ground_truth})
    # with pd.HDFStore(str(folder/'preds.h5')) as store:
    #     store['data'] = dd
    #
    logger.info(f"Evaluation for VQA contender: {results}")


def train_model(base_model_id,
                optimizer,
                post_concat_dense_units,
                lstm_units=0,
                question_category='Abnormality',
                epochs=20,
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
                 notes_suffix=notes_suffix)


def _train_model(activation, prediction_vector_name, epochs, loss_function, lstm_units, optimizer, post_concat_dense_units,
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
        results = {'bleu':-1, 'wbss':-1}

    bleu = results['bleu']
    wbss = results['wbss']


    model_db_id = mp.model_idx_in_db
    model_score = ModelScore(model_db_id, bleu=bleu, wbss=wbss)
    DAL.insert_dal(model_score)
    insert_partial_scores(model_predicate=lambda m: m.id > 72)
    logger.info('----------------------------------------------------------------------------------------')
    logger.info(f'@@@For:\tLoss: {loss_function}\tActivation: {activation}: Got results of {results}@@@')
    logger.info('----------------------------------------------------------------------------------------')


def evaluate_model(model=None):
    from evaluate.VqaMedEvaluatorBase import VqaMedEvaluatorBase
    from classes.vqa_model_predictor import DefaultVqaModelPredictor

    mp = DefaultVqaModelPredictor(model=model)
    validation_prediction = mp.predict(mp.df_validation)
    predictions = validation_prediction.prediction.values
    ground_truth = validation_prediction.answer.values
    results = VqaMedEvaluatorBase.get_all_evaluation(predictions=predictions, ground_truth=ground_truth)
    ms = ModelScore(model_id=mp.model_idx_in_db, bleu=results['bleu'], wbss=results['wbss'])
    DAL.insert_dal(ms)
    return results


def train_all():
    from common.settings import data_access
    # Doing all of this here in order to not import tensor flow for other functions
    from evaluate.VqaMedEvaluatorBase import VqaMedEvaluatorBase
    from classes.vqa_model_builder import VqaModelBuilder
    from classes.vqa_model_predictor import DefaultVqaModelPredictor
    from classes.vqa_model_trainer import VqaModelTrainer
    from keras import backend as keras_backend
    # Create------------------------------------------------------------------------
    ## good for a model to predict multiple mutually-exclusive classes:
    # loss, activation = 'categorical_crossentropy', 'softmax'
    # losses = ['categorical_crossentropy', 'binary_crossentropy', 'kullback_leibler_divergence', 'poisson',
    #           'cosine_proximity', 'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error',
    #           'mean_squared_logarithmic_error', 'squared_hinge', 'hinge', 'categorical_hinge', 'logcosh']
    # activations = ['softmax', 'sigmoid', 'relu', 'tanh']
    # losses_and_activations = list(itertools.product(losses, activations))

    optimizers = ['RMSprop', 'Adam']  # ['SGD', 'Adagrad', 'Adadelta', 'RMSprop', 'Adam']
    dense_units = [8, 16, 32, 64]
    lstm_units = [0, 8, 16]
    pred_vectors = ['answers', 'words']
    top_models = [
        ('cosine_proximity', 'sigmoid'),
        ('cosine_proximity', 'tanh'),
        ('cosine_proximity', 'relu'),
        ('cosine_proximity', 'softmax'),
        ('poisson', 'softmax'),
        ('kullback_leibler_divergence', 'softmax'),
        ('mean_absolute_percentage_error', 'relu'),
        ('mean_squared_logarithmic_error', 'relu'),
        ('logcosh', 'relu'),
        ('mean_squared_error', 'relu'),
        ('mean_absolute_error', 'relu'), ]

    la_units_opts = list(itertools.product(top_models, dense_units, optimizers, lstm_units, pred_vectors))

    existing_scores = DAL.get_scores()
    models_ids = [s.model_id for s in existing_scores]
    existing_models = DAL.get_models()
    models_with_scores = [m for m in existing_models if m.id in models_ids]
    # for loss, activation in losses_and_activations:

    batch_size = 64
    augmentations = 10
    la_units_opts = la_units_opts[60:]
    epochs=10
    pbar = tqdm(la_units_opts)
    for (loss, activation), post_concat_dense_units, opt, lstm, pred_vector in pbar:
        pbar.set_description(
            f'====== working on loss {loss}, activation {activation}, post_concat_dense_units {post_concat_dense_units}, opt {opt}, lstm {lstm}, pred_vector {pred_vector}====== ')
        keras_backend.clear_session()
        try:

            def match(m):
                notes = (m.notes or '')
                is_curr_model = m.loss_function == loss \
                                and m.activation == activation \
                                and opt in notes \
                                and str(post_concat_dense_units) in notes
                return is_curr_model

            match_model = next((m for m in models_with_scores if match(m)), None)
            if match_model is not None:
                print(f'Continuing for model:\n{match_model.notes}')
                continue

            # keras_backend.clear_session()

            _train_model(activation=activation,
                         prediction_vector_name=pred_vector,
                         epochs=epochs,
                         loss_function=loss,
                         lstm_units=lstm_units,
                         optimizer=opt,
                         post_concat_dense_units=post_concat_dense_units,
                         question_category=None)





            logger.debug(f"### Completed full flow for {loss} and {activation}")
        except Exception as ex:
            import traceback as tb
            logger.error(f"^^^ Failed full flow for {loss} and {activation}\n:{ex}")
            tb.print_exc()
            tb.print_stack()


def main():
    base_model_id = 72
    train_model(base_model_id=base_model_id ,
                optimizer='RMSprop',
                post_concat_dense_units=[8,8],
                lstm_units=0,
                epochs=4,
                notes_suffix=f'Based on #{base_model_id}, with class weights')
    # insert_partial_scores()
    # return
    # evaluate_contender()
    # return
    # evaluate_models(models=[68,69,70,71,72,66,67,21,27,39,33,41,13,6,3,15,55,72,46,50,26,32])
    # return
    # remove_folders()
    #
    # return
    # debug()
    # return

    # evaluate_model(162)
    # return
    # train_model(model_id=85, optimizer='Adam', post_concat_dense_units=16)
    pass


if __name__ == '__main__':
    # from evaluate.VqaMedEvaluatorBase import VqaMedEvaluatorBase
    # from classes.vqa_model_builder import VqaModelBuilder
    # from classes.vqa_model_predictor import DefaultVqaModelPredictor
    # from classes.vqa_model_trainer import VqaModelTrainer
    # from keras import backend as keras_backend
    #
    # mp = DefaultVqaModelPredictor(model=None)
    #
    # validation_prediction = mp.predict(mp.df_validation)
    # predictions = validation_prediction.prediction.values
    # ground_truth = validation_prediction.answer.values
    # results = VqaMedEvaluatorBase.get_all_evaluation(predictions=predictions, ground_truth=ground_truth)
    #
    # loss, activation =('mean_absolute_error', 'relu')
    # ms = ModelScore(model_id=mp.model_idx_in_db, bleu=results['bleu'], wbss=results['wbss'])
    #
    #
    # DAL.insert_dal(ms)
    #
    # model_fn =model_location
    # VqaModelTrainer.model_2_db(model, model_fn, fn_history=None, notes='')

    main()




















