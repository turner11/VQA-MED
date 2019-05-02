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
from flows.end_to_end_flow import train_model, get_folder_suffix, _post_training_prediction

sys.path.append('C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\')

logger = logging.getLogger(__name__)


def debug():
    from exported_notebooks import aaa
    aaa.do()
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
    from flows.end_to_end_flow import _train_model
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
    # debug()
    # return


    ### DELETE================================================================
    # if True:
    #     from flows.end_to_end_flow import train_model
    #     dense_units = (8,7,6)
    #     lstm_units = 128
    #     question_category = 'Abnormality'
    #     use_class_weight = False
    #     use_text_inputs_attention =False
    #     folder_suffix = get_folder_suffix(question_category, dense_units, lstm_units , use_class_weight=use_class_weight,
    #                                       use_text_inputs_attention=use_text_inputs_attention )
    #     train_model(172,'RMSprop', dense_units ,lstm_units ,question_category,epochs=3,batch_size=32,
    #                 use_class_weight=use_class_weight,use_text_inputs_attention =use_text_inputs_attention,
    #                 folder_suffix=folder_suffix)
    # else:
    #     f = ModelFolder('C:\\Users\\Public\\Documents\\Data\\2019\\models\\20190430_0104_56_Abnormality_dense_8_7_6_lstm_128_trained')
    #     results = _post_training_prediction(f)
    ### ================================================================

    create_minor_categories_models()

    from flows.end_to_end_flow import generate_multi_configuration
    generate_multi_configuration()
    return
    # from flows.end_to_end_flow import insert_partial_scores
    # insert_partial_scores(model_predicate=lambda m: m.id >= 68)

    from flows.end_to_end_flow import train_model
    base_model_id = 68
    combs_2 = list(itertools.permutations(list(range(6,10)), 2))
    combs_3 = list(itertools.permutations(list(range(6, 10)), 3))
    combs = combs_2 + combs_3
    # combs.insert(0,[8,8])
    combs = [[8]] + combs
    combs = combs[29:]
    for dense_layers in combs:
        folder_suffix = 'dense_layers' + '_'.join([str(d) for d in dense_layers])
        train_model(base_model_id=base_model_id ,
                    optimizer='RMSprop',
                    post_concat_dense_units=dense_layers,
                    lstm_units=0,
                    epochs=5,
                    batch_size=32,
                    notes_suffix=f'Based on #{base_model_id}, with class weights',
                    folder_suffix=folder_suffix )

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





def create_minor_categories_models():
    from flows.end_to_end_flow import _train_model
    from flows.end_to_end_flow import get_folder_suffix
    lstm_units = 0
    use_class_weight = False
    use_text_inputs_attention = False
    default_batch_size = 32
    dense_units_collection = (
        # [8, 6],
        [8],
        [9,6], [4])
    for dense_units in dense_units_collection:
        MinorCategoryConfig = namedtuple('MinorCategoryConfig',['category', 'epochs', 'batch_size', 'dense_units'])
        for config in [MinorCategoryConfig('Abnormality_yes_no', 2, default_batch_size, dense_units ),
                       MinorCategoryConfig('Plane', 5, default_batch_size, dense_units ),
                       MinorCategoryConfig('Organ', 7, default_batch_size, dense_units ),
                       MinorCategoryConfig('Modality',10, default_batch_size, dense_units)
                       ]:
            question_category = config.category
            epochs = config.epochs
            batch_size = config.batch_size
            ds = config.dense_units

            folder_suffix = get_folder_suffix(question_category, dense_units, lstm_units, use_class_weight,
                                          use_text_inputs_attention)
            _train_model(activation='softmax',
                         prediction_vector_name='answers',
                         epochs=epochs,
                         loss_function='categorical_crossentropy',
                         lstm_units=lstm_units,
                         optimizer='RMSprop',
                         post_concat_dense_units=ds,
                         use_text_inputs_attention=use_text_inputs_attention,
                         question_category=question_category,
                         batch_size=batch_size,
                         augmentations=20,
                         notes_suffix=f'For Category: {question_category}',
                         folder_suffix=folder_suffix,
                         use_class_weight=use_class_weight,
                         )



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




















