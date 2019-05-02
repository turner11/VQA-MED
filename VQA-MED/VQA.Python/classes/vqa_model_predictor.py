import logging
import pandas as pd
import numpy as np
from collections import defaultdict
import itertools
from pathlib import Path
from typing import Union

import tqdm
from keras import Model as keras_model

from common.constatns import questions_classifiers
from common.exceptions import InvalidArgumentException  # , NoDataException
from common.os_utils import File
from common.settings import data_access as data_acces_api
from data_access.model_folder import ModelFolder
from common.DAL import get_models_data_frame, get_model_by_id, Model as ModelDal
from common.functions import get_features
from common.utils import VerboseTimer

logger = logging.getLogger(__name__)


# import common
# import importlib
# importlib.reload(common.functions)
class VqaModelPredictor(object):
    """"""

    def __init__(self, model: Union[str, int, ModelFolder, keras_model, None], specialized_classifiers=None):
        """"""
        super().__init__()
        self.__model_arg = model
        self.__specialized_classifiers_arg = specialized_classifiers
        self.model, model_idx_in_db, model_folder = self.get_model(model)
        if model_folder.question_category:
            logger.warning(f'Expected main model to be with no question category, but got:'
                           f' "{model_folder.question_category}"')

        self.model_idx_in_db = model_idx_in_db

        self.model_folder = model_folder
        self.model_by_question_category = {}

        specialized_classifiers = specialized_classifiers or {}
        self.model_by_question_category = {}
        question_categories = questions_classifiers.keys()
        bad_category_keys = [k for k in specialized_classifiers.keys() if k not in question_categories]
        assert len(bad_category_keys) == 0, f'Got unexpected question categories classifiers: {bad_category_keys}'
        for category in question_categories:
            clf = specialized_classifiers.get(category)
            clf_model_folder = None
            if clf is not None:
                clf, clf_model_idx_in_db, clf_model_folder = self.get_model(clf)
                logging.debug(f'For {category}, got specialized model (DB: {clf_model_idx_in_db}, Folder: {clf_model_folder})')
                assert clf_model_folder.question_category is not None, 'expected specific model to have speciality'
            self.model_by_question_category[category] = (clf, clf_model_folder)

    def __repr__(self):
        return f'VqaModelPredictor(model={self.__model_arg}, specialized_classifiers={self.__specialized_classifiers_arg})'



    @staticmethod
    def get_model(model: Union[int, keras_model, ModelFolder, str, None]) -> (keras_model, int, ModelFolder):

        df_models = None
        model_id = -1
        model_idx_in_db = None
        model_dal = None

        if model is None:
            df_models = get_models_data_frame().sort_values(by='id', ascending=False)
            model = max(df_models.id)

        if isinstance(model, int):
            model_idx_in_db = model
            df_models = df_models if df_models is not None else get_models_data_frame()
            notes = df_models.loc[df_models.id == model_idx_in_db].notes.values[0]
            logger.debug(f'Getting model #{model_idx_in_db} ({notes})')
            model_dal = get_model_by_id(model_idx_in_db)
            model = model_dal

        if isinstance(model, (ModelDal, str)):
            if isinstance(model, ModelDal):
                model_dal = model
                model_location = model_dal.model_location
                model_id = model_dal.id
            else:
                model_location = model

        elif isinstance(model, ModelFolder):
            model_location = str(model.folder)
        else:
            raise InvalidArgumentException('model', arument=model)

        model_location = Path(model_location)
        assert model_location.exists()
        model_location = model_location if model_location.is_dir() else model_location.parent

        model_folder = ModelFolder(model_location)
        model = model_folder.load_model()

        return model, model_id, model_folder

    def predict(self, df_data: pd.DataFrame, percentile=99.8) -> pd.DataFrame:
        # predict
        general_prediction_vector = self.model_folder.prediction_vector
        predictions = {}
        for category, args in self.model_by_question_category.items():
            if args is None or not all(args):
                logger.info(f'Category "{category}" had no specialized classifier. using general model...')
                vqa_model = self.model
                prediction_vector = general_prediction_vector
            else:
                (specific_vqa_model, specific_model_folder) = args
                specific_model_predictions_vector = specific_model_folder.prediction_vector

                is_expecting_smaller_prediction_vector = self.model_folder.question_category is None
                if is_expecting_smaller_prediction_vector:
                    # assert len(specific_model_predictions_vector) < len(general_prediction_vector)
                    # Commenting out in order to be able to use models with older meta and mixture of words / answer...
                    pass

                logger.info(f'For Category "{category}" using specialized classifier from:\n{specific_model_folder}')

                vqa_model = specific_vqa_model
                prediction_vector = specific_model_predictions_vector

            logger.debug(f'Classifying: "{category}"')
            relevant_idxs = df_data.question_category == category
            df_relevant = df_data[relevant_idxs]
            if len(df_relevant) > 0:

                df_specific_predictions = self._predict_keras(df_relevant,
                                                              vqa_model,
                                                              words_decoder=prediction_vector,
                                                              percentile=percentile)
            else:
                logger.warning(f'Did not get any data for category "{category}"')
                continue

            predictions[category] = df_specific_predictions

        df_predictions = pd.concat(predictions.values())

        ## Converting answers to human style (de tokenizing)
        if self.model_folder.prediction_data_name == 'answers':
            df_conversions = data_acces_api.load_processed_data(columns=['answer', 'processed_answer'])
            df_conversions = df_conversions[df_conversions.processed_answer.str.len() > 0]
            # removing duplicates
            df_conversions = df_conversions.set_index('processed_answer')
            df_conversions = df_conversions[~df_conversions.index.duplicated(keep='first')]
            df_predictions['prediction'] = df_predictions.prediction.apply(lambda p: df_conversions.loc[p].answer)

        # Those are the mandatory columns
        sort_columns = ['image_name', 'question', 'answer', 'prediction', 'probabilities']
        ordered_columns = \
            sorted(df_predictions.columns,
                   key=lambda v: (v not in sort_columns, sort_columns.index(v) if v in sort_columns else 100),
                   reverse=False)

        ret = df_predictions[ordered_columns].sort_index()
        return ret

    @classmethod
    def _predict_keras(cls, df_data: pd.DataFrame, model, words_decoder, percentile: float) -> pd.DataFrame:
        features = get_features(df_data)
        with VerboseTimer("Raw model prediction"):
            p = model.predict(features)

        assert len(words_decoder) == len(p[0]), f'Expected decoder ({len(words_decoder)}) to be in the same length of probabilities ({len(p[0])})'
        allow_multi_predictions = all(len(txt.split()) <= 1 for txt in words_decoder.values)

        # noinspection PyTypeChecker
        percentiles = [np.percentile(curr_pred, percentile) for curr_pred in p]
        enumrated_p = [[(i, v) for i, v in enumerate(curr_p)] for curr_p in p]
        pass_vals = [([(i, curr_pred) for i, curr_pred in curr_pred_arr if curr_pred >= curr_percentile])
                     for curr_pred_arr, curr_percentile in zip(enumrated_p, percentiles)]
        # [(i,len(curr_pass_arr)) for i, curr_pass_arr in  pass_vals]
        # vector-to-value. First - the results, second - the propabilities
        predictions = [[i for i, curr_p in curr_pass_arr] for curr_pass_arr in pass_vals]
        probabilities = [[curr_p for i, curr_p in curr_pass_arr] for curr_pass_arr in pass_vals]
        # dictionary for creating a data frame
        cols_to_transfer = ['image_name', 'question', 'answer', 'path']
        df_dict = {col_name: df_data[col_name] for col_name in cols_to_transfer}
        df_data_light = pd.DataFrame(df_dict).reset_index()

        results = []
        pbar = tqdm.tqdm(enumerate(zip(predictions, probabilities)), total=len(predictions))
        for i, (curr_prediction, curr_probabilities) in pbar:
            pbar.set_description(f'Prediction: {str(curr_prediction)[:20]}; probabilities: {str(curr_probabilities)[:20]}')
            prediction_df = pd.DataFrame({'word_idx': curr_prediction,
                                          'prediction': list(words_decoder.iloc[curr_prediction].str.strip().values),
                                          'probabilities': curr_probabilities}
                                         ).sort_values(by='probabilities', ascending=False).reset_index(drop=True)


            if not allow_multi_predictions:
                prediction_df = prediction_df.head(1)

            curr_prediction_str = ' '.join([str(w) for w in list(prediction_df.prediction.values)])
            probabilities_str = ', '.join(['({:.3f})'.format(p) for p in list(prediction_df.probabilities.values)])

            light_pred_df = pd.DataFrame({
                'prediction': [curr_prediction_str],
                'probabilities': [probabilities_str]
            })
            results.append(light_pred_df)

        df_aggregated = pd.DataFrame({
            'prediction': list(itertools.chain.from_iterable([curr_df.prediction.values for curr_df in results])),
            'probabilities': [curr_df.probabilities.values for curr_df in results]
        })
        ret = df_data_light.merge(df_aggregated, how='outer', left_index=True, right_index=True)
        ret = ret.set_index('index')
        return ret


class DefaultVqaModelPredictor(VqaModelPredictor):
    """"""

    def __init__(self, model: Union[str, int, ModelFolder, keras_model, None], data_access=None, specialized_classifiers=None):
        """"""
        super().__init__(model, specialized_classifiers=specialized_classifiers)

        self.data_access = data_access or data_acces_api
        df_test, df_validation = self.get_data(self.data_access)
        self.df_validation = df_validation
        self.df_test = df_test

    @staticmethod
    def get_data(data_access):
        df_test = data_access.load_processed_data(group='test')
        df_validation = data_access.load_processed_data(group='validation')
        return df_test, df_validation

    @staticmethod
    def get_contender():
        main_model = 5
        specialized_classifiers = {'Abnormality': 72, 'Modality': 69, 'Organ': 70, 'Plane': 71}
        main_model = 78
        specialized_classifiers = {'Abnormality': main_model , 'Modality': 69, 'Organ': 70, 'Plane': 71}

        main_model = 68
        specialized_classifiers = {'Abnormality': main_model, 'Modality': 69, 'Organ': 70, 'Plane': 71}
        with VerboseTimer(f"Loading  VQA contender"):
            vqa_contender = DefaultVqaModelPredictor(model=main_model, specialized_classifiers=specialized_classifiers)
        return vqa_contender


def main():
    pass
    # from evaluate.VqaMedEvaluatorBase import VqaMedEvaluatorBase
    # mp = VqaModelPredictor(model=None)
    # validation_prediction = mp.predict(mp.df_validation)
    # predictions = validation_prediction.prediction.values
    # ground_truth = validation_prediction.answer.values
    # results = VqaMedEvaluatorBase.get_all_evaluation(predictions=predictions, ground_truth=ground_truth)
    # print(f'Got results of {results}')


if __name__ == '__main__':
    main()
