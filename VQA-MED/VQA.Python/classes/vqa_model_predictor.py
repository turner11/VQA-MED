from pathlib import Path
from typing import Union
from keras import Model as keras_model
import pandas as pd
import numpy as np
import itertools
from common.exceptions import InvalidArgumentException, NoDataException
from common.settings import data_access
from data_access.model_folder import ModelFolder
from common.DAL import get_models_data_frame, get_model_by_id, Model as ModelDal
from common.functions import get_features
from common.utils import VerboseTimer
import logging

logger = logging.getLogger(__name__)


# import common
# import importlib
# importlib.reload(common.functions)
class VqaModelPredictor(object):
    """"""

    def __init__(self, model: Union[str, int, ModelFolder]):
        """"""
        super(VqaModelPredictor, self).__init__()
        self.model, model_idx_in_db, model_folder = self.get_model(model)
        self.model_idx_in_db = model_idx_in_db

        self.model_folder = model_folder

    def __repr__(self):
        return super(VqaModelPredictor, self).__repr__()

    @staticmethod
    def get_model(model: Union[int, keras_model, str, None]) -> (keras_model, int, ModelFolder):

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
            model_location = str(model)
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
        prediction_vector = self.model_folder.prediction_vector
        df_predictions = self._predict_keras(df_data, self.model, words_decoder=prediction_vector,percentile=percentile)

        # Those are the mandatory columns
        sort_columns = ['image_name', 'question', 'answer', 'prediction', 'probabilities']
        ordered_columns = sorted(df_predictions.columns, key=lambda v: v in sort_columns, reverse=True)

        ret = df_predictions[ordered_columns]
        return ret

    @classmethod
    def _predict_keras(cls, df_data: pd.DataFrame, model, words_decoder, percentile: float) -> pd.DataFrame:
        features = get_features(df_data)
        with VerboseTimer("Raw model prediction"):
            p = model.predict(features)

        assert len(words_decoder) == len(p[0])
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
        for i, (curr_prediction, curr_probabilities) in enumerate(zip(predictions, probabilities)):
            prediction_df = pd.DataFrame({'word_idx': curr_prediction,
                                          'word': list(words_decoder.iloc[curr_prediction].values),
                                          'probabilities': curr_probabilities})

            curr_prediction_str = ' '.join([str(w) for w in list(prediction_df.word.values)])
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

    def __init__(self, model, df_test=None, df_validation=None):
        """"""
        super(DefaultVqaModelPredictor, self).__init__(model)
        df_test, df_validation = self.get_data(df_test, df_validation)
        self.df_validation = df_validation
        self.df_test = df_test

    @staticmethod
    def get_data(df_test=None, df_validation=None):
        if df_test is None:
            try:
                df_test = data_access.load_processed_data(group='test')
            except NoDataException as ex:
                logger.warning('No data found for test set')
                df_test = None
        if df_validation is None:
            df_validation = data_access.load_processed_data(group='validation')

        return df_test, df_validation


def main():
    # mp = VqaModelPredictor(model=None)
    # validation_prediction = mp.predict(mp.df_validation)
    # predictions = validation_prediction.prediction.values
    # ground_truth = validation_prediction.answer.values
    # results = VqaMedEvaluatorBase.get_all_evaluation(predictions=predictions, ground_truth=ground_truth)
    # print(f'Got results of {results}')
    pass


if __name__ == '__main__':
    main()
