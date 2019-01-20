from typing import Union
from keras import Model as keras_model
import pandas as pd
import numpy as np
import itertools
from evaluate.VqaMedEvaluatorBase import VqaMedEvaluatorBase
from common.DAL import get_models_data_frame, get_model_by_id, Model as ModelDal
from pandas import HDFStore
from common.functions import get_features
from keras.models import load_model
from common.constatns import vqa_specs_location
from common.utils import VerboseTimer

from common.os_utils import File
from evaluate.statistical import f1_score, recall_score, precision_score
import logging

logger = logging.getLogger(__name__)


# import common
# import importlib
# importlib.reload(common.functions)
class VqaModelPredictor(object):
    """"""

    def __init__(self, model):
        """"""
        super(VqaModelPredictor, self).__init__()

        self.vqa_specs = File.load_pickle(vqa_specs_location)
        self.model, model_idx_in_db = self.get_model(model)
        self.model_idx_in_db = model_idx_in_db

        pp = 'C:\\Users\\Public\\Documents\\Data\\2018\\vqa_models\\imaging_device_classifier\\20190111_1444_32_imaging_devince\\vqa_model_imaging_device_classifier.h5'
        self.image_device_classifier, model_id = self.get_model(pp)

    def __repr__(self):
        return super(VqaModelPredictor, self).__repr__()

    def get_model(self, model: Union[int, keras_model, str, None]) -> (keras_model, int):
        df_models = None
        model_id = -1
        model_idx_in_db = None
        model_dal = None

        if model is None:
            df_models = get_models_data_frame()
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

            with VerboseTimer("Loading Model"):
                model = load_model(model_location, custom_objects={'f1_score': f1_score, 'recall_score': recall_score,
                                                                   'precision_score': precision_score})

        if isinstance(model, keras_model):
            pass
        else:
            assert False, f'Expected model to be of type "{ModelDal.__name__}" or "{keras_model.__name__}"' \
                f'but got: "{model.__class__.__name__}" ({model})'
            # We are going to fail now
        assert model is not None, f'Unexpectedly got a None model\n(Model is "{type(model).__name__}"\n{model})'

        return model, model_id

    def predict(self, df_data: pd.DataFrame, percentile=99.8) -> pd.DataFrame:
        # predict

        vqa_data, imaging_data = self.split_data_to_vqa_and_imaging(df_data)

        df_predictions_imaging = self._predict_imaging_device(imaging_data)
        df_predictions_vqa = self._predict_vqa(vqa_data, percentile)

        df_predictions = pd.concat([df_predictions_vqa,df_predictions_imaging],
                                   ignore_index=False)\
                                   .sort_index()
        # Those are the mandatory columns
        sort_columns = ['image_name', 'question', 'answer', 'prediction', 'probabilities']
        ordered_columns = sorted(df_predictions.columns, key=lambda v: v in sort_columns, reverse=True)

        ret = df_predictions[ordered_columns]
        return ret

    def _predict_vqa(self, df_data: pd.DataFrame, percentile: float) -> pd.DataFrame:
        meta_data_location = self.vqa_specs.meta_data_location
        df_meta_words = pd.read_hdf(meta_data_location, 'words')
        return self._predict_keras(df_data, self.model, words_decoder=df_meta_words, percentile=percentile)

    @classmethod
    def _predict_keras(self, df_data: pd.DataFrame, model, words_decoder, percentile: float,
                       feature_prep: callable = None) -> pd.DataFrame:
        features = get_features(df_data)
        if feature_prep is not None:
            features = feature_prep(features)

        with VerboseTimer("Raw model prediction"):
            p = model.predict(features)

        assert len(words_decoder) == len(p[0])
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
                                          'word': list(words_decoder.iloc[curr_prediction].word.values),
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

    def _predict_imaging_device(self, df_data: pd.DataFrame) -> pd.DataFrame:
        meta_data_location = self.vqa_specs.meta_data_location
        words_decoder = pd.read_hdf(meta_data_location, 'imaging_devices')
        words_decoder = words_decoder.rename(columns={'imaging_device': 'word'})
        # HACK:
        words_decoder = pd.DataFrame({'word':['ct', 'mri']})#words_decoder[words_decoder.word != 'mra']

        def feature_prep(features):
            new_features = features[1]  # image only
            return new_features

        p = self._predict_keras(df_data,
                                model=self.image_device_classifier,
                                words_decoder=words_decoder,
                                percentile=1,
                                feature_prep=feature_prep)

        return p

    def split_data_to_vqa_and_imaging(self, df_data: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        imaging_data = df_data[df_data.is_imaging_device_question == 1]
        vqa_data = df_data[df_data.is_imaging_device_question == 0]

        return vqa_data, imaging_data


class DefaultVqaModelPredictor(VqaModelPredictor):
    """"""

    def __init__(self, model, df_test=None, df_validation=None):
        """"""
        super(DefaultVqaModelPredictor, self).__init__(model)
        df_test, df_validation = self.get_data(df_test, df_validation)
        self.df_validation = df_validation
        self.df_test = df_test

    def get_data(self, df_test=None, df_validation=None):
        data_location = self.vqa_specs.data_location
        logger.debug(f"Loading test data from {data_location}")
        with VerboseTimer("Loading Test & validation Data"):
            with HDFStore(data_location) as store:
                df_test = df_test if df_test is not None else store['test']

                if df_validation is None:
                    df_training = store['data']

                    # The validation is for evaluating
                    df_validation = df_training[df_training.group == 'validation'].copy()
                    del df_training

        return df_test, df_validation


def main():
    mp = VqaModelPredictor(model=None)
    validation_prediction = mp.predict(mp.df_validation)
    predictions = validation_prediction.prediction.values
    ground_truth = validation_prediction.answer.values
    results = VqaMedEvaluatorBase.get_all_evaluation(predictions=predictions, ground_truth=ground_truth)
    print(f'Got results of {results}')


if __name__ == '__main__':
    main()
