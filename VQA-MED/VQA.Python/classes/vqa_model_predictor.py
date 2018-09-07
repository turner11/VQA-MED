import pandas as pd
import numpy as np

from evaluate.VqaMedEvaluatorBase import VqaMedEvaluatorBase
from common.DAL import get_models_data_frame, get_model_by_id
from pandas import HDFStore
from common.functions import get_features
from keras.models import load_model
from common.constatns import vqa_specs_location
from common.utils import VerboseTimer
from vqa_logger import logger
from common.os_utils import File
from evaluate.statistical import f1_score, recall_score, precision_score
from vqa_logger import logger


# import common
# import importlib
# importlib.reload(common.functions)


class VqaModelPredictor(object):
    """"""

    def __init__(self, model, df_test=None, df_validation=None):
        """"""
        super(VqaModelPredictor, self).__init__()

        self.vqa_specs = File.load_pickle(vqa_specs_location)
        self.model = self.get_model(model)
        df_test, df_validation = self.get_data(df_test, df_validation)
        self.df_validation = df_validation
        self.df_test = df_test

    def __repr__(self):
        return super(VqaModelPredictor, self).__repr__()

    def get_model(self, model):
        df_models = None
        if model is None:
            df_models = get_models_data_frame()
            model = max(df_models.id)
        if isinstance(model, int):
            model_idx_in_db = model
            df_models = df_models if df_models is not None else get_models_data_frame()
            notes = df_models.loc[df_models.id == model_idx_in_db].notes.values[0]
            logger.debug(f'Getting model #{model_idx_in_db} ({notes})')
            model_dal = get_model_by_id(model_idx_in_db)
            model_location = model_dal.model_location
            with VerboseTimer("Loading Model"):
                model = load_model(model_location, custom_objects={'f1_score': f1_score, 'recall_score': recall_score,
                                                                   'precision_score': precision_score})

        return model

    def get_data(self, df_test=None, df_validation=None):

        data_location = self.vqa_specs.data_location
        logger.debug(f"Loading test data from {data_location}")
        with VerboseTimer("Loading Test & validation Data"):
            with HDFStore(data_location) as store:
                df_test = df_test or store['test']

                if df_validation is None:
                    df_training = store['data']

                    # The validation is for evaluating
                    df_validation = df_training[df_training.group == 'validation'].copy()
                    del df_training

        return df_test, df_validation

    def predict(self, df_data: pd.DataFrame, percentile=99.8):

        # predict
        features = get_features(df_data)
        with VerboseTimer("Raw model prediction"):
            p = self.model.predict(features)

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
        df_data_light = pd.DataFrame(df_dict)

        meta_data_location = self.vqa_specs.meta_data_location
        df_meta_words = pd.read_hdf(meta_data_location, 'words')

        results = []
        for i, (curr_prediction, curr_propabilities) in enumerate(zip(predictions, probabilities)):
            prediction_df = pd.DataFrame({'word_idx': curr_prediction,
                                          'word': list(df_meta_words.loc[curr_prediction].word.values),
                                          'probabilities': curr_propabilities})

            curr_prediction_str = ' '.join([str(w) for w in list(prediction_df.word.values)])
            probabilities_str = ', '.join(['({:.3f})'.format(p) for p in list(prediction_df.probabilities.values)])

            light_pred_df = pd.DataFrame({
                'prediction': [curr_prediction_str],
                'probabilities': [probabilities_str]
            })
            results.append(light_pred_df)
        import itertools
        df_aggregated = pd.DataFrame({
            'prediction': list(itertools.chain.from_iterable([curr_df.prediction.values for curr_df in results])),
            'probabilities': [curr_df.probabilities.values for curr_df in results]
        })

        ret = df_data_light.merge(df_aggregated, how='outer', left_index=True, right_index=True)
        sort_columns = ['image_name', 'question', 'answer', 'prediction', 'probabilities']
        oredered_columns = sorted(ret.columns, key=lambda v: v in sort_columns, reverse=True)
        ret = ret[oredered_columns]
        return ret



def main():
    mp = VqaModelPredictor(model=None)
    validation_prediction = mp.predict(mp.df_validation)
    predictions = validation_prediction.prediction.values
    ground_truth = validation_prediction.answer.values
    results = VqaMedEvaluatorBase.get_all_evaluation(predictions=predictions, ground_truth=ground_truth)
    print(f'Got results of {results}')


if __name__ == '__main__':
    main()
