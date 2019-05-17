import math
import os
import logging
from collections import OrderedDict
import string

import numpy as np
from functools import partial
from nltk.corpus import stopwords
import dask.dataframe as dd


from common.os_utils import File
from common.utils import VerboseTimer
from common.settings import input_length, get_nlp, embedding_dim
from common.constatns import questions_classifiers
import pandas as pd

logger = logging.getLogger(__name__)


def pre_process_raw_data(df):
    with VerboseTimer("Pre processing"):
        df['image_name'] = df['image_name'].apply(lambda q: q if q.lower().endswith('.jpg') else q + '.jpg')
        paths = df['path']

        dirs = {os.path.split(c)[0] for c in paths}
        files_by_folder = {folder: os.listdir(folder) for folder in dirs}
        existing_files = [os.path.normpath(os.path.join(folder, fn))
                          for folder, fn_arr in files_by_folder.items() for fn in fn_arr]
        df.path = df.path.apply(lambda path: os.path.normpath(path))

        existing_idxs = df['path'].isin(existing_files)
        assert existing_idxs.all()
        # df = df.loc[df['path'].isin(existing_idxs)]


        # Getting text features. This is the heavy task...
        df = df.reset_index(drop=True)
        def process_text(txt):
            exclude = set(string.punctuation)
            no_punctuation = ''.join(ch.lower() if ch not in exclude else ' ' for ch in txt)
            no_single_chars = ' '.join(w for w in no_punctuation.split() if len(w) > 1)
            no_multi_space = ' '.join(no_single_chars.split())
            # no_stop_words = ' '.join([w for w in no_multi_space.split() if w not in english_stopwords])
            return no_multi_space


        logger.info('Answer: removing stop words and tokenizing')

        for col in ['processed_answer', 'answer']:
            if col not in df.columns:  # e.g. in test set...
                df[col] = ''

        df.answer.fillna('', inplace=True)
        df.question.fillna('', inplace=True)

        with VerboseTimer("Answer Tokenizing"):
            df['processed_answer'] = df['answer'].apply(process_text)

        logger.info('Question: removing stop words and tokenizing')
        with VerboseTimer("Question Tokenizing"):
            df['processed_question'] = df['question'].apply(process_text)

        ddata = dd.from_pandas(df, npartitions=8)

        def get_string_features(s, *a, **kw):
            features = get_text_features(s)
            return features

        paralelized_get_features = partial(_apply_heavy_function, dask_df=ddata, apply_func=get_string_features)

        logger.info('Getting answers embedding')
        with VerboseTimer("Answer Embedding"):
            df['answer_embedding'] = paralelized_get_features(column='processed_answer')

        logger.info('Getting questions embedding')
        with VerboseTimer("Question Embedding"):
            df['question_embedding'] = paralelized_get_features(column='processed_question')

    __add_category_prediction(df)

    __add_augmented_categories(df)

    logger.debug('Done')
    return df


def __add_augmented_categories(df):
    import re
    from common.functions import get_features
    from classes.vqa_model_predictor import VqaModelPredictor
    from data_access.model_folder import ModelFolder

    abnormality_rows = df.question_category == 'Abnormality'
    yes_no_abnormality_rows = abnormality_rows & \
                              df.question.apply(lambda s: s.split()[0].lower() in ['does', 'is', 'are'])
    df.loc[yes_no_abnormality_rows, 'question_category'] = 'Abnormality_yes_no'

    organ_system_folder = ModelFolder(folder="C:\\Users\\Public\\Documents\\Data\\2019\\models\\20190329_0440_18")
    organ_model = organ_system_folder.load_model()
    abnormality_rows = df.question_category == 'Abnormality'
    a = []
    df_organs = df[df.question_category == 'Organ']
    for i, row in df[abnormality_rows].iterrows():
        image_name = row.image_name
        dd = df_organs[df_organs .image_name == image_name]
        if len(dd):
            organ = dd.iloc[0].answer
            clean_organ = re.sub(r'[^0-9a-zA-Z]+', '_', organ)
            new_category = f'Abnormality_{clean_organ}'
            df.loc[i,'question_category'] = new_category
    abnormality_rows = df.question_category == 'Abnormality'

    df_no_data = df[abnormality_rows]
    with VerboseTimer("Abnormality category prediction"):
        df_preds = VqaModelPredictor._predict_keras(df_no_data,organ_model,organ_system_folder.prediction_vector,0.001)

    df.loc[abnormality_rows,'question_category'] = df_preds.prediction.apply(
        lambda organ:f"Abnormality_{re.sub(r'[^0-9a-zA-Z]+', '_', organ)}")













def __add_category_prediction(df):
    df_with_category = df[~pd.isnull(df.question_category)]
    category_by_question = {row.processed_question: row.question_category
                            for i, row in df_with_category.iterrows()}
    df.loc[:, 'question_category'] = df.processed_question.apply(lambda pq: category_by_question.get(pq))
    df_no_category = df[pd.isnull(df.question_category)]

    if len(df_no_category) > 0:
        questions_to_predict = OrderedDict({i: row.processed_question for i, row in df_no_category.iterrows()})

        idxs_predict = list(questions_to_predict.keys())
        x = np.array([np.array(xi) for xi in df.iloc[idxs_predict].question_embedding])
        predictions = {}
        with VerboseTimer("Predicting question category"):
            for category, classifier_location in questions_classifiers.items():
                if not classifier_location:
                    continue
                with VerboseTimer(f"Predicting for '{category}'"):
                    classifier = File.load_pickle(classifier_location)
                    prediction_result = classifier.predict_proba(x)
                    curr_predictions = {idx: np.argmax(probs) for idx, probs in zip(idxs_predict, prediction_result)}
                    probabilities = {idx: probs[prediction] for (idx, prediction), probs in
                                     zip(curr_predictions.items(), prediction_result)}

                    for idx in idxs_predict:
                        curr_pred = curr_predictions[idx]
                        if curr_pred != 1:
                            continue
                        highest_prob = predictions.get(idx, -1)
                        curr_prob = probabilities[idx]
                        if curr_prob > highest_prob:
                            predictions[idx] = category
        predictions_by_question = {row.processed_question: predictions[i] for i, row in df.iloc[idxs_predict].iterrows()}
        df.loc[idxs_predict, 'question_category'] = df.iloc[idxs_predict].processed_question.apply(
            lambda pq: predictions_by_question[pq])


def _apply_heavy_function(dask_df, apply_func, column, scheduler='processes'):
    res = dask_df.map_partitions(lambda df: df[column].apply(apply_func)).compute(scheduler=scheduler)
    return res


def get_text_features(txt):
    """ For a given txt, a unicode string, returns the time series vector
    with each word (token) transformed into a 300 dimension representation
    calculated using Glove Vector """
    # print(txt)

    text_features = np.zeros((1, input_length, embedding_dim), dtype=float)

    no_data = txt == '' or txt is None or (isinstance(txt,float) and math.isnan(txt))
    if no_data :
        pass
    elif not isinstance(txt, str):
        raise Exception(f'Got an unexpected type for text features: {type(txt).__name__}\n (value was {str(txt)[:20]})')
    else:
        try:

            nlp = get_nlp()
            tokens = nlp(txt)
            num_tokens_to_take = min([input_length, len(tokens)])
            trimmed_tokens = tokens[:num_tokens_to_take]

            for j, token in enumerate(trimmed_tokens):
                # print(len(token.vector))
                text_features[0, j, :] = token.vector
            # Bringing to shape of (1, input_length * embedding_dim)

        except Exception as ex:
            print(f'Failed to get embedding for {txt}:\n{ex}')
            raise

    text_features = text_features.reshape(input_length * embedding_dim)
    return text_features


def normalize_data_strucrture(df, group, image_folder):
    # assert group in ['train', 'validation']
    cols = ['image_name', 'question', 'answer']
    df_c = df[cols].copy()
    df_c['group'] = group
    df_c['path'] = ''

    if len(df_c) == 0:
        return df_c

    def get_image_path(image_name):
        return os.path.join(image_folder, image_name + '.jpg')

    df_c['path'] = df_c.apply(lambda x: get_image_path(x['image_name']),
                              axis=1)  # x: get_image_path(x['group'],x['image_name'])

    return df_c


def main():
    from common.settings import data_access
    df = data_access.load_processed_data()
    __add_augmented_categories(df)


if __name__ == '__main__':
    main()
