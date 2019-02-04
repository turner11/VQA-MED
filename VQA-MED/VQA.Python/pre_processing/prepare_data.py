import math
import os
import logging
import numpy as np
from functools import partial
import dask.dataframe as dd
from common.os_utils import File
from common.utils import VerboseTimer
from common.settings import input_length, get_nlp
from common.settings import embedding_dim

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
        df = df.loc[df['path'].isin(existing_files)]

        # Getting text features. This is the heavy task...
        df = df.reset_index()
        ddata = dd.from_pandas(df, npartitions=8)

        def get_string_fetures(s, *a, **kw):
            features = get_text_features(s)
            return features

        paralelized_get_features = partial(_apply_heavy_function, dask_df=ddata, apply_func=get_string_fetures)
        logger.info('Getting answers embedding')
        if 'answer' not in df.columns:  # e.g. in test set...
            df['answer'] = ''

        with VerboseTimer("Answer Embedding"):
            df['answer_embedding'] = paralelized_get_features(column='answer')

        logger.info('Getting questions embedding')
        with VerboseTimer("Question Embedding"):
            df['question_embedding'] = paralelized_get_features(column='question')

    df.answer.fillna('', inplace=True)
    df.question.fillna('', inplace=True)
    logger.debug('Done')
    return df


def _apply_heavy_function(dask_df, apply_func, column, scheduler='processes'):
    res = dask_df.map_partitions(lambda df: df[column].apply(apply_func)).compute(scheduler=scheduler)
    return res


def get_text_features(txt):
    """ For a given txt, a unicode string, returns the time series vector
    with each word (token) transformed into a 300 dimension representation
    calculated using Glove Vector """
    # print(txt)

    text_features = np.zeros((1, input_length, embedding_dim),dtype=float)

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
