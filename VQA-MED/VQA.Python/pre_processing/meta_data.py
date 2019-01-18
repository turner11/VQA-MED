import os
import pandas as pd
import logging
from pandas import HDFStore
from nltk.corpus import stopwords
import itertools
import string
from common.utils import VerboseTimer

logger = logging.getLogger(__name__)


def _get_data_frame_from_arg(df_arg):
    df_data = None

    if isinstance(df_arg, pd.DataFrame):
        df_data = df_arg
    elif isinstance(df_arg, str):
        data_location = df_arg
        logger.debug(f'loading from:\n{data_location}')
        with VerboseTimer("Loading Data"):
            with HDFStore(data_location) as store:
                df_data = store['data']

        df_data = df_data[df_data.group.isin(['train', 'validation'])]
        logger.debug(f'Data length: {len(df_data)}')

    elif isinstance(df_data, dict):
        df_data = pd.DataFrame(dict)

    if not isinstance(df_data, pd.DataFrame):
        raise TypeError(f'Could not load data for argument "{df_arg}"')

    requiered_columns = {'question', 'answer', 'imaging_device'}
    existing_columns = set(df_data.columns)
    missing_columns = requiered_columns - existing_columns
    assert len(missing_columns) == 0, f'Some columns that are mandatory for metadata where missing:\n{missing_columns}'

    return df_data


def create_meta(df, hdf_output_location):
    df = _get_data_frame_from_arg(df)

    logger.debug(f"Dataframe had {len(df)} rows")
    english_stopwords = set(stopwords.words('english'))

    def get_unique_words(col):
        single_string = " ".join(df[col])
        exclude = set(string.punctuation)
        s_no_panctuation = ''.join(ch.lower() for ch in single_string if ch not in exclude)
        unique_words = set(s_no_panctuation.split(" ")).difference({'', ' '})
        unique_words = unique_words.difference(english_stopwords)
        logger.debug("column {0} had {1} unique words".format(col, len(unique_words)))
        return unique_words

    cols = ['question', 'answer']
    df_unique_words = set(itertools.chain.from_iterable([get_unique_words(col) for col in cols]))
    unique_answers = set([ans.lower() for ans in df['answer']])

    unknown_devices = {'both', 'unknown'}
    unique_imaging_devices = list(set(df['imaging_device']) - unknown_devices)


    words = sorted(list(df_unique_words), key=lambda w: (len(w), w))
    words = [w for w in words if
             w in ['ct', 'mri']
             or len(w) >= 3
             and not w[0].isdigit()]

    metadata_dict = {}
    metadata_dict['words'] = {'word': words}
    metadata_dict['answers'] = {'answer': list(unique_answers)}
    metadata_dict['imaging_devices'] = {'imaging_device': unique_imaging_devices}

    try:
        os.remove(hdf_output_location)
    except OSError:
        pass

    for name, dictionary in metadata_dict.items():
        df_curr = pd.DataFrame(dictionary, dtype=str)
        df_curr.to_hdf(hdf_output_location, name, format='table')

    with HDFStore(hdf_output_location) as metadata_store:
        logger.debug("Meta number of unique answers: {0}".format(len(metadata_store['answers'])))
        logger.debug("Meta number of unique words: {0}".format(len(metadata_store['words'])))

#         df_ix_to_word = pd.DataFrame.from_dict(metadata['ix_to_word'])
#         light.to_hdf(data_location, 'light', mode='w', data_columns=['image_name', 'imaging_device', 'path'], format='table')
