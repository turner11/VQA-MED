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

    required_columns = {'processed_question', 'processed_answer'}
    existing_columns = set(df_data.columns)
    missing_columns = required_columns - existing_columns
    assert len(missing_columns) == 0, f'Some columns that are mandatory for metadata where missing:\n{missing_columns}'

    return df_data


def create_meta(df):
    df = _get_data_frame_from_arg(df)

    logger.debug(f"Data frame had {len(df)} rows")
    english_stopwords = set(stopwords.words('english'))

    def get_unique_words(col):
        single_string = " ".join(df[col])
        exclude = set(string.punctuation)
        no_punctuation = ''.join(ch.lower() for ch in single_string if ch not in exclude)
        unique_words = set(no_punctuation.split(" ")).difference({'', ' '})
        unique_words = unique_words.difference(english_stopwords)
        logger.debug("column {0} had {1} unique words".format(col, len(unique_words)))
        return unique_words

    cols = ['processed_question', 'processed_answer']
    df_unique_words = set(itertools.chain.from_iterable([get_unique_words(col) for col in cols]))
    unique_answers = set([ans.lower() for ans in df['processed_answer']])

    words = sorted(list(df_unique_words), key=lambda w: (len(w), w))
    words = [w for w in words if
             w in ['ct', 'mri']
             or len(w) >= 3
             and not w[0].isdigit()]

    metadata_dict = {'words': {'word': words}, 'answers': {'answer': list(unique_answers)}}

    df_dict = {k: pd.DataFrame(dictionary, dtype=str) for k, dictionary in metadata_dict.items()}
    return df_dict
