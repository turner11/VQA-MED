import pandas as pd
import logging
from pandas import HDFStore
from nltk.corpus import stopwords
import itertools

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


        logger.debug(f'Data length: {len(df_data)}')

    elif isinstance(df_data, dict):
        df_data = pd.DataFrame(dict)

    if not isinstance(df_data, pd.DataFrame):
        raise TypeError(f'Could not load data for argument "{df_arg}"')

    df_data = df_data[df_data.group.isin(['train', 'validation'])]
    required_columns = {'processed_question', 'processed_answer', 'question_category'}
    existing_columns = set(df_data.columns)
    missing_columns = required_columns - existing_columns
    assert len(missing_columns) == 0, f'Some columns that are mandatory for metadata where missing:\n{missing_columns}'

    return df_data


def create_meta(df):
    df = _get_data_frame_from_arg(df)
    logger.debug(f"Data frame had {len(df)} rows")


    ## Answers
    ans_columns = ['processed_answer', 'question_category']

    dd = df[ans_columns].groupby('processed_answer').agg(lambda x: tuple(x)).applymap(set).applymap(' '.join).reset_index()
    dd = dd[dd.processed_answer.str.strip().str.len() > 0]
    dd['processed_answer'] = dd.processed_answer.str.strip()
    df_ans = dd.drop_duplicates(subset='processed_answer')

    ## Words
    # Splitting answers
    df_words = pd.DataFrame(df_ans.processed_answer.str.split(' ').tolist(), index=df_ans.question_category).stack()
    df_words = df_words.reset_index()[[0, 'question_category']]  # answer is currently labeled 0
    df_words.columns = ['word', 'question_category']  # renaming
    df_words['word'] = df_words.word.str.replace('[^a-zA-Z]', '').str.strip()

    english_stopwords = set(stopwords.words('english'))
    stops = (english_stopwords - {'no', 'yes'}).union({'th'})

    df_words = df_words[~df_words.word.isin(stops)]

    # Grouping words
    df_unique_words = df_words.groupby('word').agg(lambda x: tuple(x)).applymap(set).applymap(' '.join).reset_index()
    df_unique_words = df_unique_words[df_unique_words.word.str.len() > 0]


    ret_words = df_unique_words.reset_index(drop=True)
    ret_answers = df_ans.reset_index(drop=True)
    df_dict = {'words': ret_words, 'answers': ret_answers }
    return df_dict

def main():
    print("----- Creating meta -----")
    from common.settings import data_access
    df_data = data_access.load_processed_data(
    columns=['path', 'question', 'answer', 'processed_question', 'processed_answer', 'group', 'question_category'])
    df_data = df_data[df_data.group.isin(['train', 'validation'])]
    meta_data_dict = create_meta(df_data)
    data_access.save_meta(meta_data_dict)
   
   
if __name__ == '__main__':
    main()
    