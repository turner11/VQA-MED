import itertools
import os
import tempfile
import pandas as pd
import pytest
from _pytest.tmpdir import TempPathFactory
import logging
from pre_processing.meta_data import create_meta

logger = logging.getLogger(__name__)

data_dict = {
    'processed_question': {0: 'what is abnormal in the mammograph', 1: 'what type of image modality is seen',
                           2: 'what organ is this image of', 3: 'which plane is this image in',
                           4: 'what is most alarming about this ultrasound', 5: 'what was this image taken with',
                           6: 'which organ system is imaged', 7: 'in what plane is this ultrasound',
                           8: 'what is the primary abnormality in this image', 9: 'how was this image taken',
                           10: 'what part of the body is being imaged', 11: 'what plane is this'},
    'processed_answer': {0: 'secretory calcifications of the breast', 1: 'mr other pulse seq', 2: 'skull and contents',
                         3: 'ap', 4: 'sarcoidosis', 5: 'mra mr angiography venography', 6: 'face sinuses and neck',
                         7: 'transverse', 8: '', 9: '', 10: '', 11: ''},
    'question_category': {0: 'Abnormality', 1: 'Modality', 2: 'Organ', 3: 'Plane', 4: 'Abnormality', 5: 'Modality',
                          6: 'Organ', 7: 'Plane', 8: 'Abnormality', 9: 'Modality', 10: 'Organ', 11: 'Plane'},
    'group': {0: 'validation', 1: 'validation', 2: 'validation', 3: 'validation', 4: 'train', 5: 'train', 6: 'train',
              7: 'train', 8: 'test', 9: 'test', 10: 'test', 11: 'test'}
}

## Create meta_dict:
# import pandas as pd
# df_data = data_access.load_processed_data(columns=['path','question','answer', 'processed_question','processed_answer', 'group','question_category'])
# df_data = df_data[df_data.group.isin(['train','validation', 'test'])]
# g_question = df_data.groupby(['question_category'])
# agg = []
# for i, gdf in g_question:
#     g_group = gdf.groupby(['group'])
#     for ii, gg in g_group:
#         agg.extend(gg.sample(1).index.values)
#
#
# agg_df = df_data.iloc[agg].sort_values(by=['group', 'question_category'], ascending=[False,True]).reset_index(drop=True)
# agg_df = agg_df[['processed_question', 'processed_answer', 'question_category', 'group']]
# agg_df.to_dict()
# print(agg_df.to_dict())
# # agg_df

_meta_data = None


@pytest.fixture(scope='module')
def meta_data():
    global _meta_data
    if _meta_data is None:
        data = pd.DataFrame(data_dict)
        _meta_data = create_meta(data)
    return _meta_data


@pytest.mark.parametrize('meta_kay', {'answers', 'words'})
def test_meta_keys(meta_data, meta_kay):
    assert meta_kay in meta_data
    curr_data_set = meta_data[meta_kay]
    logger.debug(f'For \'{meta_kay}\' Got a data set with {len(curr_data_set)} records')


def test_answer_count(meta_data):
    df_answers = meta_data['answers']
    meta_answers = set(df_answers.processed_answer)
    meta_answers = {ans.lower() for ans in meta_answers}

    data_answers = pd.Series(data_dict['processed_answer'])
    data_answers = {ans.lower() for ans in data_answers if ans}

    diff = (data_answers ^ meta_answers) - {''}
    assert len(diff) == 0, f'Actual answers differed from expected by {len(diff)}'


def test_word_count(meta_data):
    answers = meta_data['answers'].processed_answer
    all_sentences = answers
    potential_words = {w for w in itertools.chain.from_iterable([s.split() for s in all_sentences])}

    df_actual_words = meta_data['words']
    actual_words = df_actual_words.word.apply(lambda w: w.lower())
    assert len(actual_words) < len(potential_words), 'Words count was larger than expected'


if __name__ == '__main__':
    rand = next(tempfile._get_candidate_names())
    base_temp = os.path.join(tempfile.gettempdir(), rand)


    def trace(*args, **kwargs):
        logger.debug(args)

    temp_factory = TempPathFactory(base_temp, trace=trace)
    meta_loc = meta_data(temp_factory)
    test_meta_keys(meta_loc)
    test_answer_count(meta_loc)
    test_word_count(meta_loc)
