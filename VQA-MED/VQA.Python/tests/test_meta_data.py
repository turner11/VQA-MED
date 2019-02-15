import itertools
import os
import tempfile
import pandas as pd
import pytest
from _pytest.tmpdir import tmpdir_factory, TempPathFactory
import logging

from common.settings import data_access

logger = logging.getLogger(__name__)

from pre_processing.meta_data import create_meta

data_dict = {'processed_question': ['what does MRI show?',
                          'where does axial seCTion MRI abdomen show hypoechoic mass?',
                          'what do arrows denote  noncontrast CT pelvis?',
                          'what was normal?',
                          'what shows evidence a contaed rupture?'],

             'processed_answer': ['tumor at tail pancreas',
                        'distal pancreas',
                        'complex fluid colleCTion with layerg consistent with hematoma',
                        'blood supply to bra',
                        'repeat CT  abdomen'],
             }

_meta_data = None


@pytest.fixture(scope='module')
def meta_data():
    global _meta_data
    if _meta_data is None:
        data = pd.DataFrame(data_dict)
        _meta_data = create_meta(data)
    return _meta_data


@pytest.mark.parametrize('meta_kay',{'answers', 'words'})
def test_meta_keys(meta_data, meta_kay):
    assert meta_kay in meta_data
    curr_data_set = meta_data[meta_kay]
    logger.debug(f'For \'{meta_kay}\' Got a data set with {len(curr_data_set)} records')



def test_answer_count(meta_data):
    data_answers = set(data_dict['processed_answer'])
    data_answers = {ans.lower() for ans in data_answers}

    meta_answers = meta_data['answers'].answer
    meta_answers = {ans.lower() for ans in meta_answers}

    assert meta_answers == data_answers, 'Actual answers differed from expected'


def test_word_count(meta_data):
    answers = data_dict['processed_answer']
    questions = data_dict['processed_question']
    all_sentences = answers + questions
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
    test_imaging_devices(meta_loc)
    test_answer_count(meta_loc)
    test_word_count(meta_loc)
