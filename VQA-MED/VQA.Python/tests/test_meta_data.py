import itertools
import os
import tempfile

import pandas as pd
import pytest
from _pytest.tmpdir import tmpdir_factory, TempPathFactory

from pre_processing.meta_data import create_meta

data_dict = {'question': ['what does MRI show?',
                          'where does axial seCTion MRI abdomen show hypoechoic mass?',
                          'what do arrows denote  noncontrast CT pelvis?',
                          'what was normal?',
                          'what shows evidence a contaed rupture?'],

             'answer': ['tumor at tail pancreas',
                        'distal pancreas',
                        'complex fluid colleCTion with layerg consistent with hematoma',
                        'blood supply to bra',
                        'repeat CT  abdomen'],

             'imaging_device': ['mri', 'mri', 'ct', 'ct', 'unknown']}

_meta_location = None


@pytest.fixture(scope='module')
def meta_location(tmpdir_factory):
    global _meta_location
    if _meta_location is None:
        data = pd.DataFrame(data_dict)
        temp_dir = tmpdir_factory.mktemp('meta_tests')
        out_put_file = os.path.join(temp_dir, 'meta_data.hdf')
        create_meta(data, out_put_file)
        _meta_location = out_put_file

    return _meta_location






def test_meta_keys(meta_location):
    expected_data_sets = {'answers', 'imaging_devices', 'words'}
    with pd.HDFStore(meta_location) as store:
        for ds_key in expected_data_sets:
            curr_data_set = store[ds_key]
            print(f'For \'{ds_key}\' Got a data set with {len(curr_data_set)} records')


def test_imaging_devices(meta_location):
    expected_imaging_devices = {'ct', 'mri'}
    with pd.HDFStore(meta_location) as store:
        df_imaging_devices = store['imaging_devices']
        imaging_devices = df_imaging_devices.imaging_device.values
    assert set(imaging_devices) == expected_imaging_devices, 'Actual imaging devices differed from expected'


def test_answer_count(meta_location):
    expected_answers = set(data_dict['answer'])
    expected_answers = {ans.lower() for ans in expected_answers}
    with pd.HDFStore(meta_location) as store:
        df_actual_answers = store['answers']
        actual_answers = df_actual_answers.answer.apply(lambda ans: ans.lower())
    assert set(actual_answers) == expected_answers, 'Actual answers differed from expected'


def test_word_count(meta_location):
    answers = data_dict['answer']
    questions = data_dict['question']
    all_sentences = answers + questions
    potential_words = {w for w in itertools.chain.from_iterable([s.split() for s in all_sentences])}
    with pd.HDFStore(meta_location) as store:
        df_actual_words = store['words']
        actual_answers = df_actual_words.word.apply(lambda w: w.lower())
    assert len(actual_answers) < len(potential_words), 'Words count was larger than expected'


if __name__ == '__main__':
    rand = next(tempfile._get_candidate_names())
    base_temp = os.path.join(tempfile.gettempdir(), rand)
    def trace(*args, **kwargs):
        print(args)
    temp_factory = TempPathFactory(base_temp , trace=trace)
    meta_loc = meta_location(temp_factory)
    test_meta_keys(meta_loc)
    test_imaging_devices(meta_loc)
    test_answer_count(meta_loc)
    test_word_count(meta_loc)
