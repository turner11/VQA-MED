import os
import tempfile
import itertools
import pandas as pd
from io import StringIO

import pytest

from pre_processing.prepare_data import pre_process_raw_data
from common.functions import generate_image_augmentations
from pre_processing.data_enrichment import enrich_data
from pre_processing.data_cleaning import clean_data
from common.os_utils import File
from pre_processing.known_find_and_replace_items import find_and_replace_collection
from common.settings import set_nlp_vector
from tests.conftest import image_folder
import logging
logger = logging.getLogger(__name__)

normalized_csv = \
    '''
,image_name,question,answer,group,question_category,path
0,test_image,question 0 what does abcts showed on sitting?,answer magnetic resonance imaging 0,test,Abnormality,C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\tests\\test_images\\test_image.jpg
1,test_image,question 1?,answer magnetic resonance angiography 1,test,Abnormality,C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\tests\\test_images\\test_image.jpg
2,test_image,question 2?,answer ct 2,test,Abnormality,C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\tests\\test_images\\test_image.jpg
3,test_image,question 3?,answer ct scan 3,test,Abnormality,C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\tests\\test_images\\test_image.jpg
4,test_image,question 4?,answer mri scan 4,test,Abnormality,C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\tests\\test_images\\test_image.jpg
5,test_image,question 5?,answer reveal 5,test,Abnormality,C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\tests\\test_images\\test_image.jpg
6,test_image,question 6?,answer 6,test,Abnormality,C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\tests\\test_images\\test_image.jpg
7,test_image,question 7?,answer 7,test,Abnormality,C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\tests\\test_images\\test_image.jpg
8,test_image,question 8?,answer 8,test,Modality,C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\tests\\test_images\\test_image.jpg
9,test_image,question 9?,answer 9,test,Modality,C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\tests\\test_images\\test_image.jpg
10,test_image,question 10?,answer 10,test,Modality,C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\tests\\test_images\\test_image.jpg
11,test_image,question 11?,answer 11,test,Modality,C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\tests\\test_images\\test_image.jpg
12,test_image,question 12?,answer 12,test,Modality,C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\tests\\test_images\\test_image.jpg
13,test_image,question 13?,answer 13,test,Modality,C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\tests\\test_images\\test_image.jpg
14,test_image,question 14?,answer 14,test,Modality,C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\tests\\test_images\\test_image.jpg
15,test_image,question 15?,answer 15,test,Modality,C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\tests\\test_images\\test_image.jpg
16,test_image,question 16?,answer 16,test,Plain,C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\tests\\test_images\\test_image.jpg
17,test_image,question 17?,answer 17,test,Plain,C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\tests\\test_images\\test_image.jpg
18,test_image,question 18?,answer 18,test,Plain,C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\tests\\test_images\\test_image.jpg
19,test_image,question 19?,answer 19,test,Plain,C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\tests\\test_images\\test_image.jpg
20,test_image,question 20?,answer 20,test,Plain,C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\tests\\test_images\\test_image.jpg
21,test_image,question 21?,answer 21,test,Plain,C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\tests\\test_images\\test_image.jpg
22,test_image,question 22?,answer 22,test,Plain,C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\tests\\test_images\\test_image.jpg
23,test_image,question 23?,answer 23,test,Plain,C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\tests\\test_images\\test_image.jpg
24,test_image,question 24?,answer 24,test,Organ,C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\tests\\test_images\\test_image.jpg
25,test_image,question 25?,answer 25,test,Organ,C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\tests\\test_images\\test_image.jpg
26,test_image,question 26?,answer 26,test,Organ,C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\tests\\test_images\\test_image.jpg
27,test_image,question 27?,answer 27,test,Organ,C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\tests\\test_images\\test_image.jpg
28,test_image,question 28?,answer 28,test,Organ,C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\tests\\test_images\\test_image.jpg
29,test_image,question 29?,answer 29,test,Organ,C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\tests\\test_images\\test_image.jpg
30,test_image,question 29?,,test,Organ,C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\tests\\test_images\\test_image.jpg

'''

_df_processed = None

@pytest.fixture
def normalized_data():

    # Note: it is normalized in the sense it was passed by normalize_data_strucrture function
    stream = StringIO(normalized_csv)
    normalized_data = pd.read_csv(stream)
    normalized_data.fillna('', inplace=True)

    return normalized_data


@pytest.fixture
def df_processed(normalized_data):
    global _df_processed
    if _df_processed is None:
        set_nlp_vector(-1)  # smallest one...
        normalized_data['processed_answer'] = normalized_data.answer
        normalized_data['processed_question'] = normalized_data.question
        df = pre_process_raw_data(normalized_data)
        _df_processed = df
    return _df_processed



def test_proccessed_length(normalized_data,df_processed):
    assert len(normalized_data) == len(df_processed), 'processed data was not in same length as normalized data'

def test_new_columns_added(df_processed):
    new_columns = ['answer_embedding', 'question_embedding']
    has_new_columns = all(c in df_processed for c in new_columns)
    assert has_new_columns, f'Did not have all columns of pre processing ({new_columns})'

# @pytest.mark.parametrize("expected_length", [])
def test_has_embedding(df_processed):
    max_embedding_val_per_row = df_processed.question_embedding.apply(lambda embedding: max(embedding))
    all_have_values = all(v > 0 for v in max_embedding_val_per_row)
    assert all_have_values , 'Not all embedding had values'


def test_data_cleaning(normalized_data):
    banned_words = (tpl.orig for tpl in find_and_replace_collection)

    def _extract_question_and_answers(df):
        qs = df.question.values
        ans = df.answer.values
        all_strs = qs + ans
        words = [s.split() for s in all_strs]
        distinct = set(itertools.chain.from_iterable(words))
        return distinct

    strings = _extract_question_and_answers(normalized_data)
    banned_in_norm = [w for w in banned_words if w in strings]
    assert len(banned_in_norm) > 0, 'No use in this test if banned words are not in original text'

    clean_df = clean_data(normalized_data)
    clean_strings = _extract_question_and_answers(clean_df)
    banned_in_clean = [w for w in banned_words if w in clean_strings]
    assert len(banned_in_clean) == 0, 'Got banned word in clean data'


def test_data_enrichment():
    csv_txt = \
        '''
image_name,question,answer,processed_question,processed_answer
image1,what shows in mr?,a dolphin, what shows in mr?,a dolphin
image2,pick any word,mr, pick any word,mr
image3,ct is the thing ?,or is it?, ct is the thing ?,or is it?
image4,pick any 2 letters ?,ct, pick any 2 letters ?,ct
image4,and now 2 more ?,mr, and now 2 more ?,mr
image4,what is the difference between hematoma and schwannoma?,with no info what so ever, what is the difference between hematoma and schwannoma?,with no info what so ever
image5,what should i choose ct or mr, just pick one, what should i choose ct or mr, just pick one
image6,a new image,with no available info, a new image,with no available info
synpic20385,what is abnormal in the ct scan?,lung adenocarcinoma and cns metastasis
synpic45585,what organ system is shown in this mri?,skull and contents
synpic38560,what is the plane of this mri?,axial
synpic27731,what imaging method was used?,us - ultrasound
    '''

    df = pd.read_csv(StringIO(csv_txt))
    df_enriched = enrich_data(df)

    assert 'diagnosis' in df_enriched.columns, 'Expected enriched data to contain diagnosis'
    assert 'question_category' in df_enriched.columns, 'Expected enriched data to contain question_category'


    question_category_by_image_name = df_enriched.loc[:, ['image_name', 'question_category']]\
                                                  .set_index('image_name')\
                                                  .to_dict()['question_category']

    category_err_msg = 'Got unexpected question category'
    assert question_category_by_image_name ['synpic20385'] == 'Abnormality', category_err_msg
    assert question_category_by_image_name ['synpic45585'] == 'Organ', category_err_msg
    assert question_category_by_image_name ['synpic38560'] == 'Plane', category_err_msg
    assert question_category_by_image_name ['synpic27731'] == 'Modality', category_err_msg
    str()



def test_data_augmentation():
    augmentation_count = 5
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = os.path.join(temp_dir, 'augmentations')
        output_dir = os.path.normpath(output_dir)
        logger.info(f'Augmentations are at:\n{output_dir }')
        File.validate_dir_exists(output_dir)
        image_name = next(f for f in os.listdir(image_folder) if 'pytest' not in f)
        image_path = os.path.join(image_folder, image_name)
        generate_image_augmentations(image_path, output_dir, augmentation_count=augmentation_count)
        out_put_results = os.listdir(output_dir)
        assert len(out_put_results) == augmentation_count, \
            f'Expected {augmentation_count} augmentations, but got {len(out_put_results) }'


def main():
    nd = normalized_data()
    p = df_processed(nd)
    # df_norm = normalized_data()
    # test_data_cleaning(df_norm )
    # test_embedding()
    # test_data_augmentation()
    test_data_enrichment()
    # test_data_cleaning()
    # test_embedding()
    pass


if __name__ == '__main__':
    main()
