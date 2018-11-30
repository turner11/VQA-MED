import os
import tempfile

import pytest
import itertools
import pandas as pd
from io import StringIO

from common.functions import pre_process_raw_data, generate_image_augmentations
from common.functions import enrich_data, clean_data
from common.os_utils import File
from pre_processing.known_find_and_replace_items import find_and_replace_collection
from common.settings import set_nlp_vector
from tests import image_folder

normalized_csv =\
'''
,image_name,question,answer,group,path
0,test_image,question 0?,answer magnetic resonance imaging 0,test,C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\tests\\test_images\\test_image.jpg
1,test_image,question 1?,answer magnetic resonance angiography 1,test,C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\tests\\test_images\\test_image.jpg
2,test_image,question 2?,answer ct 2,test,C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\tests\\test_images\\test_image.jpg
3,test_image,question 3?,answer ct scan 3,test,C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\tests\\test_images\\test_image.jpg
4,test_image,question 4?,answer mri scan 4,test,C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\tests\\test_images\\test_image.jpg
5,test_image,question 5?,answer reveal 5,test,C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\tests\\test_images\\test_image.jpg
6,test_image,question 6?,answer 6,test,C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\tests\\test_images\\test_image.jpg
7,test_image,question 7?,answer 7,test,C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\tests\\test_images\\test_image.jpg
8,test_image,question 8?,answer 8,test,C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\tests\\test_images\\test_image.jpg
9,test_image,question 9?,answer 9,test,C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\tests\\test_images\\test_image.jpg
10,test_image,question 10?,answer 10,test,C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\tests\\test_images\\test_image.jpg
11,test_image,question 11?,answer 11,test,C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\tests\\test_images\\test_image.jpg
12,test_image,question 12?,answer 12,test,C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\tests\\test_images\\test_image.jpg
13,test_image,question 13?,answer 13,test,C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\tests\\test_images\\test_image.jpg
14,test_image,question 14?,answer 14,test,C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\tests\\test_images\\test_image.jpg
15,test_image,question 15?,answer 15,test,C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\tests\\test_images\\test_image.jpg
16,test_image,question 16?,answer 16,test,C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\tests\\test_images\\test_image.jpg
17,test_image,question 17?,answer 17,test,C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\tests\\test_images\\test_image.jpg
18,test_image,question 18?,answer 18,test,C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\tests\\test_images\\test_image.jpg
19,test_image,question 19?,answer 19,test,C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\tests\\test_images\\test_image.jpg
20,test_image,question 20?,answer 20,test,C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\tests\\test_images\\test_image.jpg
21,test_image,question 21?,answer 21,test,C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\tests\\test_images\\test_image.jpg
22,test_image,question 22?,answer 22,test,C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\tests\\test_images\\test_image.jpg
23,test_image,question 23?,answer 23,test,C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\tests\\test_images\\test_image.jpg
24,test_image,question 24?,answer 24,test,C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\tests\\test_images\\test_image.jpg
25,test_image,question 25?,answer 25,test,C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\tests\\test_images\\test_image.jpg
26,test_image,question 26?,answer 26,test,C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\tests\\test_images\\test_image.jpg
27,test_image,question 27?,answer 27,test,C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\tests\\test_images\\test_image.jpg
28,test_image,question 28?,answer 28,test,C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\tests\\test_images\\test_image.jpg
29,test_image,question 29?,answer 29,test,C:\\Users\\avitu\\Documents\\GitHub\\VQA-MED\\VQA-MED\\VQA.Python\\tests\\test_images\\test_image.jpg

'''

def _get_normalized_data_frame():
    # Note: it is normalized in the sense it was passed by normalize_data_strucrture function
    stream = StringIO(normalized_csv)
    normalized_data = pd.read_csv(stream)
    return normalized_data

# @pytest.mark.parametrize("expected_length", [])
def test_embedding():
    set_nlp_vector(-1)  # smallest one...
    normalized_data = _get_normalized_data_frame()

    df_processed = pre_process_raw_data(normalized_data)
    assert len(normalized_data ) == len(df_processed), 'processed data was not in same length as normalized data'

    new_columns = ['answer_embedding', 'question_embedding']
    has_new_columns = all(c in df_processed for c in new_columns)
    assert has_new_columns, f'Did not have all columns of pre processing ({new_columns})'

def test_data_cleaning():
    banned_words = (tpl.orig for tpl in find_and_replace_collection)
    normalized_data = _get_normalized_data_frame()

    def get_question_and_answers(df):
        qs = df.question.values
        ans = df.answer.values
        all_strs = qs + ans
        words = [s.split() for s in all_strs ]
        distinct = set(itertools.chain.from_iterable(words))
        return distinct

    strings = get_question_and_answers(normalized_data)
    banned_in_norm = [w for w in banned_words if w in strings]
    assert len(banned_in_norm) > 0, 'No use in this test if banned words are not in original text'

    clean_df = clean_data(normalized_data)
    clean_strings = get_question_and_answers(clean_df)
    banned_in_clean = [w for w in banned_words if w in clean_strings]
    assert len(banned_in_clean) > 0, 'Got banned word in clean data'

def test_data_enrichment():
    csv_txt=\
    '''
image_name,question,answer
image1,what shows in MRI?,a dolphin
image2,Pick any word,mri
image3,cT is the thing ?,or Is it?
image4,pick any 2 letters ?,ct
image4,another question?,with no info what so ever
image5,What should I choose cT or MrI, Just pick one
image6,A new image,with no available info
    '''

    df = pd.read_csv(StringIO(csv_txt))
    df_enriched = enrich_data(df)

    assert 'imaging_device' in df_enriched.columns, 'Expected enriched data to contain imaging_device'

    imaging_device_by_image_name = df[['image_name','imaging_device' ]].set_index('image_name').to_dict()['imaging_device']

    err_msg = 'Got unexpected imaging devices'
    assert imaging_device_by_image_name['image1'] == 'mri', err_msg
    assert imaging_device_by_image_name['image2'] == 'mri', err_msg
    assert imaging_device_by_image_name['image3'] == 'ct', err_msg
    assert imaging_device_by_image_name['image4'] == 'ct', err_msg
    assert imaging_device_by_image_name['image5'] == 'unknown', err_msg
    assert imaging_device_by_image_name['image6'] == 'unknown', err_msg

def test_data_augmentation():
    AUGMENTATION_COUNT = 5
    temp_dir = tempfile.gettempdir()
    output_dir = os.path.join(temp_dir ,'augmentations')
    output_dir = os.path.normpath(output_dir )
    print(f'Augmentations are at:\n{output_dir }')
    File.validate_dir_exists(output_dir)
    image_name = os.listdir(image_folder)[0]
    image_path = os.path.join(image_folder,image_name)
    generate_image_augmentations(image_path , output_dir,augmentation_count=AUGMENTATION_COUNT)
    out_put_results = os.listdir(output_dir)
    assert len(out_put_results) == AUGMENTATION_COUNT, f'Expected {AUGMENTATION_COUNT} augmentations, but got {len(out_put_results) }'
    



def main():
    test_data_augmentation()
    # test_data_enrichment()
    # test_data_cleaning()
    # test_embedding()
    pass


if __name__ == '__main__':
    main()
