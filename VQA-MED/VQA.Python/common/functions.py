import inspect
import os
import re
import textwrap
from collections import defaultdict


import pandas as pd
import cv2
import numpy as np

from common.settings import input_length, image_size, get_nlp
from common.settings import embedding_dim
from vqa_logger import logger

def get_size(file_name):
    suffixes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    nbytes = os.path.getsize(file_name)
    i = 0
    while nbytes >= 1024 and i < len(suffixes) - 1:
        nbytes /= 1024.
        i += 1
    f = ('%.2f' % nbytes).rstrip('0').rstrip('.')
    return '%s %s' % (f, suffixes[i])


def get_highlited_function_code(foo, remove_comments=False):
    """
    Prints code of a given function with highlighted syntax in a jupyter notebook
    :rtype: IPython.core.display.HTML
    :param foo: the function to print its code
    :param remove_comments: should comments be remove
    """
    from pygments import highlight
    from pygments.lexers import PythonLexer
    from pygments.formatters import HtmlFormatter
    import IPython

    txt = inspect.getsource(foo)
    if remove_comments:
        lines = txt.split('\n')
        lines = [l for l in lines if not l.lstrip().startswith('#')]
        txt = '\n'.join(lines)

    textwrap.dedent(txt)

    formatter = HtmlFormatter()
    ipython_display_object = \
        IPython.display.HTML('<style type="text/css">{}</style>{}'.format(
            formatter.get_style_defs('.highlight'),
            highlight(txt, PythonLexer(), formatter)))

    return ipython_display_object
    # print(txt)


def get_image(image_file_name):
    im = cv2.resize(cv2.imread(image_file_name), image_size)

    # convert the image to RGBA
    #     im = im.transpose((2, 0, 1))
    return im




def generate_image_augmentations(image_path,
                                 output_dir,
                               rotation_range=40,  # Units: degrees
                               width_shift_range=0.2,
                               height_shift_range=0.2,
                               shear_range=0.2,  # Units: degrees
                               zoom_range=0.2,
                               fill_mode='nearest',
                               augmentation_count=20):
    from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


    datagen = ImageDataGenerator(
        rotation_range = rotation_range,
        width_shift_range = width_shift_range,
        height_shift_range = height_shift_range,
        shear_range = shear_range,
        zoom_range = zoom_range,
        horizontal_flip = False,
        vertical_flip=False,
        fill_mode = fill_mode)




    img = load_img(image_path)  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, X, Y)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, X, Y)

    # the .flow() command below generates batches of randomly transformed images
    # and saves the results to the `preview/` directory
    ext = image_path.split('.')[-1]
    i = 0
    for batch in datagen.flow(x, batch_size=1, save_to_dir=output_dir,  save_format=ext):
        i += 1
        if i >= augmentation_count:
            break  # otherwise the generator would loop indefinitely


def get_text_features(txt):
    ''' For a given txt, a unicode string, returns the time series vector
    with each word (token) transformed into a 300 dimension representation
    calculated using Glove Vector '''
    # print(txt)
    try:

        nlp = get_nlp()
        tokens = nlp(txt)
        text_features = np.zeros((1, input_length, embedding_dim))

        num_tokens_to_take = min([input_length, len(tokens)])
        trimmed_tokens = tokens[:num_tokens_to_take]

        for j, token in enumerate(trimmed_tokens):
            # print(len(token.vector))
            text_features[0, j, :] = token.vector
        # Bringing to shape of (1, input_length * embedding_dim)
        ## ATTN - nlp vector:
        text_features = np.reshape(text_features, (1, input_length * embedding_dim))
    except Exception as ex:
        print(f'Failed to get embedding for {txt}')
        raise
    return text_features


def pre_process_raw_data(df):
    df['image_name'] = df['image_name'].apply(lambda q: q if q.lower().endswith('.jpg') else q + '.jpg')
    paths = df['path']

    dirs = {os.path.split(c)[0] for c in paths}
    files_by_folder = {dir: os.listdir(dir) for dir in dirs}
    existing_files = [os.path.join(dir, fn) for dir, fn_arr in files_by_folder.items() for fn in fn_arr]

    df = df.loc[df['path'].isin(existing_files)]

    # df = df[df['path'].isin(existing_files)]
    # df = df.where(df['path'].isin(existing_files))
    logger.debug('Getting answers embedding')
    # Note: Test has noanswer...
    if 'answer' not in df.columns:
        df['answer'] = ''
    df['answer_embedding'] = df['answer'].apply(lambda q: get_text_features(q) if isinstance(q, str) else "")

    logger.debug('Getting questions embedding')
    df['question_embedding'] = df['question'].apply(lambda q: get_text_features(q))

    logger.debug('Getting image features')
    # df['image'] = df['path'].apply(lambda im_path: get_image(im_path))

    logger.debug('Done')
    return df


def normalize_data_strucrture(df, group, image_folder):
    # assert group in ['train', 'validation']
    cols = ['image_name', 'question', 'answer']

    df_c = df[cols].copy()
    df_c['group'] = group

    def get_image_path(image_name):
        return os.path.join(image_folder, image_name + '.jpg')

    df_c['path'] = df_c.apply(lambda x: get_image_path(x['image_name']),
                              axis=1)  # x: get_image_path(x['group'],x['image_name'])

    return df_c


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    from pre_processing.known_find_and_replace_items import find_and_replace_collection

    find_and_replace_data = find_and_replace_collection

    def replace_func(val: str) -> str:
        new_val = val
        if isinstance(new_val, str):
            for tpl in find_and_replace_data:
                pattern = re.compile(tpl.orig, re.IGNORECASE)
                new_val = pattern.sub(tpl.sub, new_val).strip()
        return new_val

    df['question'] = df['question'].apply(replace_func)
    df['answer'] = df['answer'].apply(replace_func)
    return df


def _consolidate_image_devices(df):
    def get_imaging_device(r):
        if r.ct and r.mri:
            res = 'both'
        elif r.ct and not r.mri:
            res = 'ct'
        elif not r.ct and r.mri:
            res = 'mri'
        else:
            res = 'unknown'
        return res

    df['imaging_device'] = df.apply(get_imaging_device, axis=1)

    imaging_device_by_image = defaultdict(lambda: set())
    for i, r in df.iterrows():
        imaging_device_by_image[r.image_name].add(r.imaging_device)

    for name, s in imaging_device_by_image.items():
        if 'both' in s or ('ct' in s and 'mri' in s):
            s.clear()
            s.add('unknown')
        elif 'unknown' in s and ('ct' in s or 'mri' in s):
            is_ct = 'ct' in s
            s.clear()
            s.add('ct' if is_ct else 'mri')

    non_consolidated_vals = [s for s in list(imaging_device_by_image.values()) if len(s) != 1]
    imaging_device_by_image = {k: list(s)[0] for k, s in imaging_device_by_image.items()}
    assert len(
        non_consolidated_vals) == 0, f'got {len(non_consolidated_vals)} non consolodated image devices. for example:\n{non_consolidated_vals[:5]}'
    df['imaging_device'] = df.apply(lambda r: imaging_device_by_image[r.image_name], axis=1)
    return df


def enrich_data(df: pd.DataFrame) -> pd.DataFrame:
    from pre_processing.known_find_and_replace_items import imaging_devices, diagnosis, locations

    # add_imaging_columns
    _add_columns_by_search(df, indicator_words=imaging_devices, search_columns=['question', 'answer'])
    # add_diagnostics_columns
    _add_columns_by_search(df, indicator_words=diagnosis, search_columns=['question', 'answer'])
    # add_locations_columns
    _add_columns_by_search(df, indicator_words=locations, search_columns=['question', 'answer'])

    _consolidate_image_devices(df)
    for col in imaging_devices:
        del df[col]
    return df


def _add_columns_by_search(df, indicator_words, search_columns):
    from common.utils import has_word
    for word in indicator_words:
        res = None
        for col in search_columns:
            curr_res = df[col].apply(lambda s: has_word(word, s))
            if res is None:
                res = curr_res
            res = res | curr_res
        if any(res):
            df[word] = res
        else:
            logger.warn("found no matching for '{0}'".format(word))


def _concat_row(df: pd.DataFrame, col: str):
    # return np.concatenate(df[col], axis=0)
    return np.concatenate([row for row in df[col]])


def get_features(df: pd.DataFrame):
    image_features = np.asarray([np.array(get_image(im_path)) for im_path in df['path']])
    # np.concatenate(df['question_embedding'], axis=0).shape
    question_features = _concat_row(df, 'question_embedding')
    # question_features = np.concatenate([row for row in df.question_embedding])
    reshaped_q = np.array([a.reshape(a.shape + (1,)) for a in question_features])

    features = ([f for f in [reshaped_q, image_features]])

    return features


def sentences_to_hot_vector(sentences:pd.Series, words_df:pd.DataFrame)->iter:
    from sklearn.preprocessing import MultiLabelBinarizer
    classes = words_df.values
    splatted_answers = [ans.lower().split() for ans in sentences]
    clean_splitted_answers = [[w for w in arr if w in classes] for arr in splatted_answers]

    mlb = MultiLabelBinarizer(classes=classes.reshape(classes.shape[0]), sparse_output=False)
    mlb.fit(classes)

    print(f'Classes: {mlb.classes_}')
    arr_one_hot_vector = mlb.transform(clean_splitted_answers)
    return arr_one_hot_vector

def hot_vector_to_words(hot_vector, words_df):
    max_val = hot_vector.max()
    max_loc = np.argwhere(hot_vector == max_val)
    max_loc = max_loc.reshape(max_loc.shape[0])
    return words_df.iloc[max_loc]





def main():
    pass
    # print_function_code(get_nlp, remove_comments=True)


if __name__ == '__main__':
    main()
