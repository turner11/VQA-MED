import os
import inspect
import textwrap
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path

import pandas as pd
import numpy as np
import cv2
import tqdm

from common.settings import image_size

import logging

logger = logging.getLogger(__name__)


def get_size(file_name):
    suffixes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    nbytes = os.path.getsize(file_name)
    i = 0
    while nbytes >= 1024 and i < len(suffixes) - 1:
        nbytes /= 1024.
        i += 1
    f = ('%.2f' % nbytes).rstrip('0').rstrip('.')
    return '%s %s' % (f, suffixes[i])


def get_highlighted_function_code(foo, remove_comments=False):
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
                                 rotation_range=25,  # Units: degrees
                                 width_shift_range=0.15,
                                 height_shift_range=0.15,
                                 shear_range=0.,  # Units: degrees
                                 zoom_range=0.15,
                                 fill_mode='nearest',
                                 augmentation_count=20):
    from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img  # ,array_to_img

    datagen = ImageDataGenerator(
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        shear_range=shear_range,
        zoom_range=zoom_range,
        horizontal_flip=False,
        vertical_flip=False,
        fill_mode=fill_mode)

    img = load_img(image_path)  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, X, Y)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, X, Y)

    # the .flow() command below generates batches of randomly transformed images
    # and saves the results to the `preview/` directory
    ext = image_path.split('.')[-1]
    i = 0
    for _ in datagen.flow(x, batch_size=1, save_to_dir=output_dir, save_format=ext):
        i += 1
        if i >= augmentation_count:
            break  # otherwise the generator would loop indefinitely


def normalize_data_structure(df: pd.DataFrame, group: str, image_folder: str) -> pd.DataFrame:
    # assert group in ['train', 'validation']
    cols = ['image_name', 'question', 'answer']
    df_c = df[cols].copy()
    df_c['group'] = group
    df_c['path'] = ''

    if len(df_c) == 0:
        return df_c

    def get_image_path(image_name):
        return str(Path(image_folder)/(image_name + '.jpg'))

    df_c['path'] = df_c.apply(lambda x: get_image_path(x['image_name']),
                              axis=1)  # x: get_image_path(x['group'],x['image_name'])

    return df_c


def get_features(df: pd.DataFrame):
    series_reshaped = df.question_embedding.apply(lambda embedding: embedding.reshape((embedding.shape[0], 1)))
    shape_sample = series_reshaped.values[0].shape
    set_shape = (len(series_reshaped.values), shape_sample[0],1)

    question_features = np.reshape(list(series_reshaped.values), set_shape)

    pool = ThreadPool(processes=7)
    unique_image_paths = df.path.drop_duplicates()
    logger.debug('Getting image features')
    worker_generator = pool.imap(lambda im_path: np.array(get_image(im_path)), unique_image_paths)
    images = list(tqdm.tqdm(worker_generator , total=len(unique_image_paths)))

    # images = pool.map(lambda im_path: np.array(get_image(im_path)), unique_image_paths)
    image_by_path = {im_path:img for im_path, img in zip(unique_image_paths, images)}

    # image_by_path = {im_path:np.array(get_image(im_path)) for im_path in df.path.drop_duplicates()}
    image_features = np.asarray([image_by_path[im_path] for im_path in df['path']])

    # image_features = np.asarray([np.array(get_image(im_path)) for im_path in df['path']])
    features = [question_features, image_features]

    return features

def sentences_to_hot_vector(labels: iter, classes: iter) -> iter:
    from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer
    labels_arr = list(labels)
    classes_arr = np.asarray(classes)

    number_of_items_in_class = max([len(c.split()) for c in classes_arr])
    # If we are using words as labels - allow multi labels, otherwise only 1
    # e.g. we can have a label of both 'ct' and 'skull' but not 'double aortic arch' and 'radial head fracture'
    if number_of_items_in_class == 1:
        # logger.debug('Using multi label')
        splitted_labels = [ans.lower().split() for ans in labels_arr]
        # clean_splitted_labels = [[w for w in arr if w in labels] for arr in splitted_labels]
        clean_splitted_labels = [[w for w in arr if w in classes_arr] for arr in splitted_labels]

    else:
        # logger.debug('Using single label')
        clean_splitted_labels = [[lbl] for lbl in labels_arr]

    mlb = MultiLabelBinarizer(classes=classes_arr.reshape(classes_arr.shape[0]), sparse_output=False)
    mlb.fit(labels_arr)
    arr_hot_vector = mlb.transform(clean_splitted_labels)

    # logger.debug(f'Classes: {labels_arr}')
    return arr_hot_vector


def hot_vector_to_words(hot_vector, classes_df):
    max_val = hot_vector.max()
    max_loc = np.argwhere(hot_vector == max_val)
    max_loc = max_loc.reshape(max_loc.shape[0])
    return classes_df.iloc[max_loc]


def main():
    pass
    # print_function_code(get_nlp, remove_comments=True)


if __name__ == '__main__':
    main()
