import os
from collections import namedtuple
from functools import partial

from keras.utils import to_categorical
from keras.preprocessing import image

from parsers.VQA18 import Vqa18Base
from vqa_flow.image_utils import image_to_pre_process_input
from vqa_flow.vqa_predictor import VqaPredictorFactory
import numpy as np

ImageInfo = namedtuple("ImageInfo", ['image_features', 'embedding_features','classification'])


class EnrichedData(object):
    @property
    def classes(self):
        if self.__classes is None:
            ix_to_ans = self._meta['ix_to_ans']
            classes = [ix_to_ans[i] for i in self.classes_indices]
            self.__classes = classes
        return self.__classes

    @property
    def classes_indices(self):
        if self.__classes_indices is None:
            self.__classes_indices = list(self._meta['ix_to_ans'].keys())
        return self.__classes_indices

    @property
    def class_count(self):
        return len(self.classes)



    @property
    def categorial_labels(self):
        if self.__categorial_labels is None:
            self.__categorial_labels = to_categorical(self.classes_indices, num_classes=self.class_count)
        return self.__categorial_labels

    def __repr__(self):
        return "EnrichedData(image_infos={0}, , name={1}, meta_fn={2})".format(self.image_infos, self.name, self.meta_fn)

    def __init__(self, image_infos , name, meta_fn):
        """"""
        self.image_infos = image_infos
        self.name = name
        self.meta_fn = meta_fn
        self._meta = VqaPredictorFactory.get_metadata(meta_file_location=meta_fn)
        self.__categorial_labels = None
        self.__classes_indices = None
        self.__classes = None
        # construct the image generator for data augmentation
        # aug =  image.ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
        #                                height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
        #                                horizontal_flip=True, fill_mode="nearest")
        #
        # features = [(f.image_features, f.embedding_features) for f in image_infos ]
        #
        # d = {v:k for k,v in self._meta['ix_to_ans'].items()}
        # labels = [f.classification for f in image_infos ]
        # categorial_labels = [d[lbl] for lbl in labels]
        #
        # image_features = np.asanyarray([f.image_features for f in image_infos])
        # self.aug_generator = aug.flow(image_features, categorial_labels)
        # self.aug_generator = aug.flow(features, categorial_labels)

    @staticmethod
    def df_to_data(df,image_fodler, image_size,  name, meta_fn):
        features = EnrichedData.df_to_features(df, image_fodler, image_size)
        data = EnrichedData(features, name, meta_fn)
        return data

    @staticmethod
    def df_to_features(df, image_fodler, image_size):
        relevant_images = df[Vqa18Base.COL_IMAGE_NAME]

        existing_files = [os.path.join(image_fodler, fn)  for fn in os.listdir(image_fodler)]

        images_files = [os.path.join(image_fodler, fn) + ".jpg" for fn in relevant_images]
        images_files = [fn for fn in images_files if fn in existing_files]

        import multiprocessing as mult_proc
        cpu_count = mult_proc.cpu_count()-1
        with mult_proc.Pool(processes=cpu_count) as pool:
            args = [(fn, image_size) for fn in images_files]
            proc_func = partial(image_to_pre_process_input, image_size=image_size)
            features_list = pool.map(proc_func, images_files)

        # features_list = [image_to_pre_process_input(fn, image_size) for fn in images_files]
        image_features = np.asanyarray(features_list)

        answers = np.asanyarray(df[Vqa18Base.COL_ANSWER])

        embedding_features = df[Vqa18Base.COL_QUESTION]

        image_infos = [ImageInfo(f_im, f_emb, ans) for f_im, f_emb, ans in zip(image_features, embedding_features, answers)]
        return image_infos