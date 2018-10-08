import random
from functools import lru_cache

import numpy as np
import pandas as pd
import keras

from common.functions import get_features, sentences_to_hot_vector, get_image
from common.os_utils import File
from common.utils import VerboseTimer


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'


    def __init__(self,
                 vqa_specs_location,
                 batch_size: int = 32,
                 n_channels: object = 1,
                 shuffle: object = True) -> None:
        'Initialization'
        self.vqa_specs = File.load_pickle(vqa_specs_location)
        self.meta_data_location = self.vqa_specs.meta_data_location
        self.df_meta_words = pd.read_hdf(self.meta_data_location, 'words')
        n_classes = len(self.df_meta_words.word.values)

        with pd.HDFStore(self.vqa_specs.data_location) as store:
            index_data_frame = store['index']

        if shuffle:
            groups = [df for _, df in index_data_frame.groupby('augmentation_key')]
            groups = [group.sample(frac=1) for group in groups]
            # groups = [group.sample(frac=1).reset_index(drop=True) for group in groups]
            index_data_frame = pd.concat(groups)


        self.index_data_frame = index_data_frame
        self.batch_size = batch_size
        self.n_channels = n_channels



        self.n_classes = n_classes

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.index_data_frame) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        try:
            indexes = self.indexes[index * self.batch_size :(index+1) *self.batch_size]

            # Find list of IDs


            temp_index = pd.DataFrame([self.index_data_frame.iloc[k] for k in indexes])

            stores = temp_index.store_path.drop_duplicates().values
            assert len(stores) == 1, "Did not implement multi stores..."
            store_location = stores[0]

            augmentation_keys = temp_index.store_key.drop_duplicates().values
            dfs = []
            for k in augmentation_keys:
                df_aug = self._get_df_by_key(k, store_location)
                dfs.append(df_aug)

            df = dfs[0] if len(dfs) == 1 else pd.concat(dfs)
            df_filtered = df[df.idx.isin(indexes)].copy()
            df_filtered['image'] = df_filtered.path.apply(lambda path: get_image(path))


            # safe_index = [i for i in temp_index.index if i in df.index]
            # df_filtered = df.loc[safe_index]
            print(f'Len: {len(df_filtered)}')
            assert len(df_filtered) <= self.batch_size
            if len(df_filtered) == 0:
                File.dump_pickle(temp_index.index, "D:\\Users\\avitu\\Downloads\\tempIndex.pkl")
                File.dump_pickle(df.index, "D:\\Users\\avitu\\Downloads\\df.pkl")
                str(1)
            # Generate data
            X, y = self.__data_generation(df_filtered)
        except Exception as ex:
            str(ex)
            raise
        return X, y

    @lru_cache(2)
    def _get_df_by_key(self, k, store_location):
        with pd.HDFStore(store_location) as store:
            df = store[k]
        return df

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.index_data_frame))

    def __data_generation(self, df: pd.DataFrame) -> (iter, iter):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        # X = np.empty((self.batch_size, *self.dim, self.n_channels))
        # y = np.empty((self.batch_size), dtype=int)

        try:
            with VerboseTimer('Getting train features'):
                features = get_features(df)
            with VerboseTimer('Getting train labels'):
                labels = sentences_to_hot_vector(df.answer, words_df=self.df_meta_words.word)

        except Exception as ex:
            print(f'Failed to get features:\n:{ex}')
            raise


        X = features
        y = labels
        return X,y
        # return X, keras.utils.to_categorical(y, num_classes=self.n_classes)