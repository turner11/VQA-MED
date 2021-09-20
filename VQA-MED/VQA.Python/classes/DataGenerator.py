from collections import Iterable
from functools import lru_cache
import numpy as np
import pandas as pd
import keras
from common.exceptions import InvalidArgumentException
from common.functions import get_features, sentences_to_hot_vector
import logging

from data_access.api import DataAccess

logger = logging.getLogger(__name__)


class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, data_access: DataAccess, prediction_vector: iter,
                 batch_size: int = 32,
                 n_channels: int = 1,
                 shuffle: bool = True,
                 augmentations=10) -> None:
        """Initialization"""

        self.shuffle = shuffle
        self.prediction_vector = self.__get_prediction_vector(prediction_vector)

        orig_data = data_access.load_processed_data()

        df_augmentations = data_access.load_augmentation_data(augmentations=augmentations).sort_values(
            'augmentation').reset_index(drop=True)

        data = orig_data.set_index('path')
        augs = df_augmentations.set_index('original_path')
        joined = data.join(augs, how='left').reset_index(drop=True)
        self.data = joined.sort_values(by='augmentation')

        self.batch_size = int(batch_size)
        self.n_channels = n_channels

        self.indexes = np.arange(0)# Will be set in on_epoch_end
        self.on_epoch_end()

    @staticmethod
    def __get_prediction_vector(prediction_vector):
        ret = prediction_vector
        if isinstance(prediction_vector, pd.DataFrame):
            if len(prediction_vector.columns) > 1:
                raise InvalidArgumentException(argument_name='prediction_vector',
                                               argument=prediction_vector,
                                               message='Cannot infer prediction vector')
            ret = prediction_vector[prediction_vector.columns[0]].values
        elif isinstance(prediction_vector, Iterable):
            ret = np.asarray(prediction_vector)
        else:
            raise InvalidArgumentException(argument_name='prediction_vector',
                                           argument=prediction_vector,
                                           message='Cannot infer prediction vector')

        return ret

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        try:
            start = int(index * self.batch_size)
            end = int((index + 1) * self.batch_size)
            indexes = self.indexes[start:end]
            # Find list of IDs
            data = self.data.iloc[indexes]
            # Make sure not to get same question/image from different augmentations at the same pass
            data = data.drop_duplicates(subset=['path', 'question'], keep='first')

            if self.shuffle:
                data = data.sample(frac=1)  # .reset_index(drop=True)

            X, y = self._generate_data(data, self.prediction_vector)
        except Exception as ex:
            logger.exception('Got an error while loading data')
            raise
        return X, y

    def get_full_data(self):
        X, y = self._generate_data(self.data, self.prediction_vector)
        return X, y

    @lru_cache(2)
    def _get_df_by_key(self, k, store_location):
        with pd.HDFStore(store_location) as store:
            df = store[k]
        return df

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.data))

    @staticmethod
    def _generate_data(df: pd.DataFrame, prediction_vector: iter) -> (iter, iter):
        """Generates data containing batch_size samples"""  # X : (n_samples, *dim, n_channels)
        # Initialization
        # X = np.empty((self.batch_size, *self.dim, self.n_channels))
        # y = np.empty((self.batch_size), dtype=int)
        try:
            # with VerboseTimer(f'Getting {item_count} train features'):
            features = get_features(df)
            # with VerboseTimer(f'Getting {item_count} train labels'):
            labels = sentences_to_hot_vector(labels=df.processed_answer, classes=prediction_vector)

        except Exception as ex:
            logger.warning(f'Failed to get features:\n:{ex}')
            raise

        X = features
        y = labels
        return X, y


def main():
    from common.settings import data_access
    meta_dict = data_access.load_meta()
    pred_vec = meta_dict['words']
    d_gen = DataGenerator(data_access=data_access, prediction_vector=pred_vec)
    str()
    res = []
    for i in range(10):
        X, y = d_gen[i]
        res.append((X, y))
    str(res)


if __name__ == '__main__':
    main()
