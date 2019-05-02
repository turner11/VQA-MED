import logging
import os
import shutil
from functools import lru_cache

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

from pyarrow.lib import ArrowInvalid, Table
from typing import Union

from common.exceptions import NoDataException, InvalidArgumentException
from common.utils import VerboseTimer

logger = logging.getLogger(__name__)


class DataAccess(object):
    RAW_DATA_FILE_NAME = 'raw_data.h5'
    RAW_DATA_KEY = 'data'

    PROCESSED_DATA_FILE_NAME = 'model_input.parquet'

    def __init__(self, folder):
        """"""
        super().__init__()
        self.folder = Path(str(folder))

        assert self.folder.exists()

    def __repr__(self):
        return f'{self.__class__.__name__}(folder={str(self.folder)})'

    @property
    def raw_data_location(self):
        return self.folder / self.RAW_DATA_FILE_NAME

    @property
    def processed_data_location(self):
        return self.folder / self.PROCESSED_DATA_FILE_NAME

    @property
    def fn_meta(self):
        return self.folder / 'meta_data.h5'

    @property
    def augmentation_location(self):
        return self.folder / 'augmentations.parquet'

    def save_raw_input(self, df: pd.DataFrame) -> str:
        """
        For saving the normalized raw data
        :param df: the raw data data frame
        """
        full_path = str(self.raw_data_location)
        try:
            os.remove(full_path)
        except OSError:
            pass

        with pd.HDFStore(full_path) as store:
            store[self.RAW_DATA_KEY] = df

        return full_path

    def load_raw_input(self) -> pd.DataFrame:
        """
        For loading the normalized raw data
        :return: the raw data data frame
        """
        full_path = str(self.raw_data_location)
        logger.debug(f'Loading data from: {full_path}')

        with VerboseTimer("Loading raw data"):
            with pd.HDFStore(full_path) as store:
                image_name_question = store[self.RAW_DATA_KEY]
            return image_name_question

    def save_processed_data(self, df: pd.DataFrame) -> str:
        full_path = str(self.processed_data_location)
        logger.debug(f"Saving the processed data to:\n{full_path}")
        with VerboseTimer("Saving processed data"):
            self._save_parquet(df, full_path, 'group')
        return full_path

    def load_processed_data(self, group: str = None, columns: list = None) -> pd.DataFrame:

        if group is not None:
            filters = [('group', '==', str(group)), ]
        else:
            filters = None

        df_data = self._load_processed_data(filters=filters, columns=columns)
        return df_data

    def _load_processed_data(self, filters: list = None, columns: list = None) -> pd.DataFrame:
        full_path = str(self.processed_data_location)
        logger.debug(f'loading processed data from:\n{full_path}')
        affective_columns = tuple(columns or {}) if columns is not None else None
        affective_filters = tuple(filters or {}) if filters is not None else None
        df_data = self._load_parquet(full_path, affective_columns, filters=affective_filters)
        return df_data

    def save_augmentation_data(self, df_augmentations):
        path = str(self.augmentation_location)
        logger.debug(f"Saving augmentations:\n{path}")
        with VerboseTimer("Saving augmentations"):
            self._save_parquet(df=df_augmentations, location=path, partition_col='augmentation')
        return path

    def load_augmentation_data(self, columns=None, augmentations=None):
        path = str(self.augmentation_location)
        logger.debug(f"Loading augmentations:\n{path}")

        if isinstance(augmentations, (int, float)):
            augmentations = max(augmentations, 1)

        if augmentations is not None:
            filters = (('augmentation', '<', int(augmentations)), )
        else:
            filters = None

        df_augmentations = self._load_parquet(path, columns=columns, filters=filters)
        return df_augmentations

    @staticmethod
    def _save_parquet(df, location, partition_col):
        path = str(location)
        p_location = Path(path)

        if p_location.exists():
            try:
                shutil.rmtree(path)
            except Exception as ex:
                logger.warning(f'Failed to delete parquet: {ex}')

        # noinspection PyArgumentList
        table: Table = pa.Table.from_pandas(df)
        return pq.write_to_dataset(table,
                                   root_path=path,
                                   partition_cols=[partition_col],
                                   )

    @staticmethod
    @lru_cache(maxsize=5)
    def _load_parquet(path, columns=None, filters=None, convert_to_pandas=True):
        logger.debug(f'loading parquet from:\n{path}')

        data_set = pq.ParquetDataset(path, filters=filters)
        with VerboseTimer("Loading parquet"):
            try:
                prqt = data_set.read(columns=columns)
            except ArrowInvalid as e:
                raise NoDataException(str(e)) from e

        if not convert_to_pandas:
            return prqt

        with VerboseTimer("Converting to pandas"):
            df_data = prqt.to_pandas()
        return df_data  # [:150]

    def save_meta(self, meta_df_dict):
        meta_location = str(self.fn_meta)
        try:
            os.remove(meta_location)
        except OSError:
            pass

        for name, df_curr in meta_df_dict.items():
            df_curr.to_hdf(meta_location, name, format='table')

        with pd.HDFStore(meta_location) as metadata_store:
            logger.debug("Meta number of unique answers: {0}".format(len(metadata_store['answers'])))
            logger.debug("Meta number of unique words: {0}".format(len(metadata_store['words'])))

    def load_meta(self):
        return self.load_meta_from_location(self.fn_meta)

    @classmethod
    def load_meta_from_location(cls, meta_location):
        meta_location = str(meta_location)
        with pd.HDFStore(meta_location) as metadata_store:
            ret = {key: metadata_store[key] for key in ['answers', 'words']}

        return ret


    @classmethod
    def get_prediction_data(cls, meta_data, prediction_data_name, question_category):
        vector = meta_data[prediction_data_name]
        col_question_category = 'question_category'
        non_category_columns = [c for c in vector.columns if c != col_question_category]

        assert len(non_category_columns) == 1, 'Expected to get a single vector for prediction'

        if question_category and col_question_category in vector.columns:
            categories = vector[col_question_category]
            # idxs = categories.str.contains(question_category)  # Categories are space delimited strings
            idxs = categories.str.contains(r'\b{0}\b'.format(question_category), regex=True)  # Categories are space delimited strings
            vector = vector[idxs]

        ret = vector[non_category_columns[0]].drop_duplicates().reset_index(drop=True)
        return ret


class SpecificDataAccess(DataAccess):
    def __init__(self, folder: Union[str, Path], group: str = None, question_category: str = None) -> None:
        super().__init__(folder)
        self.group = group
        self.question_category = question_category

    def load_processed_data(self, group: str = None, columns: list = None) -> pd.DataFrame:
        if group is not None and self.group is not None:
            msg = f'For {self.__class__.__name__}, group cannot be differ from instance group. {group} != {self.group}'
            raise InvalidArgumentException(group, msg)

        affective_group = group or self.group
        df_data = super().load_processed_data(affective_group, columns)

        if self.question_category:
            df_data = df_data[df_data.question_category == self.question_category]
        if len(df_data) == 0:
            logger.warning('Something is fishy. Got an empty data frame.')
        return df_data

    def __repr__(self):
        return f'{self.__class__.__name__}(folder="{str(self.folder)}", group="{self.group}", ' \
            f'question_category="{self.question_category}") '

    @staticmethod
    def factory(data_access: Union[DataAccess, str, Path], group: str = None, question_category: str = None):
        folder = data_access if not isinstance(data_access, DataAccess) else data_access.folder

        if isinstance(data_access, SpecificDataAccess):
            group = group or data_access.group
            question_category = data_access.question_category

        return SpecificDataAccess(folder, group, question_category)
