import logging
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
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
    def vqa_specs_location(self):
        return self.folder / 'vqa_specs.pkl'

    @property
    def fn_meta(self):
        return self.folder / 'meta_data.h5'

    @property
    def augmentation_index(self):
        return self.folder / 'augmentation_index.h5'


    def save_raw_input(self, df: pd.DataFrame) -> None:
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

    def save_processed_data(self, df: pd.DataFrame) -> None:
        full_path = str(self.processed_data_location)
        logger.debug(f"Saving the processed data to: {full_path}")
        with VerboseTimer("Saving processed data"):
            return self.__save_parquet(df, full_path, 'group')

    def load_processed_data(self) -> pd.DataFrame:
        full_path = str(self.processed_data_location)
        print(f'loading from:\n{full_path}')
        with VerboseTimer("Loading Data"):
            prqt = pq.read_table(full_path)

        with VerboseTimer("Converting to pandas"):
            df_data = prqt.to_pandas()
        return df_data

    @staticmethod
    def __save_parquet(df, location, partition_col):
        table = pa.Table.from_pandas(df)
        return pq.write_to_dataset(table,
                                   root_path=str(location),
                                   partition_cols=[partition_col],
                                   )
