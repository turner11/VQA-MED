import logging
from pathlib import Path
from keras.models import load_model as keras_load_model
from common.os_utils import File
from data_access.api import DataAccess
from evaluate.statistical import f1_score, recall_score, precision_score
from common.utils import VerboseTimer

logger = logging.getLogger(__name__)


class ModelFolder(object):
    """"""
    ADDITIONAL_INFO_FILE_NAME = 'additional_info.json'
    META_DATA_FILE_NAME = 'meta_data.h5'
    MODEL_FILE_NAME = 'vqa_model_.h5'

    def __init__(self, folder):
        """"""
        super().__init__()
        self.folder = Path(str(folder))
        self.additional_info = File.load_json(str(self.folder / self.ADDITIONAL_INFO_FILE_NAME))
        self.prediction_data_name = self.additional_info['prediction_data']

        assert self.folder.exists()

    def __repr__(self):
        return super().__repr__()

    @property
    def meta_data_path(self):
        return self.folder / self.META_DATA_FILE_NAME
    
    @property
    def model_path(self):
        return self.folder / self.MODEL_FILE_NAME

    @property
    def prediction_vector(self):
        meta = DataAccess.load_meta_from_location(self.meta_data_path)
        vector = meta[self.prediction_data_name]
        assert len(vector.columns) == 1, 'Expected to get a single vector for prediction'
        ret = vector[vector.columns[0]].drop_duplicates().reset_index(drop=True)
        return ret

    def load_model(self):
        model = keras_load_model(str(self.model_path),
                           custom_objects={'f1_score': f1_score,
                                           'recall_score': recall_score,
                                           'precision_score': precision_score})
        return model



