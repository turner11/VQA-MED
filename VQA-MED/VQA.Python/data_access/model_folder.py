import logging
import shutil
from pathlib import Path
from keras import Model
from keras.callbacks import History
from keras.models import load_model as keras_load_model
from keras.utils import plot_model
from common.os_utils import File
from common.settings import data_access
from common.utils import VerboseTimer
from data_access.api import DataAccess
from evaluate.statistical import f1_score, recall_score, precision_score

logger = logging.getLogger(__name__)


class ModelFolderStructure(object):
    """"""

    ADDITIONAL_INFO_FILE_NAME = 'additional_info.json'
    META_DATA_FILE_NAME = 'meta_data.h5'
    MODEL_FILE_NAME = 'vqa_model.h5'
    HISTORY_FILE_NAME = 'model_history.pkl'
    MODEL_SUMMARY_FILE_NAME = 'model_summary.txt'
    IMAGE_FILE_NAME = 'model.png'

    def __init__(self, folder):
        """"""
        super().__init__()
        self.folder = Path(str(folder))

    @property
    def meta_data_path(self):
        return self.folder / self.META_DATA_FILE_NAME

    @property
    def model_path(self):
        return self.folder / self.MODEL_FILE_NAME

    @property
    def image_file_path(self):
        return self.folder / self.IMAGE_FILE_NAME

    @property
    def summary_path(self):
        return self.folder / self.MODEL_SUMMARY_FILE_NAME

    @property
    def summary(self):
        try:
            smmry = File.read_text(self.summary_path)
        except Exception as ex:
            smmry = f'Failed to read summary:\n{ex}'
        return smmry

    @property
    def history_path(self):
        return self.folder / self.HISTORY_FILE_NAME

    @property
    def additional_info_path(self):
        return self.folder / self.ADDITIONAL_INFO_FILE_NAME

    def __repr__(self):
        return f'{self.__class__.__name__}(folder={str(self.folder)})'


class ModelFolder(ModelFolderStructure):
    """"""

    def __init__(self, folder):
        """"""
        super().__init__(folder)
        self.additional_info = File.load_json(str(self.additional_info_path))
        self.prediction_data_name = self.additional_info['prediction_data']

        assert self.folder.exists()

    @staticmethod
    def create(folder: str, model: Model, additional_info: dict, meta_data_location: str,
               history: History = None) -> object:
        folder_structure = ModelFolderStructure(folder)

        # will allow failure for all items except model itself

        try:
            model_fn = str(folder_structure.model_path)
            File.validate_dir_exists(folder_structure.folder)
            model.save(model_fn)  # creates a HDF5 file 'vqa_model.h5'
            logger.debug("model saved")

            logger.debug("saving prediction vector")
            File.dump_json(additional_info, folder_structure.additional_info_path)
            logger.debug("saved prediction vector")

            shutil.copy(str(data_access.fn_meta), str(folder_structure.meta_data_path))
        except Exception as ex:
            location_message = "Failed to save model:\n{0}".format(ex)
            logger.error(location_message)
            raise

        try:
            logger.debug("Writing Summary")
            summary_fn = str(folder_structure.summary_path)

            with open(summary_fn, 'w') as fh:
                # Pass the file handle in as a lambda function to make it callable
                model.summary(print_fn=lambda x: fh.write(x + '\n'))

            logger.debug("Done Writing Summary")

            fn_image = str(folder_structure.image_file_path)
            logger.debug("Saving image")
            plot_model(model, to_file=fn_image)
            logger.debug(f"Image saved ('{fn_image}')")
        except Exception as ex:
            logger.warning("{0}".format(ex))

        if history is not None:
            try:
                history_fn = str(folder_structure.history_path)
                logger.debug("Saving History")
                File.dump_pickle(history.history, history_fn)
                logger.debug("History saved to '{0}'".format(history_fn))
            except Exception as ex:
                logger.warning("Failed to write history:\n\t{0}".format(ex))

        return ModelFolder(folder_structure.folder)

    @property
    def prediction_vector(self):
        meta = DataAccess.load_meta_from_location(self.meta_data_path)
        vector = meta[self.prediction_data_name]
        assert len(vector.columns) == 1, 'Expected to get a single vector for prediction'
        ret = vector[vector.columns[0]].drop_duplicates().reset_index(drop=True)
        return ret

    @property
    def history(self):
        return File.load_pickle(self.history_path)

    def load_model(self):
        with VerboseTimer("Loading Model"):
            model = keras_load_model(str(self.model_path),
                                     custom_objects={'f1_score': f1_score,
                                                     'recall_score': recall_score,
                                                     'precision_score': precision_score})
        return model
