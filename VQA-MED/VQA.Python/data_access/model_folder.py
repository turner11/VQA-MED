import math
import logging
import numpy as np
import matplotlib.pyplot as plt
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
        folder_str = str(self.folder).replace('\\', '\\\\')
        return f'{self.__class__.__name__}(folder="{folder_str}")'


class ModelFolder(ModelFolderStructure):
    """"""

    def __init__(self, folder):
        """"""
        super().__init__(folder)
        self.additional_info = File.load_json(str(self.additional_info_path))
        self.prediction_data_name = self.additional_info['prediction_data']
        self.question_category = self.additional_info.get('question_category')

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

            shutil.copy(str(meta_data_location), str(folder_structure.meta_data_path))
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
        prediction_data_name = self.prediction_data_name
        ret = DataAccess.get_prediction_data(meta, prediction_data_name, self.question_category)
        return ret

    @property
    def history(self):
        return File.load_pickle(self.history_path, read_mode='rb')

    def load_model(self) -> Model:  # object:#keras.engine.training.Model:
        with VerboseTimer("Loading Model"):
            model = keras_load_model(str(self.model_path),
                                     custom_objects={'f1_score': f1_score,
                                                     'recall_score': recall_score,
                                                     'precision_score': precision_score})
        return model

    def plot(self, block=True, title='', metric=None):
        return self.plot_history(self.history, block=block, title=title, metric=metric)

    @staticmethod
    def plot_from_path(path, block=True, title='', metric=None):
        mf = ModelFolder(path)
        return mf.plot(block=block, title=title, metric=metric)

    @staticmethod
    def plot_history(history, block=True, title='', metric=None):

        min_val = min([min(v) for k, v in history.items() if len(v) > 0])
        max_val = max([max(v) for k, v in history.items() if len(v) > 0])

        use_canonical_ticks = min_val - max_val > 0.05
        if use_canonical_ticks:
            major_ticks = np.arange(0., 1.1, 0.1)
            minor_ticks = np.arange(0., 1.1, 0.05)
        else:
            major_ticks = [np.inf]
            minor_ticks = [np.inf]


        # plt.gcf().clear()
        idx = -1
        plot_items = [k for k, v in history.items() if 'val' in k and len(v) > 0 and max(v) < max(major_ticks)]

        if metric is not None:
            plot_items = [k for k in plot_items if metric in k]

        if not plot_items:
            logger.warning('Have no metrics to plot')
            return



        if len(plot_items) > 1:
            title_key = "val_acc"
            nrows = 2
            ncols = math.ceil(len(plot_items) / 2)

            share_y = 'rows' if use_canonical_ticks else False
            fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharey=share_y)  # ,sharex='col'
            for row in ax:
                try:
                    iterator = iter(row)
                except TypeError:
                    # not iterable, if there is only 1
                    row = [row]
                for col in row:
                    idx += 1
                    if idx >= len(plot_items):
                        continue

                    curr_metric = plot_items[idx]
                    name = curr_metric
                    vals = history[curr_metric]
                    x = list(range(1, len(vals) + 1))
                    col.plot(x, vals, color='blue', alpha=0.8, label=name)

                    non_val_key = name.replace('val_', '')
                    if non_val_key in history:
                        vals_train = history[non_val_key]
                        x_train = list(range(1, len(vals_train) + 1))
                        col.plot(x_train, vals_train, color='green', alpha=0.8)



                    if not use_canonical_ticks:
                        max_val = max(vals)
                        min_val = min(vals)
                        if max_val == min_val:
                            major_ticks = [max_val - 0.1, max_val, max_val + 0.1]
                            minor_ticks = []
                        else:
                            major_step = (max_val - min_val) / 10.0
                            major_ticks = np.arange(min_val, max_val + major_step, major_step)
                            minor_ticks = []#np.arange(min_val, max_val, major_step / 2.0)
                    else:
                        min_val, max_val = [0.5, 1.0]



                    col.set_yticks(major_ticks)
                    col.set_yticks(minor_ticks, minor=True)

                    col.set(ylabel=name, title=f"{name} / epochs ", ylim=[min_val , 1])  # set_xlim =[0, 5]
                    col.set_ylim([min_val, max_val])

            #     fig.legend(loc='upper left')
            #     fig.legend(loc='best')
        else:
            fig = plt.figure()
            key = plot_items[0]
            vals = history[key]
            x = list(range(1, len(vals) + 1))
            plt.plot(x, vals)

            non_val_key = key.replace('val_', '')
            if non_val_key in history:
                vals_train = history[non_val_key]
                x_train = list(range(1, len(vals_train) + 1))
                plt.plot(x_train, vals_train, color='green', alpha=0.8)

            title_key = key

        max_vals = [-1]
        epoch_i = -1
        try:
            title_vals = history[title_key]
            epoch_i = np.argmax(title_vals)
            max_vals = title_vals#[epoch_i]
            max_val = title_vals[epoch_i]
            title = title or f'"{title_key}" Best epoch = {epoch_i+1} ({max_val:.3f})'
        except Exception as ex:
            print(ex)

        fig.suptitle(str(title), fontsize=20)

        plt.show(block=block)
        return max_vals, epoch_i



def main():
    folder = "C:\\Users\\Public\\Documents\\Data\\2019\\models\\20190423_1229_10_Abnormality_trained"
    folder = 'C:\\Users\\Public\\Documents\\Data\\2019\\models\\20190422_1546_38_Abnormality_yes_no_trained\\'
    mmf = ModelFolder(folder)
    mmf.plot()




if __name__ == '__main__':
    main()
