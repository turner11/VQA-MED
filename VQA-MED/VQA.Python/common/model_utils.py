import datetime
import os
import time
from pathlib import Path

from keras.utils import plot_model
import numpy as np
import pandas as pd
from common.os_utils import File
import logging
from keras import Model, backend as K
import warnings
from keras.callbacks import Callback

from data_access.model_folder import ModelFolder

logger = logging.getLogger(__name__)


def _get_time_stamp():
    now = time.time()
    ts = datetime.datetime.fromtimestamp(now).strftime('%Y%m%d_%H%M_%S')
    return ts

def save_model(model, base_folder, additional_info, meta_data_location, history=None):
    ts = _get_time_stamp()
    now_folder = Path(str(base_folder)) / ts

    model_folder = ModelFolder.create(now_folder, model, additional_info, meta_data_location, history)
    return model_folder



def get_trainable_params_distribution(model: Model, params_threshold: int = 1000) -> pd.DataFrame:
    names_and_trainable_params = {(w.name, np.prod(K.get_value(w).shape)) for w in model.trainable_weights}
    a = {'layer': [tpl[0] for tpl in names_and_trainable_params],
         'trainable_params': [tpl[1] for tpl in names_and_trainable_params]
         }
    df = pd.DataFrame.from_dict(a)
    df_sorted = df.sort_values(['trainable_params'], ascending=[False]).reset_index()
    df_sorted['pretty_value'] = df_sorted.apply(lambda x: "{:,}".format(x['trainable_params']), axis=1)
    top = df_sorted[df_sorted.trainable_params > params_threshold]
    print(f'Got a total of {"{:,}".format(sum(df_sorted.trainable_params))} trainable parameters')
    return top


class EarlyStoppingByAccuracy(Callback):
    def __init__(self, monitor='accuracy', value=0.98, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn(f"Early stopping requires {self.monitor} available!", RuntimeWarning)

        if current is None or self.value is None:
            pass
        elif current >= self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True
def main():
    pass
    # from common import DAL
    # from keras.models import load_model
    # from evaluate.statistical import f1_score, recall_score, precision_score
    # model_location = 'C:\\Users\\Public\\Documents\\Data\\2018\\vqa_models\\20180831_1244_55\\vqa_model_.h5'
    # model = load_model(model_location,
    #                    custom_objects={'f1_score': f1_score,
    #                                    'recall_score': recall_score,
    #                                    'precision_score': precision_score})
    #
    # DAL.insert_models(model)


if __name__ == '__main__':
    main()
