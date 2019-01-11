import graphviz
import pydot
import datetime
import os
import time
from keras.utils import plot_model
import numpy as np
import pandas as pd
from common.os_utils import File
import logging
logger = logging.getLogger(__name__)
from keras import Model


def _get_time_stamp():
    now = time.time()
    ts = datetime.datetime.fromtimestamp(now).strftime('%Y%m%d_%H%M_%S')
    return ts

def _print_model_summary_to_file(fn, model):
    # Open the file
    with open(fn, 'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))


def save_model(model, base_folder, name_suffix="", history=None):
    ts = _get_time_stamp()

    now_folder = os.path.abspath('{0}\\{1}\\'.format(base_folder, ts))
    model_name = f'vqa_model_{name_suffix }.h5'
    model_fn = os.path.join(now_folder, model_name)
    model_image_fn = os.path.join(now_folder, 'model_vqa.png')
    summary_fn = os.path.join(now_folder, 'model_summary.txt')
    history_fn = os.path.join(now_folder, 'model_history.pkl')
    logger.debug("saving model to: '{0}'".format(model_fn))

    fn_image = os.path.join(now_folder, 'model.png')
    logger.debug(f"saving model image to {fn_image}")

    try:
        File.validate_dir_exists(now_folder)
        model.save(model_fn)  # creates a HDF5 file 'my_model.h5'
        logger.debug("model saved")
        location_message = f"model_location = '{model_fn}'"

    except Exception as ex:
        location_message = "Failed to save model:\n{0}".format(ex)
        logger.error(location_message)
        model_fn=""
        raise

    try:
        logger.debug("Writing Symmary")
        _print_model_summary_to_file(summary_fn, model)
        logger.debug("Done Writing Summary")

        logger.debug("Saving image")
        plot_model(model, to_file=fn_image)
        logger.debug(f"Image saved ('{fn_image}')")
    #     logger.debug("Plotting model")
    #     plot_model(model, to_file=model_image_fn)
    #     logger.debug("Done Plotting")
    except Exception as ex:
        logger.warning("{0}".format(ex))

    history_res_path = ''
    if history is not None:
        try:
            logger.debug("Saving History")
            File.dump_pickle(history.history,history_fn)
            logger.debug("History saved to '{0}'".format(history_fn))
            history_res_path = history_fn
        except Exception as ex:
            logger.warning("Failed to write history:\n\t{0}".format(ex))


    return model_fn, summary_fn, fn_image, history_res_path


def get_trainable_params_distribution(model: Model, params_threshold: int = 1000) -> pd.DataFrame:
    from keras import backend as K

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
