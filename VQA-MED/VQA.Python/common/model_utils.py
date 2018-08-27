import datetime
import graphviz
import os
import pydot
import time
from keras.utils import plot_model

from common.os_utils import File
from vqa_logger import logger


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
    if history:
        try:
            logger.debug("Saving History")
            File.dump_pickle(history,history_fn)
            logger.debug("History saved to '{0}'".format(history_fn))
            history_res_path = history_fn
        except Exception as ex:
            logger.warning("Failed to write history:\n{0}".format(ex))


    return model_fn, summary_fn, fn_image, history_res_path