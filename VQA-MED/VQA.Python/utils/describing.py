import os
from glob import glob

from tensorflow.python.framework.errors_impl import InternalError

from pre_processing.known_find_and_replace_items import test_data
import logging
logger = logging.getLogger(__name__)


def describe_models(models_folder):
    import keras
    MODEL_SUFFIX = '_model.h5'
    clean_folder = (models_folder + os.sep).replace("\\\\",'\\')
    pattern = clean_folder + '*' + MODEL_SUFFIX
    files = glob(pattern)
    results = {}
    errors = {}
    for fn in files:
        tag = fn.replace(clean_folder,'').rstrip(MODEL_SUFFIX)
        logger.debug('--== evaluating model for tag "{0}" ==--'.format(tag))
        try:
            model = keras.models.load_model(fn)
            features, labels = get_tag_data(test_data, tag=tag)

            batch_size = None
            score, acc = model.evaluate(features, labels,
                                        batch_size=batch_size)

            logger.debug('Test accuracy and score for tag "{0}": ({1},{2})'.format(tag, acc ,score))

            results[tag] = acc
        except InternalError as tesor_ex:
            msg = "Error:\n{0}".format(tesor_ex)
            if tesor_ex.error_code == 13:
                # https://stackoverflow.com/questions/37313818/tensorflow-dst-tensor-is-not-initialized#answer-40389498
                # https://github.com/aymericdamien/TensorFlow-Examples/issues/38
                msg += "\nGPU Memory is full.\n" \
                       "The batch size might to large"
            errors[tag] = msg
        except Exception as ex:
            errors[tag] = "Error:\n{0}".format(ex)

    if errors:
        errors_str = "\n".join(["{0}: {1}".format(tag, err) for tag, err in errors.items()])
        logger.error(errors_str)
    if results:
        results_s = sorted([(tag, acc) for tag, acc in results.items()], key=lambda tpl: tpl[1], reverse=True)
        acc_str = "\n".join(["{0}: {1}".format(tag, acc) for tag, acc in results_s])
        logger.info(acc_str)


def main():
    models_folder = 'C:\\Users\\Public\\Documents\\Data\\2018\\models'
    describe_models(models_folder)
    
   
   
if __name__ == '__main__':
    main()
    