import logging
import argparse
from utils.gpu_utils import test_gpu
from vqa_logger import init_log, test_log
# logger = init_log(__name__)
init_log()
logger = logging.getLogger(__name__)

def main():
    '''
    Using this main entry point so loggers will be using the main logger settings & for running frmo shell
    :return:
    '''
    func_dicts = {
        # 'flow':main_flow,
        'gpu':test_gpu,

    }
    parser = argparse.ArgumentParser(description='')
    help_txt = 'The main function to call. Expecting: {0}'.format("/".join([str(k) for k in list(func_dicts.keys())]))
    parser.add_argument('-f', dest='func', help=help_txt)
    args = parser.parse_args()

    func = func_dicts[args.func]
    func()


def ad_hock_main():
    from tests.test_model import test_model_training
    test_model_training()

    # from vqa_flow.imaging_device_classifier import retrain_model
    # retrain_model()

    return


if __name__ == '__main__':
    try:
        ad_hock_main()
        # main()
    except Exception as e:
        logger.error("Got an error:\n{0}".format(e))
        raise
        # sys.exit(1)
