# import logging
# import argparse
#
#
# from utils.gpu_utils import test_gpu
# from vqa_logger import init_log, test_log
# # logger = init_log(__name__)
# init_log()
# logger = logging.getLogger(__name__)
#
# def main():
#     '''
#     Using this main entry point so loggers will be using the main logger settings & for running frmo shell
#     :return:
#     '''
#     func_dicts = {
#         # 'flow':main_flow,
#         'gpu':test_gpu,
#
#     }
#     parser = argparse.ArgumentParser(description='')
#     help_txt = 'The main function to call. Expecting: {0}'.format("/".join([str(k) for k in list(func_dicts.keys())]))
#     parser.add_argument('-f', dest='func', help=help_txt)
#     args = parser.parse_args()
#
#     func = func_dicts[args.func]
#     func()
#
#
# def ad_hock_main():
#     from tests.test_model import test_model_predicting, data_frame
#     df = data_frame()
#     test_model_predicting(df)
#
#     # from vqa_flow.imaging_device_classifier import retrain_model
#     # retrain_model()
#
#     return
#
# # This code install line by line a list of pip package



def install(package):

    import pip
    from pip._internal import main as pip_main #(pip>=18)
    pip_main(['install', package]) # pip_main(['install', package]) (pip>=18)





if __name__ == '__main__':
    import cv2
    if False:
        import sys
        from tqdm import tqdm
        with open(sys.argv[1]) as f:
            lines = [l.strip() for l in f.readlines()]
        pbar = tqdm(lines )
        for line in pbar:
            pbar.set_description(f'working on {line}')
            install(line)

    if False:
        try:
            ad_hock_main()
            # main()
        except Exception as e:
            logger.error("Got an error:\n{0}".format(e))
            raise
            # sys.exit(1)
