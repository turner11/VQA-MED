import argparse

from utils.gpu_utils import test_gpu
from vqa_flow.main import main as main_flow


def main():
    func_dicts = {
        'flow':main_flow,
        'gpu':test_gpu,

    }
    parser = argparse.ArgumentParser(description='')
    help_txt = 'The main function to call. Expecting: {0}'.format("/".join([str(k) for k in list(func_dicts.keys())]))
    parser.add_argument('-f', dest='func', help=help_txt)
    args = parser.parse_args()

    func = func_dicts[args.func]
    func()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print("Got an error:\n{0}".format(e))
        raise
        # sys.exit(1)





