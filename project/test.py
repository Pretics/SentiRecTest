import argparse
from utils.test_manager import TestManager, TestArgs

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        action='store',
        dest='config',
        help='config.yaml',
        required=True)
    parser.add_argument(
        '--ckpt',
        action='store',
        dest='ckpt',
        help='checkpoint to load',
        required=True)
    args = parser.parse_args()
    args = TestArgs(**vars(args))
    return args

if __name__ == '__main__':
    args = get_args()
    test_manager = TestManager(args)
    test_manager.test()