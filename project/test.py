import argparse
from utils.model_manager import ModelManager, ManagerArgs

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
        dest='test_ckpt_path',
        help='checkpoint to load',
        required=True)
    args = parser.parse_args()
    args = ManagerArgs(**vars(args))
    return args

if __name__ == '__main__':
    args = get_args()
    test_manager = ModelManager(args)
    test_manager.test()