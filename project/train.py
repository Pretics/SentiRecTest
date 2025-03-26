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
        '--resume',
        action='store',
        dest='resume_ckpt_path',
        help='resume training from ckpt',
        required=False)
    args = parser.parse_args()
    args = ManagerArgs(**vars(args))
    return args

if __name__ == '__main__':
    args = get_args()
    train_manager = ModelManager(args)
    train_manager.fit()