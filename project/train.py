import argparse
from train_manager import TrainManager, TrainArgs

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
        dest='resume',
        help='resume training from ckpt',
        required=False)
    args = parser.parse_args()
    args = TrainArgs(**vars(args))
    return args

if __name__ == '__main__':
    args = get_args()
    train_manager = TrainManager(args)
    train_manager.fit()