import argparse
import for_train

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
    args = for_train.TrainArgs(**vars(args))
    return args

if __name__ == '__main__':
    args = get_args()
    for_train.cli_main(args)