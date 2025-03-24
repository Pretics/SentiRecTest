import argparse
import for_test

def get_args():
    # ------------
    # args -> config
    # ------------
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
    args = for_test.TestArgs(**vars(args))
    return args

if __name__ == '__main__':
    args = get_args()
    for_test.cli_main(args)