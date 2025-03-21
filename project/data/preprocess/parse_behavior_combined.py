import argparse
import prep_behavior_combined

#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--train-file', action='store', dest='train_behavior_path',
                    help='behaviour file', required=True)
parser.add_argument('--test-file', action='store', dest='test_behavior_path',
                    help='behaviour file', required=True)
parser.add_argument('--train-out', action='store', dest='train_out_dir',
                    help='parsed/pre-processed behaviour file dir', required=True)
parser.add_argument('--test-out', action='store', dest='test_out_dir',
                    help='parsed/pre-processed behaviour file dir', required=True)
parser.add_argument('--user2int', action='store', dest='user2int_path',
                    help='user index map', required=True)
parser.add_argument('--split', action='store', dest='split_test_size',
                    help='train/val split ratio', default=0.1)
parser.add_argument('---n-negative-samples', action='store', dest='n_negative',
                    help='number of negative samples per positive sample', default=4)

args = parser.parse_args()

prep_behavior_combined.prep_behavior_combined(args)