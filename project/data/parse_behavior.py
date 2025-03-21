import argparse
import prep_behavior

#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--in-file', action='store', dest='in_file',
                    help='behaviour file', required=True)
parser.add_argument('--out-dir', action='store', dest='out_dir',
                    help='parsed/pre-processed behaviour file dir', required=True)
parser.add_argument('--mode', action='store', dest='mode',
                    help='train or test', required=True)
parser.add_argument('--user2int', action='store', dest='user2int',
                    help='user index map')
parser.add_argument('--split', action='store', dest='split',
                    help='train/val split', default=0.1)
parser.add_argument('---n-negative-samples', action='store', dest='n_negative',
                    help='number of negative samples per positive sample', default=4)

args = parser.parse_args()

prep_behavior.prep_behavior(args)