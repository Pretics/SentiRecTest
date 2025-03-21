import argparse
import prep_news_combined

#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--train-file', action='store', dest='train_news_path',
                    help='train news file', required=True)
parser.add_argument('--test-file', action='store', dest='test_news_path',
                    help='test news file', required=True)
parser.add_argument('--train-out', action='store', dest='train_out_dir',
                    help='parsed/pre-processed content dir', required=True)
parser.add_argument('--test-out', action='store', dest='test_out_dir',
                    help='parsed/pre-processed content dir', required=True)
parser.add_argument('--word-embeddings', action='store', dest='word_embeddings',
                    help='pre-trained word embeddings', required=True)
parser.add_argument('--max-title', action='store', dest='max_title',
                    help='max title length', default=20)
parser.add_argument('--max-abstract', action='store', dest='max_abstract',
                    help='max abstract length', default=50)
args = parser.parse_args()


# prep embedings/vocab
embeddings = prep_news_combined.process_word_embeddings(args.word_embeddings)

prep_news_combined.prep_news_combined(args, embeddings)