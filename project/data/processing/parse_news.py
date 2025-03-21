import argparse
import prep_news

#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--in-file', action='store', dest='in_file',
                    help='news file', required=True)
parser.add_argument('--out-dir', action='store', dest='out_dir',
                    help='parsed/pre-processed content dir', required=True)
parser.add_argument('--mode', action='store', dest='mode',
                    help='train or test', required=True)
parser.add_argument('--max-title', action='store', dest='max_title',
                    help='max title length', default=20)
parser.add_argument('--max-abstract', action='store', dest='max_abstract',
                    help='max abstract length', default=50)
parser.add_argument('--word-embeddings', action='store', dest='word_embeddings',
                    help='pre-trained word embeddings', required=True)
parser.add_argument('--word2int', action='store', dest='word2int',
                    help='word to idx map')
parser.add_argument('--embedding-weights', action='store', dest='embedding_weights',
                    help='word embedding weights')
parser.add_argument('--category2int', action='store', dest='category2int',
                    help='category to idx map')
args = parser.parse_args()


# prep embedings/vocab
embeddings = prep_news.process_word_embeddings(args.word_embeddings)

prep_news.prep_news(args, embeddings)