from os import path
from tqdm import tqdm
import csv
from nltk.tokenize import word_tokenize
import csv
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dataclasses import dataclass

@dataclass
class PrepNewsArgs:
    train_news_path: str
    test_news_path: str
    train_out_dir: str
    test_out_dir: str
    word_embedding_path: str
    max_title: int
    max_abstract: int

# generate word2int + extract embedding weights
def process_word_embeddings(word_embeddings_file):
    with open(word_embeddings_file, 'r', encoding='utf-8') as wf:
        print("preparing/processing word-embeddings") 
        word_embeddings = wf.readlines()
        embeddings_map = {}
        for word_embedding in tqdm(word_embeddings):
            wdims = word_embedding.split(" ")
            embeddings_map[wdims[0]] = " ".join(wdims[1:])
        # np.save(path.join(out_dir, "embedding_weights.npy"), np.array(weights, dtype=np.float32))
        # with open(path.join(out_dir, 'word2int.tsv'), 'w', newline='') as file:  
        #    word_writer = csv.writer(file, delimiter='\t')
        #    for key, value in word2int.items():
        #        word_writer.writerow([key, value])
        return embeddings_map

def load_idx_map_as_dict(file_name):
    with open(file_name, 'r', encoding='utf-8') as file:
        dictionary = {}
        lines = file.readlines()
        for line in tqdm(lines):
            key, value = line.strip().split("\t")
            dictionary[key] = value
        return dictionary

def load_embedding_weights(file_name):
    embedding_weights = []
    with open(file_name, 'r', encoding='utf-8') as file: 
        lines = file.readlines()
        for line in tqdm(lines):
            embedding_weights.append(line)
        return embedding_weights

def calc_sentiment_scores(title, vader_classifier, bert_classifier):  
    # vader
    vs = vader_classifier.polarity_scores(title.strip())
    vader_sentiment = vs['compound']

    # bert
    dsbs_label, dsbs_score = bert_classifier(title.strip())[0].values()
    if(dsbs_label == "POSITIVE"):
        bert_sentiment = (1-dsbs_score)*(-1) + dsbs_score
    else:
        bert_sentiment = (dsbs_score)*(-1) + (1-dsbs_score)

    return vader_sentiment, bert_sentiment

def process_sentence(
        sentence: str, 
        embeddings: dict,
        word2int: dict,
        embedding_weights: list,
        max_sentence_length: int
    ):
    """
    문장을 토큰 단위로 자르고, word2int에 중복 없이 등록합니다.
    또한 word2int와 동일한 순서로 해당 토큰에 대한 임베딩 가중치를
    embeddings로부터 가져와서 embedding_weights에 저장합니다.
    """
    tokens = word_tokenize(sentence.strip().lower())
    word_idxs = []
    for token in tokens:
        if token not in embeddings:
            continue
        if token not in word2int:
            word2int[token] = str(len(word2int) + 1)
            embedding_weights.append(embeddings[token])
        word_idxs.append(word2int[token])

    """
    문장의 총 토큰 갯수를 max_sentence_length에 맞춥니다.
    너무 길면 뒷부분을 길이에 맞게 자르고,
    너무 짧으면 앞부분에 0을 추가합니다.
    """
    if len(word_idxs) > max_sentence_length:
        word_idxs = word_idxs[:max_sentence_length]
    else:
        word_idxs = word_idxs + ["0"]*(max_sentence_length-len(word_idxs))
    word_idxs_str = " ".join(word_idxs)

    return word_idxs_str

def process_news_dataset(
        args,
        news_dataset_path,
        embeddings: dict,
        category2int: dict,
        word2int: dict,
        embedding_weights: list
    ):
    processed_news = []
    news2int = {}

    with open(news_dataset_path, 'r', encoding='utf-8') as in_file:
        news_collection = in_file.readlines()

    # sentiment analyzer
    dsb_sentiment_classifier = pipeline('sentiment-analysis')
    vader_sentiment_classifier = SentimentIntensityAnalyzer()

    max_title_length = int(args.max_title)
    max_abstract_length = int(args.max_abstract)
        
    # iterate over news
    for news in tqdm(news_collection):
        newsid, category, subcategory, title, abstract, _, _, _ = news.strip().split("\t")
        if newsid not in news2int:
            news2int[newsid] = len(news2int) + 1

        # category to int
        if category not in category2int:
            category2int[category] = len(category2int) + 1
        if subcategory not in category2int:
            category2int[subcategory] = len(category2int) + 1
        category_id = category2int[category]
        subcategory_id = category2int[subcategory]

        # change title, abstract to indexes
        title_word_idxs_str = process_sentence(title, embeddings, word2int, embedding_weights, max_title_length)
        abstract_word_idxs_str = process_sentence(abstract, embeddings, word2int, embedding_weights, max_abstract_length)

        # calc sentiments scores
        vader_sentiment, bert_sentiment = calc_sentiment_scores(title, vader_sentiment_classifier, dsb_sentiment_classifier)

        # prepare output
        processed_news.append([
            newsid,
            category_id,
            subcategory_id,
            title_word_idxs_str,
            abstract_word_idxs_str,
            vader_sentiment,
            bert_sentiment
        ])
    return processed_news

def save_n2int(n2int, filename, out_dir):
    with open(path.join(out_dir, filename), 'w', encoding='utf-8', newline='') as file:
        word_writer = csv.writer(file, delimiter='\t')
        for key, value in n2int.items():
            word_writer.writerow([key, value])

def save_train_data(
        args,
        category2int: dict,
        word2int: dict,
        embedding_weights: list
    ):
    save_n2int(category2int, 'category2int.tsv', args.train_out_dir)
    save_n2int(word2int, 'word2int.tsv', args.train_out_dir)
    with open(path.join(args.train_out_dir, 'embedding_weights.csv'), 'w', encoding='utf-8', newline='') as file:
        # 첫 줄에 index=0(빈칸) 패딩용 0 0 0 ... 0 가중치 추가
        file.write(" ".join(["0.0"] * 300) + "\n")
        for weights in embedding_weights:
            file.write(weights)

def save_test_data(
        args,
        word2int: dict,
        embedding_weights: list
    ):
    save_n2int(word2int, 'word2int.tsv', args.test_out_dir)
    with open(path.join(args.test_out_dir, 'embedding_weights.csv'), 'w', encoding='utf-8', newline='') as file:
        # 첫 줄에 index=0(빈칸) 패딩용 0 0 0 ... 0 가중치 추가
        file.write(" ".join(["0.0"] * 300) + "\n")
        for weights in embedding_weights:
            file.write(weights)

def prep_news_combined(args, embeddings: dict):
    """
    0. 데이터셋의 news.tsv를 불러옵니다.
    1. category2int, word2int, embedding_weights를 불러오거나 생성합니다.
    2. news.tsv 모든 뉴스 데이터에 대해 전처리 작업을 진행합니다.
    3. 생성한 전처리 데이터를 parsed_news.tsv에 저장합니다.
    4. category2int, word2int, embedding_weights를 저장합니다.
    """
    category2int = {}
    word2int = {}
    embedding_weights = []
    
    # Train 데이터셋 전처리
    with open(path.join(args.train_out_dir, 'parsed_news.tsv'), 'w', newline='') as train_news_file:
        news_writer = csv.writer(train_news_file, delimiter='\t')
        print("preparing/processing train news content")

        # prepare output
        processed_newsdata = process_news_dataset(args, args.train_news_path, embeddings, category2int, word2int, embedding_weights)
        news_writer.writerows(processed_newsdata)

    save_train_data(args, category2int, word2int, embedding_weights)

    # Test 데이터셋 전처리
    with open(path.join(args.test_out_dir, 'parsed_news.tsv'), 'w', newline='') as test_news_file:
        news_writer = csv.writer(test_news_file, delimiter='\t')
        print("preparing/processing test news content")

        # prepare output
        processed_newsdata = process_news_dataset(args, args.test_news_path, embeddings, category2int, word2int, embedding_weights)
        news_writer.writerows(processed_newsdata)

    save_test_data(args, word2int, embedding_weights)