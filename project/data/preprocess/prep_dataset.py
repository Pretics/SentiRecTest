import os
from os import path
PROJECT_DIR = path.abspath(path.join(os.getcwd(), "../.."))

from dataclasses import dataclass
from .prep_behavior import PrepBehavior, PrepBehaviorArgs
from .prep_news import PrepNews, PrepNewsArgs

@dataclass
class PrepDatasetArgs:
    """
    for behaviors)
    split_test_size, n_negative <br/>
    for news)
    max_title, max_abstract
    """
    size: str
    split_test_size: float
    n_negative: int
    max_title: int
    max_abstract: int

class PrepDataset:
    args: PrepDatasetArgs
    behavior_args: PrepBehaviorArgs
    news_args: PrepNewsArgs

    def __init__(self, args: PrepDatasetArgs):
        self.args = args

        """
        경로가 폴더를 나타낼 경우 Dir, 파일일 경우 Path로 명명

        size: 전처리를 진행할 데이터셋의 크기 (demo, small, large 등)
        """
        size = args.size

        DATA_DIR = path.join(PROJECT_DIR, "data")
        datasetDir = path.join(DATA_DIR, "MIND", size)
        trainBehaviorsPath = path.join(datasetDir, "train", "behaviors.tsv")
        testBehaviorsPath = path.join(datasetDir, "test", "behaviors.tsv")
        trainNewsPath = path.join(datasetDir, "train", "news.tsv")
        testNewsPath = path.join(datasetDir, "test", "news.tsv")

        processedDataDir = path.join(DATA_DIR, "preprocessed_data")
        preTrainDir = path.join(processedDataDir, size, "train")
        preTestDir = path.join(processedDataDir, size, "test")

        wordEmbeddingDir = path.join(DATA_DIR, "word_embeddings")
        wordEmbeddingPath = path.join(wordEmbeddingDir, "glove.840B.300d.txt")
        wordEmbeddingNpyPath = path.join(wordEmbeddingDir, "glove.840B.300d.npy")
        wordEmbeddingTokensPath = path.join(wordEmbeddingDir, "glove.840B.300d.tokens.tsv")

        self.behavior_args = PrepBehaviorArgs(
            trainBehaviorsPath,
            testBehaviorsPath,
            preTrainDir,
            preTestDir,
            f"{preTrainDir}/user2int.tsv",
            args.split_test_size,
            args.n_negative
        )
        self.news_args = PrepNewsArgs(
            trainNewsPath,
            testNewsPath,
            preTrainDir,
            preTestDir,
            wordEmbeddingPath,
            wordEmbeddingNpyPath,
            wordEmbeddingTokensPath,
            args.max_title,
            args.max_abstract
        )
        os.makedirs(preTrainDir, exist_ok=True)
        os.makedirs(preTestDir, exist_ok=True)

        self.prepare_behavior = PrepBehavior(self.behavior_args)
        self.prepare_news = PrepNews(self.news_args)

    def pre_processing_dataset(self):
        self.prepare_behavior.prep_behavior()
        self.prepare_news.prep_news()