import os
from os import path

PROJECT_DIR = path.abspath(path.join(os.getcwd(), "..", ".."))

import pandas as pd

class DatasetFrameLoader:
    def __init__(self, dataset_size: str = "demo", is_dev: bool = False):
        self.is_dev = is_dev
        self.dataset_size = dataset_size
        self.dataset_dir = path.join(PROJECT_DIR, "data", "MIND", dataset_size)
        self.behaviors_header = ["imp_id", "user_id", "time", "history", "impressions"]
        self.news_header = ["news_id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"]

        self.train_behaviors_path = path.join(self.dataset_dir, "train", "behaviors.tsv")
        self.train_news_path = path.join(self.dataset_dir, "train", "news.tsv")

        self.test_behaviors_path = path.join(self.dataset_dir, "test", "behaviors.tsv")
        self.test_news_path = path.join(self.dataset_dir, "test", "news.tsv")

        if is_dev:
            self.dev_behaviors_path = path.join(self.dataset_dir, "dev", "behaviors.tsv")
            self.dev_news_path = path.join(self.dataset_dir, "dev", "news.tsv")

    def load_behaviors_df(self, version: str):
        """
        `version`: train / test / dev
        """
        assert version == "train" or version == "test" or version == "dev"

        if version == "train":
            self.train_behaviors_df = pd.read_csv(
                self.train_behaviors_path,
                sep='\t',
                header=None,
                names=["imp_id", "user_id", "time", "history", "impressions"]
            )
        elif version == "test":
            self.test_behaviors_df = pd.read_csv(
                self.test_behaviors_path,
                sep='\t',
                header=None,
                names=["imp_id", "user_id", "time", "history", "impressions"]
            )
        elif version == "dev" and self.is_dev:
            self.dev_behaviors_df = pd.read_csv(
                self.dev_behaviors_path,
                sep='\t',
                header=None,
                names=["imp_id", "user_id", "time", "history", "impressions"]
            )

    def load_news_df(self, version):
        """
        `version`: train / test / dev
        """
        assert version == "train" or version == "test" or version == "dev"

        if version == "train":
            self.train_news_df = pd.read_csv(
                self.train_news_path,
                sep='\t',
                header=None,
                names=["news_id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"]
            )
        elif version == "test":
            self.test_news_df = pd.read_csv(
                self.test_news_path,
                sep='\t',
                header=None,
                names=["news_id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"]
            )
        elif version == "dev" and self.is_dev:
            self.dev_news_df = pd.read_csv(
                self.dev_news_path,
                sep='\t',
                header=None,
                names=["news_id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"]
            )

    def load_behaviors_all(self):
        print(f"loading all behaviors dataset... (version: {self.dataset_size})")
        self.load_behaviors_df("train")
        self.load_behaviors_df("test")
        if self.is_dev:
            self.load_behaviors_df("dev")
        print("loading complete.")
    
    def load_news_all(self):
        print(f"loading all news dataset... (version: {self.dataset_size})")
        self.load_news_df("train")
        self.load_news_df("test")
        if self.is_dev:
            self.load_news_df("dev")
        print("loading complete.")

    def load_train_all(self):
        print(f"loading all train dataset... (version: {self.dataset_size})")
        self.load_behaviors_df("train")
        self.load_news_df("train")
        print("loading complete.")
    
    def load_test_all(self):
        print(f"loading all test dataset... (version: {self.dataset_size})")
        self.load_behaviors_df("test")
        self.load_news_df("test")
        print("loading complete.")

    def load_dev_all(self):
        print(f"loading all dev dataset... (version: {self.dataset_size})")
        self.load_behaviors_df("dev")
        self.load_news_df("dev")
        print("loading complete.")

    def load_all(self):
        self.load_train_all()
        self.load_test_all()
        if self.is_dev:
            self.load_dev_all()