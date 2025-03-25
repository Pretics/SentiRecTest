from dataclasses import dataclass
from typing import Literal, Optional

@dataclass
class CheckpointConfig:
    dirpath: str
    filename: str
    save_top_k: int
    verbose: bool
    monitor: str
    mode: str
    save_last: bool

@dataclass
class LoggerConfig:
    save_dir: str
    name: str
    version: str

@dataclass
class EarlyStopConfig:
    monitor: str
    min_delta: float
    patience: int
    strict: bool
    verbose: bool
    mode: str

@dataclass
class DataLoaderConfig:
    num_workers: int
    persistent_workers: bool
    batch_size: int
    drop_last: bool

@dataclass
class TrainerConfig:
    max_epochs: int
    devices: int
    accelerator: str
    strategy: str
    fast_dev_run: bool

@dataclass
class BaseConfig:
    name: str
    dataset_size: str
    preprocess_data_dir: str
    seed: int
    train_behavior: str
    val_behavior: str
    test_behavior: str
    train_news: str
    test_news: str
    user2int: str
    word2int: str
    category2int: str
    embedding_weights: str
    max_history: int
    num_words_title: int
    num_words_abstract: int
    num_categories: int
    learning_rate: float
    dropout_probability: float
    query_vector_dim: int
    num_attention_heads: int
    word_embedding_dim: int
    freeze_word_embeddings: bool
    checkpoint: CheckpointConfig
    logger: LoggerConfig
    early_stop: EarlyStopConfig
    train_dataloader: DataLoaderConfig
    val_dataloader: DataLoaderConfig
    test_dataloader: DataLoaderConfig
    trainer: TrainerConfig
    find_unused_parameters: Optional[bool]
    sentiment_regularization: Optional[bool] = None
    sentiment_classifier: Optional[Literal['vader_sentiment', 'bert_sentiment']] = None
    sentiment_prediction_loss_coeff: Optional[float] = None
    sentiment_diversity_loss_coeff: Optional[float] = None