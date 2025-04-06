from dataclasses import dataclass, is_dataclass, fields
from typing import Any, Literal, Optional, Type, Union, get_origin, get_args
import yaml

def from_dict(cls: Type, data: dict) -> Any:
    """
    재귀적으로 dict를 dataclass로 변환 (dataclass 안의 dataclass도 dict가 아니라 dataclass로 취급하기 위함)
    """
    if not is_dataclass(cls):
        return data

    kwargs = {}
    for field in fields(cls):
        field_value = data.get(field.name)
        field_type = field.type

        origin = get_origin(field_type)
        args = get_args(field_type)

        # Optional[T] 처리
        if origin is Union and type(None) in args:
            actual_type = [arg for arg in args if arg is not type(None)][0]
            if is_dataclass(actual_type) and isinstance(field_value, dict):
                kwargs[field.name] = from_dict(actual_type, field_value)
            else:
                kwargs[field.name] = field_value
        elif is_dataclass(field_type) and isinstance(field_value, dict):
            kwargs[field.name] = from_dict(field_type, field_value)
        else:
            kwargs[field.name] = field_value

    return cls(**kwargs)


def load_config_from_yaml(path: str, cls: Type) -> Any:
    with open(path, "r") as f:
        config_dict = yaml.safe_load(f)
    return from_dict(cls, config_dict)

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
class ModelConfig:
    name: str
    dataset_size: str
    project_dir: Union[str, None]
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