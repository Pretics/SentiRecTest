from os import path
from dataclasses import dataclass
from typing import Union
from tqdm import tqdm
from dotmap import DotMap

import numpy as np
import yaml

import torch
from torch import Tensor
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer

from models.lstur import LSTUR
from models.nrms import NRMS
from models.naml import NAML
from models.naml_simple import NAML_Simple
from models.sentirec import SENTIREC
from models.robust_sentirec import ROBUST_SENTIREC

from data.dataset import BaseDataset

@dataclass
class TestArgs():
    config: str
    ckpt: str

def load_model_config(args: TestArgs):
    with open(args.config, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)
        config = DotMap(config)

    assert(config.name in ["lstur", "nrms", "naml", "naml_simple", "sentirec", "robust_sentirec"])
    pl.seed_everything(1234)
    return config

def create_logger(config: DotMap):
    logger = TensorBoardLogger(**config.logger)
    return logger

def create_dataloader(config: DotMap):
    test_dataset = BaseDataset(
        path.join(config.preprocess_data_dir, config.test_behavior),
        path.join(config.preprocess_data_dir, config.test_news), 
        config
    )
    test_loader = DataLoader(
        test_dataset,
        **config.test_dataloader
    )
    return test_dataset, test_loader

def load_embedding_weights(config: DotMap):
    # load embedding pre-trained embedding weights
    embedding_weights=[]
    with open(path.join(config.preprocess_data_dir, config.embedding_weights), 'r') as file: 
        lines = file.readlines()
        for line in tqdm(lines):
            weights = [float(w) for w in line.split(" ")]
            embedding_weights.append(weights)
    pretrained_word_embedding = torch.from_numpy(
        np.array(embedding_weights, dtype=np.float32)
    )
    return pretrained_word_embedding

def load_model_from_checkpoint(args: TestArgs, config: DotMap, pretrained_word_embedding: Tensor):
    if config.name == "lstur":
        model = LSTUR.load_from_checkpoint(
            args.ckpt, 
            config=config, 
            pretrained_word_embedding=pretrained_word_embedding
        )
    elif config.name == "nrms":
        model = NRMS.load_from_checkpoint(
            args.ckpt, 
            config=config, 
            pretrained_word_embedding=pretrained_word_embedding
        )
    elif config.name == "naml":
        model = NAML.load_from_checkpoint(
            args.ckpt, 
            config=config, 
            pretrained_word_embedding=pretrained_word_embedding
        )
    elif config.name == "naml_simple":
        model = NAML_Simple.load_from_checkpoint(
            args.ckpt, 
            config=config, 
            pretrained_word_embedding=pretrained_word_embedding
        )
    elif config.name == "sentirec":
        model = SENTIREC.load_from_checkpoint(
            args.ckpt, 
            config=config, 
            pretrained_word_embedding=pretrained_word_embedding
        )
    elif config.name == "robust_sentirec":
        model = ROBUST_SENTIREC.load_from_checkpoint(
            args.ckpt, 
            config=config, 
            pretrained_word_embedding=pretrained_word_embedding
        )
    return model

def create_trainer(config: DotMap, logger: TensorBoardLogger):
    trainer = Trainer(
        **config.trainer,
        logger=logger
    )
    return trainer

def start_test(
        trainer: Trainer,
        model: Union[LSTUR, NRMS, NAML, NAML_Simple, SENTIREC, ROBUST_SENTIREC],
        test_loader: DataLoader
    ) -> list[dict[str, float]]:
    test_result = trainer.test(
        model=model, 
        dataloaders=test_loader
    )
    return test_result

def cli_main(args: TestArgs):
    # config
    config = load_model_config(args)

    # logging
    logger = create_logger(config)

    # load data
    test_dataset, test_loader = create_dataloader(config)
    pretrained_word_embedding = load_embedding_weights(config)
   
    # init model
    model = load_model_from_checkpoint(args, config, pretrained_word_embedding)

    # Test
    trainer = create_trainer(config, logger)
    test_result = start_test(trainer, model, test_loader)