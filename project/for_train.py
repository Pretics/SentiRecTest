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
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Callback, Trainer

from models.lstur import LSTUR
from models.nrms import NRMS
from models.naml import NAML
from models.naml_simple import NAML_Simple
from models.sentirec import SENTIREC
from models.robust_sentirec import ROBUST_SENTIREC

from data.dataset import BaseDataset

@dataclass
class TrainArgs():
    config: str
    resume: str

def load_model_config(args: TrainArgs):
    with open(args.config, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)
        config = DotMap(config)

    assert(config.name in ["lstur", "nrms", "naml", "naml_simple", "sentirec", "robust_sentirec"])
    pl.seed_everything(1234)
    return config

def create_checkpoint_callback(config: DotMap):
    checkpoint_callback = ModelCheckpoint(
        **config.checkpoint
    )
    return checkpoint_callback

def create_earlystop_callback(config):
    early_stop_callback = EarlyStopping(
       **config.early_stop
    )
    return early_stop_callback

def create_logger(config: DotMap):
    logger = TensorBoardLogger(**config.logger)
    return logger

def create_dataloader(config, behavior_path, news_path, config_loader):
    dataset = BaseDataset(behavior_path, news_path, config)
    loader = DataLoader(
        dataset,
        **config_loader)
    return dataset, loader

def create_train_dataloader(config):
    train_dataset, train_loader = create_dataloader(
        config,
        path.join(config.preprocess_data_dir, config.train_behavior),
        path.join(config.preprocess_data_dir, config.train_news),
        config.train_dataloader
    )
    return train_dataset, train_loader

def create_val_dataloader(config):
    val_dataset, val_loader = create_dataloader(
        config,
        path.join(config.preprocess_data_dir, config.val_behavior),
        path.join(config.preprocess_data_dir, config.train_news),
        config.val_dataloader
    )
    return val_dataset, val_loader

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

def create_model(config: DotMap, pretrained_word_embedding: Tensor):
    if config.name == "lstur":
        model = LSTUR(config, pretrained_word_embedding)
    elif config.name == "nrms":
        model = NRMS(config, pretrained_word_embedding)
    elif config.name == "naml":
        model = NAML(config, pretrained_word_embedding)
    elif config.name == "naml_simple":
        model = NAML_Simple(config, pretrained_word_embedding)
    elif config.name == "sentirec":
        model = SENTIREC(config, pretrained_word_embedding)
    elif config.name == "robust_sentirec":
        model = ROBUST_SENTIREC(config, pretrained_word_embedding)
    return model

def create_trainer(
        config: DotMap,
        callbacks: Union[list[Callback], Callback, None],
        logger: TensorBoardLogger
    ):
    trainer = Trainer(
        **config.trainer,
        callbacks=callbacks,
        logger=logger
    )
    return trainer

def create_train_trainer(config: DotMap):
    # init callbacks & logging
    checkpoint_callback = create_checkpoint_callback(config)
    early_stop_callback = create_earlystop_callback(config)
    logger = create_logger(config)
    # init trainer
    trainer = create_trainer(
        config=config,
        callbacks=[early_stop_callback, checkpoint_callback],
        logger=logger
    )
    return trainer

def start_train(
        args: TrainArgs,
        model: Union[LSTUR, NRMS, NAML, NAML_Simple, SENTIREC, ROBUST_SENTIREC],
        trainer: Trainer,
        train_loader: DataLoader,
        val_loader: DataLoader
    ):
    trainer.fit(
        model=model, 
        train_dataloaders=train_loader, 
        val_dataloaders=val_loader,
        ckpt_path=args.resume
    )
    
def cli_main(args):
    # configs
    config = load_model_config(args)
    
    # load data
    train_dataset, train_loader = create_train_dataloader(config)
    val_dataset, val_loader = create_val_dataloader(config)
    pretrained_word_embedding = load_embedding_weights(config)

    # init model
    model = create_model(config, pretrained_word_embedding)

    # init trainer
    trainer = create_train_trainer(config)

    # start training
    start_train(model, trainer, train_loader, val_loader, args.resume)