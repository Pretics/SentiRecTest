import yaml
from dotmap import DotMap
from os import path
import numpy as np
import torch
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
from tqdm import tqdm
from dataclasses import dataclass

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

def create_dataloader(config, preprocess_data_dir):
    test_dataset = BaseDataset(
        path.join(preprocess_data_dir + config.test_behavior),
        path.join(preprocess_data_dir + config.test_news), 
        config)
    test_loader = DataLoader(
        test_dataset,
        **config.test_dataloader)
    
    return test_loader

def load_embedding_weights(config, preprocess_data_dir):
    # load embedding pre-trained embedding weights
    embedding_weights=[]
    with open(path.join(preprocess_data_dir + config.embedding_weights), 'r') as file: 
        lines = file.readlines()
        for line in tqdm(lines):
            weights = [float(w) for w in line.split(" ")]
            embedding_weights.append(weights)
    pretrained_word_embedding = torch.from_numpy(
        np.array(embedding_weights, dtype=np.float32)
        )
    
    return pretrained_word_embedding

def load_model_from_checkpoint(args: TestArgs, config, pretrained_word_embedding):
    if config.name == "lstur":
        model = LSTUR.load_from_checkpoint(
            args.ckpt, 
            config=config, 
            pretrained_word_embedding=pretrained_word_embedding)
    elif config.name == "nrms":
        model = NRMS.load_from_checkpoint(
            args.ckpt, 
            config=config, 
            pretrained_word_embedding=pretrained_word_embedding)
    elif config.name == "naml":
        model = NAML.load_from_checkpoint(
            args.ckpt, 
            config=config, 
            pretrained_word_embedding=pretrained_word_embedding)
    elif config.name == "naml_simple":
        model = NAML_Simple.load_from_checkpoint(
            args.ckpt, 
            config=config, 
            pretrained_word_embedding=pretrained_word_embedding)
    elif config.name == "sentirec":
        model = SENTIREC.load_from_checkpoint(
            args.ckpt, 
            config=config, 
            pretrained_word_embedding=pretrained_word_embedding)
    elif config.name == "robust_sentirec":
        model = ROBUST_SENTIREC.load_from_checkpoint(
            args.ckpt, 
            config=config, 
            pretrained_word_embedding=pretrained_word_embedding)
        
    return model

def cli_main(args: TestArgs):
    config = load_model_config(args)

    # ------------
    # logging
    # ------------
    logger = TensorBoardLogger(
        **config.logger
    )

    # ------------
    # data
    # ------------
    preprocess_data_dir = f"{config.preprocess_data_path}/{config.dataset_size}/"
    test_loader = create_dataloader(config, preprocess_data_dir)
   
    # ------------
    # init model
    # ------------
    pretrained_word_embedding = load_embedding_weights(config, preprocess_data_dir)
    model = load_model_from_checkpoint(args, config, pretrained_word_embedding)

    # ------------
    # Test
    # ------------
    trainer = Trainer(
        **config.trainer,
        logger=logger
    )

    trainer.test(
        model=model, 
        dataloaders=test_loader
    )