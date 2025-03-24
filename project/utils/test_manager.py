from os import path
from dataclasses import dataclass
from typing import Union
from tqdm import tqdm
from dotmap import DotMap

import numpy as np
import yaml

from torch import Tensor, from_numpy
from torch.utils.data import DataLoader

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer, LightningModule, seed_everything

from models.lstur import LSTUR
from models.nrms import NRMS
from models.naml import NAML
from models.naml_simple import NAML_Simple
from models.sentirec import SENTIREC
from models.robust_sentirec import ROBUST_SENTIREC

from data.dataset import BaseDataset

@dataclass
class TestArgs():
    """
    ``config``: config파일 경로 <br/>
    ``ckpt``: 테스트를 진행할 모델의 ckpt파일 경로
    """
    config: str
    ckpt: str

class TestManager:
    def __init__(self, args: Union[TestArgs, None]):
        if args is not None:
            self.update_by_args(args)
        else:
            self.args = None
            self.config = None
            self.test_dataset = None
            self.test_loader = None
            self.pretrained_word_embedding = None
            self.model = None
            self.trainer = None

    def update_by_args(self, args: TestArgs):
        # configs
        self.args = args
        self.config = self.load_model_config(args)

        # logging
        self.logger = self.create_logger(self.config)

        # load data
        self.test_dataset, self.test_loader = self.create_dataloader(self.config)
        self.pretrained_word_embedding = self.load_embedding_weights(self.config)
    
        # init model
        self.model = self.load_model_from_checkpoint(args, self.config, self.pretrained_word_embedding)

        # init trainer
        self.trainer = self.create_trainer(self.config, self.logger)

    def load_model_config(self, args: TestArgs):
        with open(args.config, 'r') as ymlfile:
            config = yaml.load(ymlfile, Loader=yaml.FullLoader)
            config = DotMap(config)
        assert(config.name in ["lstur", "nrms", "naml", "naml_simple", "sentirec", "robust_sentirec"])
        seed_everything(config.seed)
        return config

    def create_logger(self, config: DotMap):
        logger = TensorBoardLogger(**config.logger)
        return logger
    
    def create_dataloader(self, config, behavior_path, news_path, config_loader):
        dataset = BaseDataset(behavior_path, news_path, config)
        loader = DataLoader(
            dataset,
            **config_loader)
        return dataset, loader

    def create_test_dataloader(self, config: DotMap):
        test_dataset, test_loader = self.create_dataloader(
            config,
            path.join(config.preprocess_data_dir, config.test_behavior),
            path.join(config.preprocess_data_dir, config.test_news),
            config.test_dataloader
        )
        return test_dataset, test_loader

    def load_embedding_weights(self, config: DotMap):
        # load embedding pre-trained embedding weights
        embedding_weights=[]
        with open(path.join(config.preprocess_data_dir, config.embedding_weights), 'r') as file: 
            lines = file.readlines()
            for line in tqdm(lines):
                weights = [float(w) for w in line.split(" ")]
                embedding_weights.append(weights)
        pretrained_word_embedding = from_numpy(
            np.array(embedding_weights, dtype=np.float32)
        )
        return pretrained_word_embedding

    def load_model_from_checkpoint(self, args: TestArgs, config: DotMap, pretrained_word_embedding: Tensor):
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

    def create_trainer(self, config: DotMap, logger: TensorBoardLogger):
        trainer = Trainer(
            **config.trainer,
            logger=logger
        )
        return trainer

    def start_test(
            self, 
            trainer: Trainer,
            model: LightningModule,
            test_loader: DataLoader
        ) -> list[dict[str, float]]:
        test_result = trainer.test(
            model=model, 
            dataloaders=test_loader
        )
        return test_result
    
    def test(self):
        self.test_result = self.start_test(self.trainer, self.model, self.test_loader)
        return self.test_result
    